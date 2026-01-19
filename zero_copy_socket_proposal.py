"""
Zero-copy socket wrapper for ZMQ with automatic single/multi-node detection.

This implementation:
1. Automatically detects single-node vs multi-node scenarios based on ZMQ endpoint
2. Uses pickle protocol 5 with zero-copy for single-node (ipc:// and tcp://127.0.0.1)
3. Falls back to standard pickle for multi-node (tcp://<remote-ip>)
"""

import io
import pickle
from typing import Any, Optional
from urllib.parse import urlparse

import torch
import zmq


class TorchTensorPickler(pickle.Pickler):
    """Custom pickler that handles torch tensors efficiently."""
    dispatch_table = {}

    @classmethod
    def register(cls, type, reduce):
        cls.dispatch_table[type] = reduce


def torch_tensor_reducer(tensor):
    """
    Reducer function for torch tensors.
    Always moves tensor to CPU to ensure compatibility across machines.
    """
    assert isinstance(tensor, torch.Tensor)
    cpu_tensor = tensor.cpu()
    return torch.from_numpy, (cpu_tensor.numpy(),)


TorchTensorPickler.register(torch.Tensor, torch_tensor_reducer)


def is_local_endpoint(endpoint: str) -> bool:
    """
    Determine if a ZMQ endpoint is local (single-node) or remote (multi-node).

    Args:
        endpoint: ZMQ endpoint string (e.g., "ipc://...", "tcp://127.0.0.1:5555", "tcp://192.168.1.1:5555")

    Returns:
        True if endpoint is local (can use zero-copy), False otherwise

    Examples:
        >>> is_local_endpoint("ipc:///tmp/socket")
        True
        >>> is_local_endpoint("tcp://127.0.0.1:5555")
        True
        >>> is_local_endpoint("tcp://localhost:5555")
        True
        >>> is_local_endpoint("tcp://[::1]:5555")
        True
        >>> is_local_endpoint("tcp://*:5555")
        False  # Conservative: wildcard could accept remote connections
        >>> is_local_endpoint("tcp://192.168.1.1:5555")
        False
    """
    if not endpoint:
        return False

    # IPC sockets are always local
    if endpoint.startswith("ipc://") or endpoint.startswith("inproc://"):
        return True

    # Parse TCP endpoints
    if endpoint.startswith("tcp://"):
        try:
            # Handle tcp://* case (bind to all interfaces)
            if "tcp://*" in endpoint:
                # Conservative: treat as potentially multi-node
                return False

            # Extract host part (handle IPv6 in brackets)
            parsed = urlparse(endpoint)
            host = parsed.hostname  # This handles IPv6 brackets automatically

            if not host:
                return False

            # Check for loopback addresses
            loopback_hosts = {
                "localhost",
                "127.0.0.1",
                "::1",  # IPv6 loopback
                "0:0:0:0:0:0:0:1",  # IPv6 loopback expanded
            }

            # Normalize and check
            host_lower = host.lower()
            if host_lower in loopback_hosts:
                return True

            # Check if it starts with 127. (127.0.0.x range)
            if host.startswith("127."):
                return True

            return False

        except Exception:
            # If parsing fails, be conservative
            return False

    # Unknown protocol, be conservative
    return False


class ZeroCopySocket:
    """
    ZMQ socket wrapper with automatic zero-copy optimization.

    Automatically detects whether the connection is local or remote:
    - Local (ipc://, tcp://127.0.0.1, etc.): Uses pickle protocol 5 with zero-copy
    - Remote (tcp://<remote-ip>): Uses standard pickle (more compatible)

    Args:
        socket: ZeroMQ socket instance
        force_zerocopy: If True, always use zero-copy. If False, always use standard pickle.
                       If None (default), auto-detect based on endpoint.
    """

    def __init__(
        self,
        socket: zmq.Socket,
        force_zerocopy: Optional[bool] = None
    ):
        self._socket = socket
        self._force_zerocopy = force_zerocopy

        # Auto-detect if not forced
        if force_zerocopy is None:
            self._use_zerocopy = self._detect_zerocopy_capability()
        else:
            self._use_zerocopy = force_zerocopy

    def _detect_zerocopy_capability(self) -> bool:
        """
        Detect if zero-copy can be used based on socket endpoint.

        Tries multiple methods to determine the endpoint:
        1. Check LAST_ENDPOINT (for connected/bound sockets)
        2. Check getsockopt for various endpoint options
        3. Default to False (conservative)
        """
        try:
            # Try to get the last endpoint (works for both bind and connect)
            endpoint = self._socket.get_string(zmq.LAST_ENDPOINT, encoding='utf-8')
            if endpoint:
                return is_local_endpoint(endpoint)
        except zmq.ZMQError:
            pass

        # If we can't determine, be conservative
        # In practice, you could pass the endpoint explicitly during construction
        return False

    def set_endpoint_hint(self, endpoint: str):
        """
        Manually provide endpoint hint for zero-copy detection.

        This is useful when the socket hasn't been bound/connected yet,
        or when LAST_ENDPOINT is not available.

        Args:
            endpoint: ZMQ endpoint string
        """
        self._use_zerocopy = is_local_endpoint(endpoint)

    def send_pyobj(self, obj: Any, flags: int = 0, protocol: int = -1) -> None:
        """
        Send a Python object over ZMQ socket.

        Automatically uses zero-copy for local connections and standard pickle
        for remote connections.

        Args:
            obj: Python object to send
            flags: ZMQ flags
            protocol: Pickle protocol (default: -1, highest available)
        """
        if self._use_zerocopy:
            # Use pickle protocol 5 with zero-copy for local connections
            buffers = []

            def buffer_callback(buf):
                buffers.append(buf.raw())

            stream = io.BytesIO()
            pickler = TorchTensorPickler(
                stream,
                protocol=5,  # Protocol 5 supports out-of-band buffers
                buffer_callback=buffer_callback
            )
            pickler.dump(obj)
            main_bin = stream.getvalue()

            # Send main pickle + buffers as multipart message
            message_parts = [main_bin] + buffers
            self._socket.send_multipart(message_parts, flags=flags)
        else:
            # Use standard pickle for remote connections
            # Still use our custom pickler to handle torch tensors properly
            stream = io.BytesIO()
            pickler = TorchTensorPickler(stream, protocol=protocol)
            pickler.dump(obj)
            data = stream.getvalue()
            self._socket.send(data, flags=flags)

    def recv_pyobj(self, flags: int = 0) -> Any:
        """
        Receive a Python object from ZMQ socket.

        Automatically handles both zero-copy and standard pickle formats.

        Args:
            flags: ZMQ flags

        Returns:
            Deserialized Python object
        """
        if self._use_zerocopy:
            # Receive multipart message
            parts = self._socket.recv_multipart(flags=flags, copy=False)
            main_obj_bin = parts[0]
            buffers = [memoryview(p) for p in parts[1:]]
            return pickle.loads(main_obj_bin, buffers=buffers)
        else:
            # Receive single message
            data = self._socket.recv(flags=flags)
            return pickle.loads(data)

    def __getattr__(self, name: str) -> Any:
        """Proxy other attributes to the underlying socket."""
        return getattr(self._socket, name)


# Example usage patterns:

def example_automatic_detection():
    """Example: Automatic detection based on endpoint."""
    context = zmq.Context()

    # Local IPC socket - will use zero-copy
    socket1 = context.socket(zmq.PUSH)
    socket1.bind("ipc:///tmp/test_socket")
    wrapped1 = ZeroCopySocket(socket1)
    # wrapped1._use_zerocopy will be True

    # Local TCP socket - will use zero-copy
    socket2 = context.socket(zmq.PUSH)
    socket2.connect("tcp://127.0.0.1:5555")
    wrapped2 = ZeroCopySocket(socket2)
    # wrapped2._use_zerocopy will be True

    # Remote TCP socket - will NOT use zero-copy
    socket3 = context.socket(zmq.PUSH)
    socket3.connect("tcp://192.168.1.100:5555")
    wrapped3 = ZeroCopySocket(socket3)
    # wrapped3._use_zerocopy will be False


def example_with_endpoint_hint():
    """Example: Provide endpoint hint for early detection."""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)

    # Create wrapper before binding/connecting
    wrapped = ZeroCopySocket(socket)

    # Provide endpoint hint
    endpoint = "ipc:///tmp/test_socket"
    wrapped.set_endpoint_hint(endpoint)
    socket.bind(endpoint)

    # Now wrapped._use_zerocopy is properly set


def example_force_mode():
    """Example: Explicitly force zero-copy on or off."""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:5555")

    # Force zero-copy OFF (e.g., for debugging)
    wrapped = ZeroCopySocket(socket, force_zerocopy=False)

    # Force zero-copy ON (e.g., you know it's single-node despite tcp://*)
    wrapped = ZeroCopySocket(socket, force_zerocopy=True)


# Integration with sglang's get_zmq_socket helper:

def get_zmq_socket_with_zerocopy(
    context: zmq.Context,
    socket_type: int,
    endpoint: str,
    is_bind: bool = False,
) -> ZeroCopySocket:
    """
    Helper function to create and configure a ZMQ socket with zero-copy wrapper.

    This can be used as a drop-in replacement for sglang's get_zmq_socket.

    Args:
        context: ZeroMQ context
        socket_type: Type of ZeroMQ socket (zmq.PUSH, zmq.PULL, etc.)
        endpoint: ZMQ endpoint string
        is_bind: Whether to bind (True) or connect (False)

    Returns:
        ZeroCopySocket wrapper
    """
    socket = context.socket(socket_type)

    # Configure socket (hwm, etc.) - copied from sglang's config_socket
    if socket_type in [zmq.PUSH, zmq.PULL, zmq.DEALER, zmq.ROUTER]:
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVHWM, 0)

    # Bind or connect
    if is_bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    # Wrap with zero-copy socket
    wrapped = ZeroCopySocket(socket)
    wrapped.set_endpoint_hint(endpoint)  # Ensure detection works

    return wrapped


if __name__ == "__main__":
    # Test the detection logic
    test_cases = [
        ("ipc:///tmp/socket", True),
        ("tcp://127.0.0.1:5555", True),
        ("tcp://localhost:5555", True),
        ("tcp://[::1]:5555", True),
        ("tcp://*:5555", False),
        ("tcp://192.168.1.100:5555", False),
        ("tcp://10.0.0.1:5555", False),
    ]

    print("Testing endpoint detection:")
    for endpoint, expected in test_cases:
        result = is_local_endpoint(endpoint)
        status = "✓" if result == expected else "✗"
        print(f"{status} {endpoint}: {result} (expected {expected})")
