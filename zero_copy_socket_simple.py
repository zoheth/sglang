"""
Simplified zero-copy socket implementation based on server args.

Instead of parsing endpoints, we directly check server configuration:
- Single node (no DP or single-node DP): Use zero-copy
- Multi-node DP: Use standard pickle
"""

import io
import pickle
from typing import Any, Optional

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


def should_use_zerocopy(server_args) -> bool:
    """
    Determine if zero-copy should be used based on server configuration.

    Zero-copy is beneficial only for single-node scenarios:
    1. Normal case (no DP): always single-node
    2. DP attention with nnodes==1: single-node DP
    3. DP attention with nnodes>1: multi-node (no zero-copy benefit)

    Args:
        server_args: ServerArgs instance with configuration

    Returns:
        True if zero-copy should be used, False otherwise
    """
    if not server_args.enable_dp_attention:
        # Normal case: single node, use zero-copy
        return True

    # DP attention enabled: check if single-node or multi-node
    if server_args.nnodes == 1 and server_args.dist_init_addr is None:
        # Single-node DP: use zero-copy
        return True

    # Multi-node DP: don't use zero-copy
    return False


class ZeroCopySocket:
    """
    ZMQ socket wrapper with configurable zero-copy optimization.

    Args:
        socket: ZeroMQ socket instance
        use_zerocopy: Whether to use zero-copy optimization
    """

    def __init__(self, socket: zmq.Socket, use_zerocopy: bool):
        self._socket = socket
        self._use_zerocopy = use_zerocopy

    def send_pyobj(self, obj: Any, flags: int = 0, protocol: int = -1) -> None:
        """
        Send a Python object over ZMQ socket.

        Args:
            obj: Python object to send
            flags: ZMQ flags
            protocol: Pickle protocol (default: -1, highest available)
        """
        if self._use_zerocopy:
            # Use pickle protocol 5 with zero-copy for single-node
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
            # Use standard pickle for multi-node
            stream = io.BytesIO()
            pickler = TorchTensorPickler(stream, protocol=protocol)
            pickler.dump(obj)
            data = stream.getvalue()
            self._socket.send(data, flags=flags)

    def recv_pyobj(self, flags: int = 0) -> Any:
        """
        Receive a Python object from ZMQ socket.

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


# =============================================================================
# Integration with SGLang
# =============================================================================

def create_zmq_socket(
    context: zmq.Context,
    socket_type: int,
    endpoint: str,
    is_bind: bool,
    server_args,
) -> ZeroCopySocket:
    """
    Create and configure a ZMQ socket with zero-copy wrapper.

    This is the main integration point for SGLang. Replace get_zmq_socket
    calls with this function.

    Args:
        context: ZeroMQ context
        socket_type: Type of ZeroMQ socket (zmq.PUSH, zmq.PULL, etc.)
        endpoint: ZMQ endpoint string
        is_bind: Whether to bind (True) or connect (False)
        server_args: ServerArgs instance

    Returns:
        ZeroCopySocket wrapper
    """
    # Create and configure socket
    socket = context.socket(socket_type)

    # Configure socket (from sglang's config_socket)
    if socket_type in [zmq.PUSH, zmq.PULL, zmq.DEALER, zmq.ROUTER]:
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVHWM, 0)

    # Bind or connect
    if is_bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    # Determine if zero-copy should be used
    use_zerocopy = should_use_zerocopy(server_args)

    # Wrap with zero-copy socket
    return ZeroCopySocket(socket, use_zerocopy=use_zerocopy)


# =============================================================================
# Modifying PortArgs (Option 1: Add flag to PortArgs)
# =============================================================================

def example_portargs_with_flag():
    """
    Example: Modify PortArgs to include use_zerocopy flag.

    In server_args.py, modify PortArgs.make() to:
    """
    from dataclasses import dataclass

    @dataclass
    class PortArgs:
        tokenizer_ipc_name: str
        scheduler_input_ipc_name: str
        detokenizer_ipc_name: str
        nccl_port: int
        rpc_ipc_name: str
        metrics_ipc_name: str
        use_zerocopy: bool  # NEW FIELD
        tokenizer_worker_ipc_name: Optional[str] = None

    # Then in PortArgs.make():
    # use_zerocopy = should_use_zerocopy(server_args)
    # return PortArgs(..., use_zerocopy=use_zerocopy)


# =============================================================================
# Integration Examples
# =============================================================================

def example_usage_in_scheduler():
    """
    Example: How to use in scheduler.py

    Before:
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, True
        )

    After (Option 1: Using server_args):
        self.recv_from_tokenizer = create_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, True, server_args
        )

    After (Option 2: Using flag in PortArgs):
        socket = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, True
        )
        self.recv_from_tokenizer = ZeroCopySocket(socket, port_args.use_zerocopy)
    """
    pass


def example_modify_get_zmq_socket():
    """
    Example: Modify get_zmq_socket in common.py

    Add server_args parameter to existing get_zmq_socket:

    def get_zmq_socket(
        context: zmq.Context,
        socket_type: int,
        endpoint: str,
        is_bind: bool,
        server_args=None,  # NEW PARAMETER (optional for backward compatibility)
    ):
        socket = context.socket(socket_type)
        config_socket(socket, socket_type)

        if is_bind:
            socket.bind(endpoint)
        else:
            socket.connect(endpoint)

        # Wrap with zero-copy if server_args provided
        if server_args is not None:
            use_zerocopy = should_use_zerocopy(server_args)
            return ZeroCopySocket(socket, use_zerocopy)

        return socket
    """
    pass


# =============================================================================
# Comparison with endpoint-based detection
# =============================================================================

def comparison():
    """
    Parameter-based vs Endpoint-based detection:

    Parameter-based (THIS APPROACH):
    ✓ Simpler logic
    ✓ No string parsing
    ✓ Directly maps to design intent
    ✓ Easier to understand and maintain
    ✓ No edge cases in parsing

    Endpoint-based (PREVIOUS APPROACH):
    ✓ Works without server_args
    ✓ More generic/reusable
    ✗ Requires string parsing
    ✗ Edge cases (tcp://*, ipv6, etc.)
    ✗ Indirect (infers intent from endpoint)

    For SGLang integration, parameter-based is clearly better!
    """
    pass


if __name__ == "__main__":
    # Mock ServerArgs for testing
    class MockServerArgs:
        def __init__(self, enable_dp_attention, nnodes, dist_init_addr):
            self.enable_dp_attention = enable_dp_attention
            self.nnodes = nnodes
            self.dist_init_addr = dist_init_addr

    test_cases = [
        (MockServerArgs(False, 1, None), True, "Normal single-node"),
        (MockServerArgs(True, 1, None), True, "Single-node DP"),
        (MockServerArgs(True, 2, "192.168.1.1:5555"), False, "Multi-node DP (nnodes>1)"),
        (MockServerArgs(True, 1, "192.168.1.1:5555"), False, "Multi-node DP (dist_init_addr set)"),
    ]

    print("Testing zero-copy decision logic:")
    print("=" * 80)
    for args, expected, description in test_cases:
        result = should_use_zerocopy(args)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"{status:8s} {description:30s} -> use_zerocopy={result:5} (expected {expected})")
    print("=" * 80)
