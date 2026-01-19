"""Test endpoint detection logic without torch dependency."""
from urllib.parse import urlparse


def is_local_endpoint(endpoint: str) -> bool:
    """
    Determine if a ZMQ endpoint is local (single-node) or remote (multi-node).
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

        except Exception as e:
            print(f"Error parsing {endpoint}: {e}")
            # If parsing fails, be conservative
            return False

    # Unknown protocol, be conservative
    return False


if __name__ == "__main__":
    # Test the detection logic
    test_cases = [
        ("ipc:///tmp/socket", True, "IPC socket"),
        ("inproc://myqueue", True, "In-process socket"),
        ("tcp://127.0.0.1:5555", True, "Loopback IPv4"),
        ("tcp://localhost:5555", True, "Localhost"),
        ("tcp://[::1]:5555", True, "Loopback IPv6"),
        ("tcp://127.0.0.2:5555", True, "127.0.0.x range"),
        ("tcp://*:5555", False, "Wildcard bind"),
        ("tcp://0.0.0.0:5555", False, "All interfaces"),
        ("tcp://192.168.1.100:5555", False, "Private network"),
        ("tcp://10.0.0.1:5555", False, "Private network"),
        ("tcp://172.16.0.1:5555", False, "Private network"),
        ("tcp://8.8.8.8:5555", False, "Public IP"),
    ]

    print("Testing endpoint detection:")
    print("=" * 70)
    all_pass = True
    for endpoint, expected, description in test_cases:
        result = is_local_endpoint(endpoint)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        if result != expected:
            all_pass = False
        print(f"{status:8s} {endpoint:30s} -> {result:5} | {description}")

    print("=" * 70)
    if all_pass:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
