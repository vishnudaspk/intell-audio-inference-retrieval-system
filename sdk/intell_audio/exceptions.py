"""
Exceptions for the Intell Audio Python SDK.
"""

class IntellSDKError(Exception):
    """Base exception for all Intell Audio SDK operations."""
    pass


class APIConnectionError(IntellSDKError):
    """Raised when the client cannot connect to the backend server."""
    pass


class JobFailedError(IntellSDKError):
    """Raised when the processing job fails on the server."""
    pass


class JobTimeoutError(IntellSDKError):
    """Raised when processing exceeds the specified client timeout."""
    pass
