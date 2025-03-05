__all__ = [
    "InstellaError",
    "InstellaConfigurationError",
    "InstellaCliError",
    "InstellaEnvironmentError",
    "InstellaNetworkError",
    "InstellaCheckpointError",
]


class InstellaError(Exception):
    """
    Base class for all custom Instella exceptions.
    """


class InstellaConfigurationError(InstellaError):
    """
    An error with a configuration file.
    """


class InstellaCliError(InstellaError):
    """
    An error from incorrect CLI usage.
    """


class InstellaEnvironmentError(InstellaError):
    """
    An error from incorrect environment variables.
    """


class InstellaNetworkError(InstellaError):
    """
    An error with a network request.
    """


class InstellaCheckpointError(InstellaError):
    """
    An error occurred reading or writing from a checkpoint.
    """


class InstellaThreadError(Exception):
    """
    Raised when a thread fails.
    """
