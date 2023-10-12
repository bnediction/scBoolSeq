"""Custom exception types to yield more informative error messages. """


class NotASubsetOfExpectedColumnsError(ValueError):
    """A ValueError specific to misaligned columns"""
