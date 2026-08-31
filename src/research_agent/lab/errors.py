"""Split / leakage errors for the research lab."""


class LeakageError(ValueError):
    """A train-derived API was asked to use validation or test labels."""


class SealedSplitError(ValueError):
    """Test labels or test-derived fitness facts were requested."""
