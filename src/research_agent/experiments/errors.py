"""Typed errors for the experiment harness."""


class SpecError(ValueError):
    """ExperimentSpec is malformed or violates origin/parent rules."""


class ForbiddenTestSplit(ValueError):
    """Test split used without an explicit opt-in."""


class RegistryError(ValueError):
    """Registry operation failed."""
