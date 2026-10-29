# Exception Handling for Stingray

__all__ = ["StingrayError", "StingrayWarning"]


class StingrayError(Exception):
    pass


class StingrayWarning(UserWarning):
    """Base class for the warnings Stingray raises itself.

    It subclasses ``UserWarning``, which is what those warnings were before they had
    a class of their own, so filters written against ``UserWarning`` keep working.
    Having a class of their own means they can be silenced on their own::

        import warnings
        from stingray.exceptions import StingrayWarning

        warnings.filterwarnings("ignore", category=StingrayWarning)
    """
