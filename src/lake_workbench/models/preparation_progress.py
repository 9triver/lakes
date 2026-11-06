"""Request-scoped progress reporting without shared global task state."""
from contextlib import contextmanager
from contextvars import ContextVar

_reporter = ContextVar('comparison_preparation_reporter', default=None)


def report(stage, message, completed=None, total=None):
    callback = _reporter.get()
    if callback is not None:
        callback(stage=stage, message=message, completed=completed, total=total)


@contextmanager
def reporting(callback):
    token = _reporter.set(callback)
    try:
        yield
    finally:
        _reporter.reset(token)
