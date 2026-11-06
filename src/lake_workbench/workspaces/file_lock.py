"""Reentrant thread/process locks shared by CLI writers and the web server."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import threading

_GUARD = threading.Lock()
_LOCKS = {}
_LOCAL = threading.local()


def _after_fork():
    global _GUARD, _LOCKS, _LOCAL
    for handle in getattr(_LOCAL, 'owned', {}).values():
        handle.close()
    _GUARD, _LOCKS, _LOCAL = threading.Lock(), {}, threading.local()


os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def process_lock(path, *, blocking=True):
    key = str(Path(path).resolve())
    with _GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    if not lock.acquire(blocking=blocking):
        raise BlockingIOError(f"已有任务正在使用锁：{key}")
    owned = getattr(_LOCAL, 'owned', None)
    if owned is None:
        owned = _LOCAL.owned = {}
    try:
        if key in owned:
            yield
            return
        Path(key).parent.mkdir(parents=True, exist_ok=True)
        with open(key, 'a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            owned[key] = handle
            try:
                yield
            finally:
                del owned[key]
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        lock.release()
