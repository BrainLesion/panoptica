import multiprocessing.pool
from multiprocessing import Pool, Process
from typing import Callable


class NoDaemonProcess(Process):
    def __init__(
        self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args=None,
        kwargs=None,
        *,
        daemon: bool | None = None,
    ) -> None:
        if kwargs is None:
            kwargs = {}
        if args is None:
            args = []
        super().__init__(None, target, name, args, kwargs, daemon=daemon)

    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NonDaemonicPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
