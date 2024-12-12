import multiprocessing.pool
from multiprocessing import Pool, Process
from typing import Callable


class NoDaemonProcess(Process):
    """A subclass of `multiprocessing.Process` that overrides daemon behavior to always be non-daemonic.

    Useful for creating a process that allows child processes to spawn their own children,
    as daemonic processes in Python cannot create further subprocesses.

    Attributes:
        group (None): Reserved for future extension when using process groups.
        target (Callable[..., object] | None): The callable object to be invoked by the process.
        name (str | None): The name of the process, for identification.
        args (tuple): Arguments to pass to the `target` function.
        kwargs (dict): Keyword arguments to pass to the `target` function.
        daemon (bool | None): Indicates if the process is daemonic (overridden to always be False).
    """

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
    """A version of `multiprocessing.pool.Pool` using non-daemonic processes, allowing child processes to spawn their own children.

    This class creates a pool of worker processes using `NoDaemonProcess` for situations where nested child processes are needed.
    """

    Process = NoDaemonProcess
