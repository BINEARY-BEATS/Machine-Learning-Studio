"""Background worker base using QObject + moveToThread pattern."""

from PyQt6.QtCore import QObject, QThread, pyqtSignal


class WorkerBase(QObject):
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._thread: QThread | None = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def run_in_thread(self) -> QThread:
        self._cancelled = False
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._execute)
        self._thread.finished.connect(self.deleteLater)
        self._thread.start()
        return self._thread

    def _execute(self) -> None:
        self.started.emit()
        try:
            result = self.do_work()
            if not self._cancelled:
                self.result.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
            if self._thread:
                self._thread.quit()

    def do_work(self):
        raise NotImplementedError
