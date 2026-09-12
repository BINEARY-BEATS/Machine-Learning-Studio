from PyQt6.QtWidgets import QProgressBar


class ProgressRing(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(True)
        self.setRange(0, 100)
