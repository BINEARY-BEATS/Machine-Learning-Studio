"""Application entry point."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMessageBox


def main() -> int:
    from ml_studio.app.config import enable_pandas_copy_on_write, validate_startup_dependencies
    from ml_studio.app.container import create_container
    from ml_studio.app.logger import setup_logging
    from ml_studio.app.paths import LOGS_DIR, ensure_runtime_dirs
    from ml_studio.gui.main_window import MainWindow

    ensure_runtime_dirs()
    config = validate_startup_dependencies()
    enable_pandas_copy_on_write()
    setup_logging(LOGS_DIR, config.log_level)

    app = QApplication(sys.argv)
    app.setApplicationName(config.app_name)
    app.setApplicationVersion(config.app_version)
    app.setOrganizationName(config.organization)

    container = create_container(config)
    try:
        window = MainWindow(container)
        window.show()
    except Exception as e:
        QMessageBox.critical(None, "Startup Error", str(e))
        return 1

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
