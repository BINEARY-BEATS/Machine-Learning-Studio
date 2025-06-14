import sys
import os
import logging

log_level = logging.INFO
log_format = '%(asctime)s - %(name)-28s - %(levelname)-8s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "machine_learning_studio.log")

logging.basicConfig(
    level=log_level,
    format=log_format,
    datefmt=date_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, mode='a')
    ]
)
logger = logging.getLogger(__name__)

try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import Qt, QCoreApplication
    logger.info("PyQt5 imported successfully.")
except ImportError as e:
    print(f"FATAL ERROR: PyQt5 is not installed or cannot be imported: {e}", file=sys.stderr)
    print("Please install PyQt5: pip install PyQt5", file=sys.stderr)
    if logger: logger.critical(f"PyQt5 import failed: {e}", exc_info=True)
    sys.exit(1)

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project root added to sys.path: {project_root}")


def main():
    logger.info("========================================================")
    logger.info("    Starting Machine Learning Studio Application      ")
    logger.info("                Developed by Saeed Ur Rehman          ")
    logger.info("                     DAE CIT - FYP                    ")
    logger.info("========================================================")

    QCoreApplication.setOrganizationName("SaeedUrRehmanFYP")
    QCoreApplication.setApplicationName("Machine Learning Studio")
    QCoreApplication.setApplicationVersion("1.0.0")

    if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    try:
        from gui.main_window import MainWindow
        logger.info("MainWindow imported successfully inside main().")
    except ImportError as e:
        logger.critical(f"Failed to import MainWindow inside main(): {e}", exc_info=True)
        QMessageBox.critical(None, "Application Startup Error", f"Could not import MainWindow: {e}\nCheck logs.")
        sys.exit(1)
    except Exception as e_mw_import:
        logger.critical(f"Unexpected error importing MainWindow: {e_mw_import}", exc_info=True)
        QMessageBox.critical(None, "Application Startup Error", f"Error importing main window: {e_mw_import}\nCheck logs.")
        sys.exit(1)

    try:
        main_window = MainWindow()
        logger.info("MainWindow instance created successfully.")
        main_window.show()
        logger.info("MainWindow shown to user.")
        exit_code = app.exec_()
        logger.info(f"Application event loop finished. Exiting with code {exit_code}.")
        sys.exit(exit_code)
    except AttributeError as ae:
        logger.critical(f"AttributeError during MainWindow instantiation or execution: {ae}", exc_info=True)
        QMessageBox.critical(None, "Attribute Error", f"A programming error occurred (AttributeError):\n{ae}\n\nPlease check the logs. This might be due to a missing method or an incorrect variable name in the UI classes.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred during application execution: {e}", exc_info=True)
        QMessageBox.critical(None, "Critical Application Error", f"Unexpected error: {e}\nApp will close.")
        sys.exit(1)

if __name__ == '__main__':
    main()