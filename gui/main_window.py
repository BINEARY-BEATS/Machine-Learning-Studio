import sys
import os
import logging

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QPushButton, QStackedWidget,
    QFileDialog, QLabel, QSplitter, QMessageBox, QProgressBar, QHBoxLayout,
    QFrame, QApplication
)
import pandas as pd
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPixmap,QShowEvent

logger = logging.getLogger(__name__)

try:
    from .dashboard import Dashboard
    from .data_viewer import DataViewer
    from .data_cleaner import DataCleaner
    from .model_trainer import ModelTrainer
    from .visualizer import Visualizer
    from utils.data_loader import DataLoaderThread
except ImportError as e:
    logger.critical(f"Error importing UI/Utility modules in MainWindow: {e}", exc_info=True)
    app_temp = QApplication.instance()
    if not app_temp: app_temp = QApplication(sys.argv)
    QMessageBox.critical(None, "Fatal Module Import Error",
                         f"Could not import necessary application modules: {e}\n\n"
                         "The application cannot start. Please check installation and paths.\n"
                         "See logs for more details.")
    sys.exit(1)


ICON_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icons")
APP_ICON = os.path.join(ICON_PATH_ROOT, "ai.png")

class MainWindow(QMainWindow):
    data_updated_signal = pyqtSignal(pd.DataFrame)

    def __init__(self):
        super().__init__()
        self.current_data_frame: pd.DataFrame | None = None
        self.active_button: QPushButton | None = None
        self.data_loader_thread: DataLoaderThread | None = None
        
        self._init_properties()
        self._init_ui_components() 
        self._connect_signals()
        self._apply_stylesheet()
        
        self.update_module_button_states() 
        if hasattr(self, 'dashboard_module'):
            self.content_stack.setCurrentWidget(self.dashboard_module)
            self._update_active_nav_button(self.dashboard_module)
        else:
            logger.error("Dashboard module failed to initialize. Cannot set initial view.")

    def _init_properties(self):
        self.setWindowTitle("Machine Learning Studio by Saeed Ur Rehman")
        if os.path.exists(APP_ICON):
            self.setWindowIcon(QIcon(APP_ICON))
        self.setGeometry(30, 30, 1550, 950)
        self.setMinimumSize(1100, 750)

    def _init_ui_components(self):
        main_splitter = QSplitter(Qt.Horizontal, self)
        main_splitter.setObjectName("MainSplitter")
        main_splitter.setHandleWidth(4)
        main_splitter.setChildrenCollapsible(False)
        self.setCentralWidget(main_splitter)

        self.sidebar_widget = self._create_sidebar()
        main_splitter.addWidget(self.sidebar_widget)

        self.content_stack = QStackedWidget(self)
        self.content_stack.setObjectName("MainContentStack")
        
        try:
            self.dashboard_module = Dashboard(self)
            self.data_viewer_module = DataViewer(self)
            self.data_cleaner_module = DataCleaner(parent_window=self)
            self.model_trainer_module = ModelTrainer(parent_window=self, data_viewer_instance=self.data_viewer_module)
            self.visualizer_module = Visualizer(parent_window=self)

            self.content_stack.addWidget(self.dashboard_module)
            self.content_stack.addWidget(self.data_viewer_module)
            self.content_stack.addWidget(self.data_cleaner_module)
            self.content_stack.addWidget(self.model_trainer_module)
            self.content_stack.addWidget(self.visualizer_module)
        except Exception as e:
            logger.critical(f"Failed to initialize one or more UI modules: {e}", exc_info=True)
            QMessageBox.critical(self, "Module Initialization Error", f"Could not initialize core application modules: {e}.\nThe application may not function correctly.")
            return 

        main_splitter.addWidget(self.content_stack)
        main_splitter.setSizes([250, self.width() - 250 - main_splitter.handleWidth()])
        main_splitter.setStretchFactor(0,0)
        main_splitter.setStretchFactor(1,1)


        self.status_bar = self.statusBar()
        self.status_bar.setObjectName("AppStatusBar")
        self.status_label = QLabel(f"Ready. Welcome to Machine Learning Studio by Saeed Ur Rehman! (DAE CIT - FYP)")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16); self.progress_bar.setTextVisible(False); self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.progress_bar, 0)

    def _create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 12, 10, 12)
        sidebar_layout.setSpacing(8)

        title_layout = QHBoxLayout()
        title_layout.setSpacing(8)
        if os.path.exists(APP_ICON):
            logo_label = QLabel()
            pixmap = QPixmap(APP_ICON)
            logo_label.setPixmap(pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            title_layout.addWidget(logo_label)
        
        app_title_label = QLabel("Machine Learning Studio") 
        app_title_label.setObjectName("AppTitleLabel")
        app_title_label.setWordWrap(False)
        app_title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_layout.addWidget(app_title_label, 1)
        sidebar_layout.addLayout(title_layout)
        
        sidebar_layout.addWidget(self._create_separator())

        self.nav_buttons: dict[str, QPushButton] = {} 
        buttons_config = [
            ("Dashboard", "home.png", self._show_dashboard, False), 
            ("Import Data", "import.png", self._import_data_action, False),
            ("View Data", "table.png", self._show_data_viewer, True), 
            ("Clean & Preprocess", "clean.png", self._show_data_cleaner, True),
            ("Train Models", "train.png", self._show_model_trainer, True),
            ("Visualize Data", "chart.png", self._show_visualizer, True)
        ]

        for text, icon_name, slot, requires_data in buttons_config:
            btn = QPushButton(f" {text}") 
            btn.setObjectName("NavButton")
            icon_full_path = os.path.join(ICON_PATH_ROOT, icon_name)
            if os.path.exists(icon_full_path): btn.setIcon(QIcon(icon_full_path))
            else: logger.warning(f"Sidebar icon not found: {icon_full_path}")
            btn.setIconSize(QSize(20, 20))
            btn.setCheckable(True); btn.setAutoExclusive(True); btn.clicked.connect(slot)
            btn.setProperty("requires_data", requires_data) 
            sidebar_layout.addWidget(btn)
            self.nav_buttons[text] = btn
        
        if "Dashboard" in self.nav_buttons:
            self.nav_buttons["Dashboard"].setChecked(True); self.active_button = self.nav_buttons["Dashboard"]

        sidebar_layout.addStretch(1) 
        sidebar_layout.addWidget(self._create_separator())
        version_label = QLabel(f"FYP Version (Saeed Ur Rehman)")
        version_label.setObjectName("VersionLabel")
        sidebar_layout.addWidget(version_label, 0, Qt.AlignBottom | Qt.AlignHCenter)
        return sidebar

    def _create_separator(self):
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Plain)
        line.setFixedHeight(1); line.setStyleSheet("background-color: #40576d;")
        return line

    def _connect_signals(self):
        if hasattr(self, 'data_cleaner_module'): self.data_cleaner_module.data_cleaned.connect(self.on_data_cleaned_by_module)
        if hasattr(self, 'data_viewer_module'): self.data_updated_signal.connect(self.data_viewer_module.load_data)
        if hasattr(self, 'model_trainer_module'): self.data_updated_signal.connect(self.model_trainer_module.set_data)
        if hasattr(self, 'visualizer_module'): self.data_updated_signal.connect(self.visualizer_module.load_data)

    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: #e0e4e8; }}
            #Sidebar {{ background-color: #28333e; border-right: 1px solid #1f2830; }}
            #AppTitleLabel {{ 
                font-size: 15.5px;
                font-weight: bold; color: #f0f0f0;
                padding-left: 3px; 
                min-width: 150px;
            }}
            #VersionLabel {{ font-size: 9px; color: #8a9eb2; padding-bottom: 3px; }}
            QPushButton#NavButton {{
                color: #c5d5e5; background-color: transparent; border: none;
                padding: 9px 10px; text-align: left; 
                font-size: 13px; font-weight: normal; border-radius: 3px;
            }}
            QPushButton#NavButton:hover {{ background-color: #313f4c; }} 
            QPushButton#NavButton:checked {{ background-color: #0078d4; color: white; font-weight: bold; }}
            QPushButton#NavButton:disabled {{ color: #607080; }}
            #MainContentStack {{ background-color: #ffffff; border: none; }}
            #AppStatusBar {{ 
                font-size: 11px; color: #1c252e; background-color: #d8dde2; 
                border-top: 1px solid #bec5cb; padding: 1px 0;
            }}
            #AppStatusBar QLabel {{ color: #1c252e; padding-left: 4px; }}
            QProgressBar {{ 
                border: 1px solid #a8b0b7; border-radius: 3px; background-color: #e8ecef; 
                text-align: center; font-size: 9px; color: #1c252e; height: 16px;
            }}
            QProgressBar::chunk {{ background-color: #0078d4; border-radius: 2px; margin: 1px; }}
            QSplitter::handle:horizontal {{ background-color: #b0b8c0; width: 2px; image: none; }}
            QSplitter::handle:horizontal:hover {{ background-color: #98a0a8; }}
        """)

    def _update_active_nav_button(self, current_widget):
        target_text_map = { module: text for text, module in [
            ("Dashboard", getattr(self, 'dashboard_module', None)),
            ("View Data", getattr(self, 'data_viewer_module', None)),
            ("Clean & Preprocess", getattr(self, 'data_cleaner_module', None)),
            ("Train Models", getattr(self, 'model_trainer_module', None)),
            ("Visualize Data", getattr(self, 'visualizer_module', None))
        ] if module is not None }
        
        target_text = target_text_map.get(current_widget)
        for text, btn in self.nav_buttons.items():
            is_active = (text == target_text)
            btn.setChecked(is_active)
            if is_active: self.active_button = btn

    def update_module_button_states(self):
        data_is_loaded = self.current_data_frame is not None and not self.current_data_frame.empty
        for btn_text, btn in self.nav_buttons.items():
            if btn.property("requires_data"): btn.setEnabled(data_is_loaded)
            elif btn_text == "Import Data": btn.setEnabled(True)

    def _import_data_action(self):
        if self.nav_buttons.get("Import Data"): self.nav_buttons["Import Data"].setChecked(False)
        if self.active_button: self.active_button.setChecked(True)
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Data File", "", "Data Files (*.csv *.xlsx *.xls *.json);;All Files (*)")
        if file_path:
            self.status_label.setText(f"Loading: {os.path.basename(file_path)}...")
            self.progress_bar.setValue(0); self.progress_bar.setVisible(True)
            if self.data_loader_thread and self.data_loader_thread.isRunning():
                logger.warning("Data loader thread is already running. Please wait.")
                self._show_message_box("Busy", "A file is already being loaded. Please wait.", "warning")
                self.progress_bar.setVisible(False)
                self.status_label.setText("Previous data loading in progress...")
                return

            self.data_loader_thread = DataLoaderThread(file_path)
            self.data_loader_thread.progress_updated.connect(self.update_progress_bar)
            self.data_loader_thread.data_loaded.connect(self.on_data_loaded_successfully)
            self.data_loader_thread.loading_error.connect(self.on_data_loading_error)
            self.data_loader_thread.finished.connect(self._on_data_loader_finished)
            self.data_loader_thread.start()

    def _on_data_loader_finished(self):
        logger.debug("DataLoaderThread finished.")
        self.data_loader_thread = None

    def update_progress_bar(self, value): self.progress_bar.setValue(value)

    def on_data_loaded_successfully(self, data_frame: pd.DataFrame):
        self.current_data_frame = data_frame
        file_name = os.path.basename(self.data_loader_thread.file_path if self.data_loader_thread else "Data")
        if data_frame.empty:
            self.status_label.setText(f"File '{file_name}' loaded, but it's empty or header-only.")
            logger.warning(f"Loaded data from '{file_name}' is empty.")
        else:
            self.status_label.setText(f"'{file_name}' loaded: {data_frame.shape[0]} rows, {data_frame.shape[1]} columns.")
        self.progress_bar.setVisible(False)
        self.data_updated_signal.emit(self.current_data_frame.copy())
        self.update_module_button_states()
        self._show_data_viewer()

    def on_data_loading_error(self, error_message):
        self.current_data_frame = None
        self.status_label.setText("Failed to load data. See error message."); self.progress_bar.setVisible(False)
        self._show_message_box("Data Loading Error", error_message, "critical")
        self.update_module_button_states()
        if self.active_button: self.active_button.setChecked(True) 
        else: self._show_dashboard()

    def on_data_cleaned_by_module(self, cleaned_data_frame: pd.DataFrame):
        self.current_data_frame = cleaned_data_frame.copy()
        self.status_label.setText(f"Data cleaned/updated: {self.current_data_frame.shape[0]} rows, {self.current_data_frame.shape[1]} cols.")
        self.data_updated_signal.emit(self.current_data_frame.copy())

    def _switch_content_widget(self, widget_instance):
        self.content_stack.setCurrentWidget(widget_instance)
        self._update_active_nav_button(widget_instance)
        module_name = widget_instance.objectName().replace("Widget", "").replace("Module","").replace("_", " ").title().strip()
        if not module_name: module_name = "Unknown View"
        self.status_label.setText(f"Viewing: {module_name}")

    def _show_dashboard(self): self._switch_content_widget(self.dashboard_module)
    def _show_data_viewer(self):
        if self.current_data_frame is not None: self._switch_content_widget(self.data_viewer_module)
        else: self._handle_no_data_for_view("View Data", "Cannot view data.")
    def _show_data_cleaner(self):
        if self.current_data_frame is not None:
            if hasattr(self, 'data_cleaner_module'): self.data_cleaner_module.load_data(self.current_data_frame.copy())
            self._switch_content_widget(self.data_cleaner_module)
        else: self._handle_no_data_for_view("Clean & Preprocess", "Cannot clean data.")
    def _show_model_trainer(self):
        if self.current_data_frame is not None:
            if hasattr(self, 'model_trainer_module'): self.model_trainer_module.set_data(self.current_data_frame.copy())
            self._switch_content_widget(self.model_trainer_module)
        else: self._handle_no_data_for_view("Train Models", "Cannot train models.")
    def _show_visualizer(self):
        if self.current_data_frame is not None:
            if hasattr(self, 'visualizer_module'): self.visualizer_module.load_data(self.current_data_frame.copy())
            self._switch_content_widget(self.visualizer_module)
        else: self._handle_no_data_for_view("Visualize Data", "Cannot visualize data.")

    def _handle_no_data_for_view(self, button_text: str, base_message: str):
        self.status_label.setText(base_message + " Please import data first.")
        if button_text in self.nav_buttons: self.nav_buttons[button_text].setChecked(False)
        if self.active_button: self.active_button.setChecked(True)
        else: self._show_dashboard() 

    def _show_message_box(self, title, message, msg_type="information"):
        msg_box = QMessageBox(self); msg_box.setWindowTitle(title); msg_box.setText(message)
        icon_map = {"warning": QMessageBox.Warning, "critical": QMessageBox.Critical, "information": QMessageBox.Information}
        msg_box.setIcon(icon_map.get(msg_type, QMessageBox.Information)); msg_box.setStandardButtons(QMessageBox.Ok); msg_box.exec_()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit', "Are you sure you want to exit Machine Learning Studio?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            logger.info("Application closing by user confirmation.")
            if self.data_loader_thread and self.data_loader_thread.isRunning():
                logger.info("Terminating data loader thread..."); self.data_loader_thread.terminate(); self.data_loader_thread.wait(500)
            event.accept()
        else: logger.info("Application close cancelled by user."); event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG)
    
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())