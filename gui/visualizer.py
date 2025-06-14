import logging
import os
import sys

import pandas as pd
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QComboBox, QFormLayout, QGroupBox,
                             QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget,QCheckBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

try:
    from utils.visual_utils import VisualUtils
except ImportError:
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from utils.visual_utils import VisualUtils


logger = logging.getLogger(__name__)

ICON_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icons")

def get_icon(name):
    path = os.path.join(ICON_PATH_ROOT, name)
    return QIcon(path) if os.path.exists(path) else QIcon()

ICON_CHART_BAR = get_icon("bar-chart.png")
ICON_CHART_LINE = get_icon("line-chart.png")
ICON_CHART_PIE = get_icon("pie-chart.png")
ICON_SCATTER = get_icon("scatter-plot.png")
ICON_HISTOGRAM = get_icon("histogram.png")
ICON_BOXPLOT = get_icon("boxplot.png")
ICON_CORR = get_icon("corr-matrix.png")

class Visualizer(QWidget):
    data_visualization_requested = pyqtSignal(pd.DataFrame, str, dict)

    def __init__(self, parent_window=None):
        super().__init__(parent_window)
        self.setObjectName("VisualizerWidget")
        self.parent_window = parent_window
        self.current_df: pd.DataFrame | None = None
        self.numeric_columns: list = []
        self.categorical_columns: list = []
        self.all_columns: list = []

        self._setup_ui()
        self._apply_stylesheet()
        self._connect_signals()
        self.update_controls_state()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QFormLayout(controls_group)
        controls_layout.setSpacing(8)
        controls_layout.setLabelAlignment(Qt.AlignLeft)

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "-- Select Plot Type --",
            "Histogram / Distribution",
            "Scatter Plot",
            "Line Plot",
            "Bar Plot (Counts or Aggregate)",
            "Box Plot",
            "Pie Chart",
            "Correlation Matrix",
            "KDE Plot",
            "Pair Plot (Subsample)"
        ])
        controls_layout.addRow("Plot Type:", self.plot_type_combo)

        self.dynamic_controls_widget = QWidget()
        self.dynamic_controls_layout = QFormLayout(self.dynamic_controls_widget)
        self.dynamic_controls_layout.setSpacing(8)
        self.dynamic_controls_layout.setContentsMargins(0,0,0,0)
        controls_layout.addRow(self.dynamic_controls_widget)

        self.generate_plot_button = QPushButton(" Generate Plot")
        self.generate_plot_button.setIcon(get_icon("image.png"))
        
        controls_layout.addRow(self.generate_plot_button)
        main_layout.addWidget(controls_group)

        plot_display_group = QGroupBox("Plot Output")
        plot_display_layout = QVBoxLayout(plot_display_group)
        plot_display_layout.setContentsMargins(5, 5, 5, 5)

        self.plot_canvas = FigureCanvas(Figure(figsize=(7, 5)))
        self.plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_display_layout.addWidget(self.plot_canvas)
        
        main_layout.addWidget(plot_display_group, 1)

        self._create_placeholder_plot("Visualizer initialized. Load data to generate plots.")

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            #VisualizerWidget {
                background-color: #f4f6f8;
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d1d9e0;
                border-radius: 6px;
                margin-top: 6px; 
                padding: 8px 8px 8px 8px;
                font-size: 12.5px;
                font-weight: bold;
                color: #102a43;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                left: 10px; padding: 0px 5px 2px 5px;
                color: #005a9e; background-color: #ffffff;
                font-size: 12px; font-weight: bold;
            }
            QPushButton {
                background-color: #0078d4; color: white;
                font-size: 11px; font-weight: bold;
                padding: 6px 10px; border-radius: 4px;
                border: 1px solid #005a9e; min-height: 26px;
            }
            QPushButton:hover { background-color: #005a9e; }
            QPushButton:disabled { background-color: #d8dcde; color: #707070; border-color: #c0c4c8; }
            QComboBox, QLineEdit {
                background-color: #ffffff; border: 1px solid #b0b8bf;
                border-radius: 3px; padding: 4px 5px;
                font-size: 11px; min-height: 24px; color: #202020;
            }
            QComboBox:focus, QLineEdit:focus { border-color: #0078d4; }
            QLabel { font-size: 11px; color: #202020; }
        """)

    def _connect_signals(self):
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_change)
        self.generate_plot_button.clicked.connect(self._generate_selected_plot)

    def _create_placeholder_plot(self, message="Load data and select plot type."):
        self.plot_canvas.figure.clear()
        ax = self.plot_canvas.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10, color='grey', wrap=True)
        ax.set_axis_off()
        self.plot_canvas.draw_idle()

    def load_data(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame, pd.DataFrame):
            self._show_message("Error", "Invalid data type provided to Visualizer.", "critical")
            self.current_df = None
        else:
            self.current_df = data_frame.copy()
            logger.info(f"Visualizer: Data loaded with shape {self.current_df.shape}")
            self.numeric_columns = self.current_df.select_dtypes(include=np.number).columns.tolist()
            self.categorical_columns = self.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
            self.all_columns = self.current_df.columns.tolist()
            self._create_placeholder_plot("Data loaded. Select plot type and options.")
            if self.parent_window and hasattr(self.parent_window, 'status_label'):
                self.parent_window.status_label.setText(f"Visualizer: Data ready ({self.current_df.shape[0]}x{self.current_df.shape[1]}).")
        
        self.update_controls_state()
        self._on_plot_type_change(self.plot_type_combo.currentIndex())


    def update_controls_state(self):
        has_data = self.current_df is not None and not self.current_df.empty
        self.plot_type_combo.setEnabled(has_data)
        self.generate_plot_button.setEnabled(has_data and self.plot_type_combo.currentIndex() > 0)

    def _on_plot_type_change(self, index):
        while self.dynamic_controls_layout.rowCount() > 0:
            self.dynamic_controls_layout.removeRow(0)

        plot_type = self.plot_type_combo.currentText()
        has_data = self.current_df is not None

        if not has_data or index == 0:
            self.generate_plot_button.setEnabled(False)
            return
        
        self.generate_plot_button.setEnabled(True)

        if plot_type == "Scatter Plot":
            self.x_col_combo_scatter = QComboBox()
            self.y_col_combo_scatter = QComboBox()
            self.hue_col_combo_scatter = QComboBox()
            self.hue_col_combo_scatter.addItem("-- Optional Hue --")

            if has_data:
                self.x_col_combo_scatter.addItems(self.numeric_columns)
                self.y_col_combo_scatter.addItems(self.numeric_columns)
                self.hue_col_combo_scatter.addItems(self.categorical_columns)

            self.dynamic_controls_layout.addRow("X-axis (Numeric):", self.x_col_combo_scatter)
            self.dynamic_controls_layout.addRow("Y-axis (Numeric):", self.y_col_combo_scatter)
            self.dynamic_controls_layout.addRow("Hue (Categorical):", self.hue_col_combo_scatter)

        elif plot_type == "Histogram / Distribution":
            self.hist_col_combo = QComboBox()
            if has_data: self.hist_col_combo.addItems(self.numeric_columns)
            self.dynamic_controls_layout.addRow("Column (Numeric):", self.hist_col_combo)
            self.kde_checkbox = QCheckBox("Show KDE")
            self.kde_checkbox.setChecked(True)
            self.dynamic_controls_layout.addRow(self.kde_checkbox)

        elif plot_type == "Correlation Matrix":
            pass
        else:
            self.generate_plot_button.setEnabled(False)

    def _generate_selected_plot(self):
        if self.current_df is None or self.current_df.empty:
            self._show_message("No Data", "Please load data first.", "warning")
            return

        plot_type = self.plot_type_combo.currentText()
        if plot_type == "-- Select Plot Type --":
            self._show_message("No Plot Type", "Please select a plot type.", "info")
            return

        fig = None
        try:
            if plot_type == "Scatter Plot":
                x_col = self.x_col_combo_scatter.currentText()
                y_col = self.y_col_combo_scatter.currentText()
                hue_col = self.hue_col_combo_scatter.currentText()
                if not x_col or not y_col:
                    self._show_message("Input Missing", "Please select X and Y axis columns for Scatter Plot.", "warning"); return
                fig = VisualUtils.create_scatter_plot(self.current_df, x_col, y_col, 
                                                      hue=None if hue_col == "-- Optional Hue --" else hue_col)
            
            elif plot_type == "Histogram / Distribution":
                col = self.hist_col_combo.currentText()
                if not col:
                     self._show_message("Input Missing", "Please select a column for Histogram.", "warning"); return
                fig = VisualUtils.create_histogram(self.current_df[col], kde=self.kde_checkbox.isChecked())

            elif plot_type == "Correlation Matrix":
                fig = VisualUtils.create_correlation_matrix(self.current_df)
            
            if fig:
                self.plot_canvas.figure = fig
                self.plot_canvas.draw_idle()
                if self.parent_window and hasattr(self.parent_window, 'status_label'):
                    self.parent_window.status_label.setText(f"Visualizer: '{plot_type}' generated.")
            else:
                self._create_placeholder_plot(f"Could not generate '{plot_type}'. Check selections or data.")
        
        except Exception as e:
            logger.error(f"Error generating plot '{plot_type}': {e}", exc_info=True)
            self._show_message("Plot Generation Error", f"Could not generate plot: {e}", "critical")
            self._create_placeholder_plot(f"Error generating '{plot_type}'.")


    def _show_message(self, title: str, message: str, type: str = "information"):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        icon_map = {"critical": QMessageBox.Critical, "warning": QMessageBox.Warning, "information": QMessageBox.Information}
        msg_box.setIcon(icon_map.get(type.lower(), QMessageBox.Information))
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    dummy_data = pd.DataFrame({
        'NumericA': pd.np.random.rand(100) * 10,
        'NumericB': pd.np.random.randn(100) * 5 + 20,
        'CategoryX': pd.np.random.choice(['P', 'Q', 'R', 'S'], 100),
        'CategoryY': pd.np.random.choice(['Low', 'Medium', 'High'], 100)
    })

    main_window = QWidget()
    main_window.status_label = QLabel("Status bar for dummy parent")

    visualizer_widget = Visualizer(parent_window=main_window)
    visualizer_widget.load_data(dummy_data)
    
    layout = QVBoxLayout(main_window)
    layout.addWidget(QLabel("<h2>Visualizer Test Window</h2>"))
    layout.addWidget(visualizer_widget)
    layout.addWidget(main_window.status_label)
    main_window.setLayout(layout)
    
    main_window.setWindowTitle("Standalone Visualizer Test")
    main_window.setGeometry(100, 100, 800, 700)
    main_window.show()
    
    sys.exit(app.exec_())