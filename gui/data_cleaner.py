from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QTableView,
    QFileDialog, QDialog, QGroupBox, QCheckBox, QComboBox, QDialogButtonBox,
    QHeaderView, QLabel, QMessageBox, QListWidget, QScrollArea,
    QFormLayout, QLineEdit, QSpinBox, QAbstractItemView, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QIcon, QFont
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

try:
    from .data_viewer import PandasTableModel
except ImportError:
    from data_viewer import PandasTableModel


ICON_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icons")
ICON_BACK = os.path.join(ICON_PATH_ROOT, "back.png")
ICON_MISSING = os.path.join(ICON_PATH_ROOT, "missing.png")
ICON_SCALE = os.path.join(ICON_PATH_ROOT, "scale.png")
ICON_REMOVE_COLS = os.path.join(ICON_PATH_ROOT, "remove.png")
ICON_SAVE_CLEAN = os.path.join(ICON_PATH_ROOT, "save.png")
ICON_DUPLICATES = os.path.join(ICON_PATH_ROOT, "copy.png")

class DataCleaner(QWidget):
    data_cleaned = pyqtSignal(pd.DataFrame)

    def __init__(self, parent_window=None):
        super().__init__(parent_window)
        self.setObjectName("DataCleanerWidget")
        self.parent_window = parent_window
        self.data: pd.DataFrame | None = None
        self.original_data_for_reset: pd.DataFrame | None = None
        self.table_model: PandasTableModel | None = None 

        self._init_ui()
        self._apply_stylesheet()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12) 
        main_layout.setSpacing(10) 

        header_layout = QHBoxLayout()
        self.back_button = QPushButton()
        if os.path.exists(ICON_BACK): self.back_button.setIcon(QIcon(ICON_BACK))
        self.back_button.setIconSize(QSize(22, 22)); self.back_button.setFixedSize(34, 34) 
        self.back_button.setToolTip("Back to Data Viewer"); self.back_button.setObjectName("BackButton")
        self.back_button.clicked.connect(self._go_back)
        header_layout.addWidget(self.back_button)

        title_label = QLabel("Data Cleaning & Preprocessing")
        title_label.setObjectName("TitleLabel")
        header_layout.addWidget(title_label); header_layout.addStretch()
        main_layout.addLayout(header_layout)

        content_splitter = QSplitter(Qt.Horizontal, self)
        content_splitter.setHandleWidth(4) 
        content_splitter.setObjectName("DataCleanerSplitter")
        content_splitter.setChildrenCollapsible(False)

        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0,0,0,0)
        
        self.data_table = QTableView(self)
        self.data_table.setObjectName("CleaningDataTable")
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setSortingEnabled(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.data_table.horizontalHeader().setStretchLastSection(True) 
        self.data_table.horizontalHeader().setDefaultSectionSize(110) 
        self.data_table.horizontalHeader().setMinimumSectionSize(35)
        self.data_table.verticalHeader().setVisible(True) 
        self.data_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed) 
        self.data_table.verticalHeader().setDefaultSectionSize(23)

        table_layout.addWidget(self.data_table)
        content_splitter.addWidget(table_container)

        tools_panel_scroll = QScrollArea(self)
        tools_panel_scroll.setWidgetResizable(True); tools_panel_scroll.setObjectName("ToolsPanelScroll")
        tools_panel_scroll.setMinimumWidth(230)
        
        tools_panel_widget = QWidget()
        tools_panel_widget.setObjectName("ToolsPanelWidgetInternal")
        tools_panel_layout = QVBoxLayout(tools_panel_widget)
        tools_panel_layout.setSpacing(6); tools_panel_layout.setAlignment(Qt.AlignTop)
        tools_panel_layout.setContentsMargins(5, 5, 5, 5)

        button_info = {
            "missing": (" Missing Values", ICON_MISSING, self._open_missing_values_dialog, "Missing Values"),
            "scale": (" Scale/Normalize", ICON_SCALE, self._open_scaling_dialog, "Scaling"),
            "remove_cols": (" Remove Columns", ICON_REMOVE_COLS, self._open_remove_columns_dialog, "Column Management"),
            "duplicates": (" Duplicates", ICON_DUPLICATES, self._remove_duplicates, "Duplicate Rows")
        }

        for key, (text, icon_path, slot, group_title) in button_info.items():
            group_box = QGroupBox(group_title)
            group_box.setObjectName(f"ToolGroup_{key}")
            group_layout = QVBoxLayout(group_box)
            group_layout.setContentsMargins(6, 8, 6, 6) 
            group_layout.setSpacing(4)
            
            button = QPushButton(text)
            button.setObjectName(f"ToolButton_{key}") 
            if os.path.exists(icon_path): button.setIcon(QIcon(icon_path))
            else: logger.warning(f"Icon not found for DataCleaner tool button: {icon_path}")
            button.setIconSize(QSize(16,16)) 
            button.clicked.connect(slot)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            group_layout.addWidget(button)
            tools_panel_layout.addWidget(group_box)

        tools_panel_layout.addStretch(1)
        tools_panel_widget.setLayout(tools_panel_layout)
        tools_panel_scroll.setWidget(tools_panel_widget)
        content_splitter.addWidget(tools_panel_scroll)

        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(content_splitter, 1)

        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.addStretch()
        self.reset_data_btn = QPushButton(" Reset Changes")
        self.reset_data_btn.setObjectName("ResetDataButton")
        self.reset_data_btn.setToolTip("Revert all cleaning changes made in this session.")
        self.reset_data_btn.clicked.connect(self._reset_data)
        action_buttons_layout.addWidget(self.reset_data_btn)

        self.save_cleaned_data_btn = QPushButton(" Apply & Commit") 
        self.save_cleaned_data_btn.setObjectName("SaveCleanedDataButton")
        if os.path.exists(ICON_SAVE_CLEAN): self.save_cleaned_data_btn.setIcon(QIcon(ICON_SAVE_CLEAN))
        self.save_cleaned_data_btn.setToolTip("Apply cleaning changes and update the dataset globally.")
        self.save_cleaned_data_btn.clicked.connect(self._apply_and_save_cleaned_data)
        action_buttons_layout.addWidget(self.save_cleaned_data_btn)
        main_layout.addLayout(action_buttons_layout)

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            #DataCleanerWidget { background-color: #f7f8fa; }
            #TitleLabel { font-size: 18px; font-weight: bold; color: #2a3035; margin-left: 6px; }
            #BackButton { background-color: transparent; border: none; padding: 0; }
            #BackButton:hover { background-color: #dfe3e6; border-radius: 4px; }
            
            QGroupBox { 
                font-size: 12.5px; font-weight: bold; color: #33393e; 
                border: 1px solid #d5d9dd; border-radius: 5px; 
                margin-top: 4px; /* Reduced margin-top */
                padding: 5px; /* Reduced padding */
                background-color: #fdfefe; 
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                left: 6px; padding: 0 3px 2px 3px; color: #0078d4; 
                background-color: #fdfefe; 
            }

            /* === Specific styles for buttons in the Tools Panel === */
            #ToolsPanelScroll QGroupBox QPushButton { 
                background-color: #0078D4; /* Microsoft Blue */
                color: white;
                text-align: left; 
                padding: 6px 8px; /* Adjusted padding */
                font-size: 11.5px;   /* Adjusted font size */
                min-height: 28px;  /* Adjusted min height */
                max-height: 28px;  /* Explicit max height to control size */
                border-radius: 3px; 
                border: 1px solid #005a9e; 
                margin: 1px 0px; /* Minimal vertical margin */
            }
            #ToolsPanelScroll QGroupBox QPushButton:hover {
                background-color: #005a9e; 
            }
             #ToolsPanelScroll QGroupBox QPushButton:pressed {
                background-color: #004c87; 
            }
            #ToolsPanelScroll QGroupBox QPushButton:disabled {
                background-color: #d2d2d2; color: #777777; border-color: #b0b0b0;
            }
            /* ===================================================== */

            #CleaningDataTable { 
                border: 1px solid #cdd2d6; gridline-color: #e7eaed; font-size: 11.5px; 
                selection-background-color: #a0c8f0; selection-color: #151515; 
            }
            #CleaningDataTable QHeaderView::section { 
                background-color: #e7eaed; color: #2f353a; padding: 4px; 
                border: 1px solid #cdd2d6; font-weight: bold; font-size: 11.5px;
            }
            #ToolsPanelScroll { border: none; background-color: transparent; }
            #ToolsPanelScroll QWidget#ToolsPanelWidgetInternal { background-color: transparent; }
            
            /* General action buttons (Reset, Apply) at the bottom */
            QPushButton { /* This is for buttons NOT in #ToolsPanelScroll QGroupBox */
                background-color: #6c757d; /* Default grey for other buttons if not styled by ID */
                color: white; font-size: 12px; 
                font-weight: bold; padding: 6px 12px; 
                border-radius: 4px; border: none; min-height: 28px; 
            }
            QPushButton:hover { background-color: #5a6268; }
            QPushButton:pressed { background-color: #545b62; }
            QPushButton:disabled { background-color: #c6c6c6; color: #505050; }

            #ResetDataButton { background-color: #ffb900; color: #212529; }
            #ResetDataButton:hover { background-color: #e0a200; }
            #SaveCleanedDataButton { background-color: #107c10; color: white; }
            #SaveCleanedDataButton:hover { background-color: #0d650d; }

            QSplitter::handle:horizontal { background-color: #c0c4c8; width: 3px; margin: 2px 0; }
            QSplitter::handle:hover { background-color: #a0a4a8; }
        """)

    def load_data(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame, pd.DataFrame):
            self._show_message("Error", "Invalid data type provided to Data Cleaner.", "critical")
            self.data = None; self.original_data_for_reset = None
            self._refresh_table_display(); return
        self.data = data_frame.copy()
        self.original_data_for_reset = data_frame.copy()
        self._refresh_table_display()
        if self.parent_window and hasattr(self.parent_window, 'status_label'):
            self.parent_window.status_label.setText(f"Cleaner: Data loaded ({self.data.shape[0]}x{self.data.shape[1]}). Ready for operations.")

    def _refresh_table_display(self):
        if self.data is None or self.data.empty:
            self.table_model = PandasTableModel(pd.DataFrame({'Status': ['No data loaded or data is empty.']}))
            self.data_table.setModel(self.table_model)
            if self.table_model.columnCount() > 0:
                 self.data_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            return
        self.table_model = PandasTableModel(self.data) 
        self.data_table.setModel(self.table_model)

    def _open_missing_values_dialog(self):
        if self.data is None: self._show_message("Warning", "No data loaded.", "warning"); return
        dialog = MissingValuesDialog(self.data.copy(), self) 
        if dialog.exec_():
            self.data = dialog.get_cleaned_data()
            self._refresh_table_display()
            self.data_cleaned.emit(self.data.copy()) 
            self._show_message("Missing Values", "Missing values operation applied.", "information")

    def _open_scaling_dialog(self):
        if self.data is None: self._show_message("Warning", "No data loaded.", "warning"); return
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols: self._show_message("Info", "No numeric columns to scale.", "information"); return
        dialog = ScalingDialog(numeric_cols, self)
        if dialog.exec_():
            selected_cols, scaler_type = dialog.get_scaling_options()
            if selected_cols:
                try:
                    scaler_map = {"MinMaxScaler": MinMaxScaler(), "StandardScaler": StandardScaler(), "RobustScaler": RobustScaler()}
                    scaler = scaler_map.get(scaler_type); 
                    if not scaler: return
                    valid_selected_cols = [col for col in selected_cols if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col])]
                    if not valid_selected_cols: self._show_message("Info", "No valid numeric columns for scaling.", "information"); return
                    self.data[valid_selected_cols] = scaler.fit_transform(self.data[valid_selected_cols])
                    self._refresh_table_display(); self.data_cleaned.emit(self.data.copy())
                    self._show_message("Scaling", f"Data scaled using {scaler_type}.", "information")
                except Exception as e: self._show_message("Error", f"Scaling error: {e}", "critical"); logger.error(f"Scaling error: {e}", exc_info=True)

    def _open_remove_columns_dialog(self):
        if self.data is None: self._show_message("Warning", "No data loaded.", "warning"); return
        dialog = RemoveColumnsDialog(self.data.columns.tolist(), self)
        if dialog.exec_():
            cols_to_remove = dialog.get_columns_to_remove()
            if cols_to_remove:
                if len(cols_to_remove) >= len(self.data.columns): self._show_message("Error", "Cannot remove all columns.", "critical"); return
                valid_cols_to_remove = [col for col in cols_to_remove if col in self.data.columns]
                if not valid_cols_to_remove: self._show_message("Info", "Selected columns for removal not found.", "information"); return
                self.data.drop(columns=valid_cols_to_remove, inplace=True)
                self._refresh_table_display(); self.data_cleaned.emit(self.data.copy())
                self._show_message("Columns Removed", f"Columns removed: {', '.join(valid_cols_to_remove)}.", "information")

    def _remove_duplicates(self):
        if self.data is None: self._show_message("Warning", "No data loaded.", "warning"); return
        initial_rows = len(self.data); self.data.drop_duplicates(inplace=True, ignore_index=True)
        rows_removed = initial_rows - len(self.data)
        self._refresh_table_display(); self.data_cleaned.emit(self.data.copy())
        self._show_message("Duplicates", f"{rows_removed} duplicate row(s) removed.", "information")

    def _reset_data(self):
        if self.original_data_for_reset is not None:
            self.data = self.original_data_for_reset.copy()
            self._refresh_table_display(); self.data_cleaned.emit(self.data.copy()) 
            self._show_message("Reset", "Data reset to original state for this cleaner session.", "information")
        else: self._show_message("Warning", "No original data state to reset.", "warning")

    def _apply_and_save_cleaned_data(self):
        if self.data is None: self._show_message("Warning", "No data to apply.", "warning"); return
        self.data_cleaned.emit(self.data.copy()) 
        self._show_message("Changes Applied", "Cleaning changes committed and dataset updated globally.", "information")
        if self.parent_window and hasattr(self.parent_window, 'status_label'):
            self.parent_window.status_label.setText("Data cleaning changes committed.")

    def _go_back(self):
        if self.parent_window and hasattr(self.parent_window, 'content_stack') and hasattr(self.parent_window, 'data_viewer_module'):
            self.parent_window.content_stack.setCurrentWidget(self.parent_window.data_viewer_module)
            if self.parent_window and hasattr(self.parent_window, 'status_label'):
                 self.parent_window.status_label.setText("Viewing data.")
        else: logger.warning("Cannot navigate back - parent_window or data_viewer_module not properly set.")

    def _show_message(self, title, message, type="information"):
        msg_box = QMessageBox(self); msg_box.setWindowTitle(title); msg_box.setText(message)
        icon_map = {"critical": QMessageBox.Critical, "warning": QMessageBox.Warning, "information": QMessageBox.Information}
        msg_box.setIcon(icon_map.get(type, QMessageBox.Information)); msg_box.setStandardButtons(QMessageBox.Ok); msg_box.exec_()


class MissingValuesDialog(QDialog):
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent); self.data = data; self.setWindowTitle("Handle Missing Values")
        self.setMinimumWidth(460); self.setObjectName("MissingValuesDialog")
        layout = QVBoxLayout(self); form_layout = QFormLayout(); form_layout.setSpacing(6)
        self.numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        self.object_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_group = QGroupBox(f"Numeric Columns ({len(self.numeric_cols)})"); numeric_layout = QFormLayout(numeric_group); numeric_layout.setSpacing(6)
        self.numeric_strategy_combo = QComboBox(); self.numeric_strategy_combo.addItems(["None", "Mean", "Median", "Mode", "Fill Specific", "Remove Rows with NaN"])
        self.numeric_fill_value_edit = QLineEdit("0"); self.numeric_fill_value_edit.setVisible(False)
        self.numeric_strategy_combo.currentTextChanged.connect(lambda text: self.numeric_fill_value_edit.setVisible(text == "Fill Specific"))
        numeric_layout.addRow("Strategy:", self.numeric_strategy_combo); numeric_layout.addRow("Fill Value:", self.numeric_fill_value_edit)
        if not self.numeric_cols: numeric_group.setEnabled(False)
        form_layout.addRow(numeric_group)
        object_group = QGroupBox(f"Categorical Columns ({len(self.object_cols)})"); object_layout = QFormLayout(object_group); object_layout.setSpacing(6)
        self.object_strategy_combo = QComboBox(); self.object_strategy_combo.addItems(["None", "Mode", "Fill Specific", "Remove Rows with NaN"])
        self.object_fill_value_edit = QLineEdit("Unknown"); self.object_fill_value_edit.setVisible(False)
        self.object_strategy_combo.currentTextChanged.connect(lambda text: self.object_fill_value_edit.setVisible(text == "Fill Specific"))
        object_layout.addRow("Strategy:", self.object_strategy_combo); object_layout.addRow("Fill Value:", self.object_fill_value_edit)
        if not self.object_cols: object_group.setEnabled(False)
        form_layout.addRow(object_group)
        drop_group = QGroupBox("Global Actions (Applied After Specific Fills)"); drop_layout = QVBoxLayout(drop_group)
        self.drop_any_col_cb = QCheckBox("Drop columns if ALL values in column are missing (Applied First)")
        self.drop_any_row_cb = QCheckBox("Drop rows if ANY column still has missing value")
        drop_layout.addWidget(self.drop_any_col_cb); drop_layout.addWidget(self.drop_any_row_cb)
        form_layout.addRow(drop_group); layout.addLayout(form_layout)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons); self._apply_dialog_stylesheet()
    def _apply_dialog_stylesheet(self):
        self.setStyleSheet("""
            #MissingValuesDialog { background-color: #ffffff; } QGroupBox { font-weight: bold; margin-top: 3px; padding: 6px; } 
            QGroupBox::title { left: 5px; padding-bottom: 2px;}
            QLineEdit, QComboBox { padding: 4px; border: 1px solid #ced4da; border-radius: 3px; min-height: 25px;} 
            QPushButton { padding: 6px 12px; min-width: 70px; border-radius: 4px; font-weight:bold; }""")
    def get_cleaned_data(self) -> pd.DataFrame:
        if self.drop_any_col_cb.isChecked(): self.data.dropna(axis=1, how='all', inplace=True)
        self.numeric_cols = [col for col in self.numeric_cols if col in self.data.columns]
        self.object_cols = [col for col in self.object_cols if col in self.data.columns]
        num_strat = self.numeric_strategy_combo.currentText()
        if num_strat != "None" and self.numeric_cols:
            if num_strat == "Remove Rows with NaN": self.data.dropna(subset=self.numeric_cols, how='any', inplace=True)
            else:
                for col in self.numeric_cols:
                    if col not in self.data.columns or not self.data[col].isnull().any(): continue
                    if num_strat == "Mean": self.data[col].fillna(self.data[col].mean(), inplace=True)
                    elif num_strat == "Median": self.data[col].fillna(self.data[col].median(), inplace=True)
                    elif num_strat == "Mode": mode_val = self.data[col].mode(); self.data[col].fillna(mode_val[0] if not mode_val.empty else 0, inplace=True)
                    elif num_strat == "Fill Specific": self.data[col].fillna(float(self.numeric_fill_value_edit.text()) if self.numeric_fill_value_edit.text().strip() else 0, inplace=True)
        obj_strat = self.object_strategy_combo.currentText()
        if obj_strat != "None" and self.object_cols:
            if obj_strat == "Remove Rows with NaN": self.data.dropna(subset=self.object_cols, how='any', inplace=True)
            else:
                for col in self.object_cols:
                    if col not in self.data.columns or not self.data[col].isnull().any(): continue
                    if obj_strat == "Mode": mode_val = self.data[col].mode(); self.data[col].fillna(mode_val[0] if not mode_val.empty else "Unknown", inplace=True)
                    elif obj_strat == "Fill Specific": self.data[col].fillna(self.object_fill_value_edit.text(), inplace=True)
        if self.drop_any_row_cb.isChecked(): self.data.dropna(axis=0, how='any', inplace=True)
        return self.data.reset_index(drop=True)

class ScalingDialog(QDialog):
    def __init__(self, numeric_columns, parent=None):
        super().__init__(parent); self.numeric_columns = numeric_columns; self.setWindowTitle("Scale/Normalize Numeric Data")
        self.setMinimumWidth(350); self.setObjectName("ScalingDialog")
        layout = QVBoxLayout(self); layout.setSpacing(7); layout.addWidget(QLabel("Select numeric columns to scale:"))
        self.column_list_widget = QListWidget(); self.column_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.column_list_widget.addItems(self.numeric_columns)
        for i in range(self.column_list_widget.count()): self.column_list_widget.item(i).setSelected(True)
        layout.addWidget(self.column_list_widget)
        self.scaler_combo = QComboBox(); self.scaler_combo.addItems(["MinMaxScaler", "StandardScaler", "RobustScaler"])
        form_layout = QFormLayout(); form_layout.setSpacing(6); form_layout.addRow("Scaler Type:", self.scaler_combo)
        layout.addLayout(form_layout)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons); self._apply_dialog_stylesheet()
    def _apply_dialog_stylesheet(self):
         self.setStyleSheet("""
            #ScalingDialog { background-color: #ffffff; }
            QListWidget { border: 1px solid #ced4da; border-radius: 3px; padding: 4px; min-height: 100px;}
            QComboBox { padding: 4px; border: 1px solid #ced4da; border-radius: 3px; min-height: 26px;}
            QPushButton { padding: 6px 12px; min-width: 70px; }""")
    def get_scaling_options(self): return [item.text() for item in self.column_list_widget.selectedItems()], self.scaler_combo.currentText()

class RemoveColumnsDialog(QDialog):
    def __init__(self, all_columns, parent=None):
        super().__init__(parent); self.all_columns = all_columns; self.setWindowTitle("Remove Columns")
        self.setMinimumWidth(300); self.setObjectName("RemoveColumnsDialog")
        layout = QVBoxLayout(self); layout.setSpacing(7); layout.addWidget(QLabel("Select columns to remove:"))
        self.column_list_widget = QListWidget(); self.column_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.column_list_widget.addItems(self.all_columns); layout.addWidget(self.column_list_widget)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons); self._apply_dialog_stylesheet()
    def _apply_dialog_stylesheet(self):
        self.setStyleSheet("""
            #RemoveColumnsDialog { background-color: #ffffff; }
            QListWidget { border: 1px solid #ced4da; border-radius: 3px; padding: 4px; min-height: 150px;}
            QPushButton { padding: 6px 12px; min-width: 70px; }""")
    def get_columns_to_remove(self): return [item.text() for item in self.column_list_widget.selectedItems()]


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QStackedWidget as DummyStackedWidget 
    import sys
    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)
    class DummyMainWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.data_viewer_module = QWidget(self); self.data_viewer_module.setObjectName("DummyDataViewerModule")
            self.content_stack = DummyStackedWidget(self); self.content_stack.addWidget(self.data_viewer_module) 
            self.status_label = QLabel("Dummy Main Window Status", self)
            main_layout = QVBoxLayout(self); main_layout.addWidget(self.content_stack); main_layout.addWidget(self.status_label)
            self.setLayout(main_layout)

    data = {
        'ID_Col': range(25),
        'Age_Num': [25, 30, np.nan, 22, 45, 30, 28, 35, 40, 22, np.nan, 33, 29, 42, 38] * (25//15 + 1) [:25],
        'City_Cat': ['NY', 'LA', 'NY', 'SF', np.nan, 'LA', 'Boston', 'NY', 'SF', 'NY', 'LA', 'Boston', np.nan, 'SF', 'NY'] * (25//15 + 1) [:25],
        'Salary_Num_Missing_k': ['50k', '60k', '55k', np.nan, '120k', '60k', '70k', '80k', '110k', '55k', '90k', '75k', '65k', '130k', '85k'] * (25//15 + 1) [:25], # Keep as string initially
        'Experience_Num': [2, 5, 3, 1, 15, 5, 4, 10, 12, 1, 8, 6, 4, 14, 9] * (25//15 + 1) [:25],
        'Dept_Cat_Missing': [None, 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'HR', 'Sales'] * (25//15 + 1) [:25],
        'All_NaN_Col': [np.nan] * 25
    }
    df = pd.DataFrame(data)
    try:
        df['Salary_Num_Missing'] = df['Salary_Num_Missing_k'].astype(str).str.replace('k', '*1e3', regex=False).map(pd.eval, na_action='ignore').astype(float)
        df.drop(columns=['Salary_Num_Missing_k'], inplace=True)
    except Exception as e_salary:
        logger.error(f"Error converting salary column in dummy data: {e_salary}")
        df['Salary_Num_Missing'] = np.nan
        if 'Salary_Num_Missing_k' in df.columns: df.drop(columns=['Salary_Num_Missing_k'], inplace=True)


    df = pd.concat([df, df.iloc[[2,5,8]]], ignore_index=True)

    main_win_dummy = DummyMainWindow()
    cleaner_widget = DataCleaner(parent_window=main_win_dummy)
    main_win_dummy.content_stack.addWidget(cleaner_widget)
    cleaner_widget.load_data(df.copy()) 
    main_win_dummy.content_stack.setCurrentWidget(cleaner_widget)
    main_win_dummy.setWindowTitle("Machine Learning Studio - Data Cleaner (Enhanced Buttons)")
    main_win_dummy.setGeometry(50, 50, 1050, 680) 
    main_win_dummy.show()
    sys.exit(app.exec_())