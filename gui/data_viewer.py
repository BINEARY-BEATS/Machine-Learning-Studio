from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTableView,
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant
from PyQt5.QtGui import QFont, QColor, QIcon
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

ICON_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icons")
ICON_TABLE = os.path.join(ICON_PATH_ROOT, "table.png")
ICON_SUMMARY = os.path.join(ICON_PATH_ROOT, "clipboard.png")

class PandasTableModel(QAbstractTableModel):
    """
    Custom table model to display pandas DataFrame in a QTableView.
    Supports sorting.
    """
    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def rowCount(self, parent=None):
        return self._dataframe.shape[0]

    def columnCount(self, parent=None):
        return self._dataframe.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._dataframe.empty:
            return QVariant()
        
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()
            try:
                if not (0 <= row < self._dataframe.shape[0] and 0 <= col < self._dataframe.shape[1]):
                    logger.warning(f"PandasTableModel: Data access attempt out of bounds at row {row}, col {col}")
                    return QVariant()

                value = self._dataframe.iloc[row, col]

                if pd.isna(value):
                    return QVariant("NA")

                if isinstance(value, bool): return QVariant(value)
                if isinstance(value, int): return QVariant(int(value)) 
                if isinstance(value, float): return QVariant(float(value))

                if hasattr(np, 'isscalar') and np.isscalar(value):
                    if isinstance(value, np.bool_):
                        return QVariant(bool(value.item()))
                    elif isinstance(value, np.integer):
                        return QVariant(int(value.item()))
                    elif isinstance(value, np.floating):
                        return QVariant(float(value.item()))
                
                if hasattr(value, 'dtype'):
                    if np.issubdtype(value.dtype, np.bool_):
                         return QVariant(bool(value.item()))
                    if np.issubdtype(value.dtype, np.integer):
                        return QVariant(int(value.item()))
                    if np.issubdtype(value.dtype, np.floating):
                        return QVariant(float(value.item()))
                
                str_value = str(value)
                return QVariant(str_value)

            except IndexError: 
                logger.warning(f"PandasTableModel: IndexError during data access at row {row}, col {col}")
                return QVariant() 
            except Exception as e:
                val_for_log = "<unavailable>"
                val_type_for_log = "<unknown>"
                try: 
                    val_for_log = str(self._dataframe.iloc[row, col])[:50] 
                    val_type_for_log = type(self._dataframe.iloc[row, col])
                except:
                    pass
                
                logger.error(
                    f"PandasTableModel: Critical error converting data for display at ({row},{col}). "
                    f"Original Value Snippet: '{val_for_log}', Type: {val_type_for_log}. Error: {e}",
                    exc_info=True
                )
                return QVariant("ERR") 
        
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.DisplayRole:
            if self._dataframe.empty:
                if orientation == Qt.Horizontal: return f"Column {section + 1}"
                if orientation == Qt.Vertical: return f"Row {section + 1}"
                return QVariant()

            if orientation == Qt.Horizontal:
                if 0 <= section < len(self._dataframe.columns):
                    return str(self._dataframe.columns[section])
            elif orientation == Qt.Vertical:
                if 0 <= section < len(self._dataframe.index):
                    return str(self._dataframe.index[section] + 1)
        elif role == Qt.ToolTipRole and orientation == Qt.Horizontal:
             if not self._dataframe.empty and 0 <= section < len(self._dataframe.columns):
                return str(self._dataframe.columns[section])
        return QVariant()

    def sort(self, column: int, order: Qt.SortOrder):
        if self._dataframe.empty: return
        try:
            if 0 <= column < len(self._dataframe.columns):
                col_name = self._dataframe.columns[column]
                self.layoutAboutToBeChanged.emit()
                self._dataframe = self._dataframe.sort_values(
                    by=col_name, ascending=(order == Qt.AscendingOrder), kind='mergesort'
                )
                self.layoutChanged.emit()
            else:
                logger.warning(f"Sort called on invalid column index: {column}")
        except IndexError:
            logger.warning(f"IndexError during sort for column index: {column}")
        except Exception as e:
            logger.error(f"Error during sort on column '{self._dataframe.columns[column] if 0 <= column < len(self._dataframe.columns) else column}': {e}", exc_info=True)


class DataViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DataViewerWidget")
        self.data: pd.DataFrame | None = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.tab_widget = QTabWidget(self)
        self.tab_widget.setObjectName("DataViewerTabWidget")
        main_layout.addWidget(self.tab_widget)

        self.data_tab = QWidget()
        self.data_tab.setObjectName("DataTableTab")
        data_tab_layout = QVBoxLayout(self.data_tab)
        data_tab_layout.setContentsMargins(0,0,0,0)

        self.data_table_view = QTableView(self)
        self.data_table_view.setObjectName("DataTableView")
        self.data_table_view.setAlternatingRowColors(True)
        self.data_table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table_view.setSortingEnabled(True)
        self.data_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.data_table_view.horizontalHeader().setStretchLastSection(True)
        self.data_table_view.horizontalHeader().setDefaultSectionSize(120)
        self.data_table_view.horizontalHeader().setMinimumSectionSize(40)
        self.data_table_view.verticalHeader().setVisible(True)
        self.data_table_view.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.data_table_view.verticalHeader().setDefaultSectionSize(24)

        data_tab_layout.addWidget(self.data_table_view)
        tab1_idx = self.tab_widget.addTab(self.data_tab, "Data Table")
        if os.path.exists(ICON_TABLE):
            self.tab_widget.setTabIcon(tab1_idx, QIcon(ICON_TABLE))

        self.summary_tab = QWidget()
        self.summary_tab.setObjectName("SummaryTableTab")
        summary_tab_layout = QVBoxLayout(self.summary_tab)
        summary_tab_layout.setContentsMargins(0,0,0,0)

        self.summary_table = QTableWidget(self)
        self.summary_table.setObjectName("SummaryTableView")
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary_table.setAlternatingRowColors(True)
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.horizontalHeader().setDefaultSectionSize(140)
        self.summary_table.verticalHeader().setVisible(False)

        summary_tab_layout.addWidget(self.summary_table)
        tab2_idx = self.tab_widget.addTab(self.summary_tab, "Data Summary")
        if os.path.exists(ICON_SUMMARY):
            self.tab_widget.setTabIcon(tab2_idx, QIcon(ICON_SUMMARY))

        self._apply_stylesheet()
        self.show_empty_message()

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            #DataViewerWidget { background-color: #f8f9fa; }
            #DataViewerTabWidget::pane {
                border: 1px solid #d8dcdf; border-top: none; background: white;
                border-bottom-left-radius: 6px; border-bottom-right-radius: 6px;
            }
            #DataViewerTabWidget QTabBar::tab {
                background: #eef1f3; border: 1px solid #d8dcdf; border-bottom: none;
                border-top-left-radius: 5px; border-top-right-radius: 5px;
                padding: 8px 16px; margin-right: 1px; color: #495057; font-weight: bold; min-width: 90px;
            }
            #DataViewerTabWidget QTabBar::tab:selected { background: #4a90e2; color: white; }
            #DataViewerTabWidget QTabBar::tab:hover:!selected { background: #dce4f0; color: #2c3e50; }
            QTableView, QTableWidget {
                border: 1px solid #d8dcdf; gridline-color: #e4e7eb; font-size: 12px;
                selection-background-color: #b3d9ff; selection-color: #181818;
            }
            QTableView::item, QTableWidget::item {
                padding: 4px 6px; border-bottom: 1px solid #f2f4f6;
            }
            QHeaderView::section {
                background-color: #eef1f3; color: #343a40; padding: 6px 4px;
                border: 1px solid #d8dcdf; border-bottom-width: 2px;
                font-size: 12px; font-weight: bold;
            }
            QTableView QHeaderView::section:horizontal { border-top: none; }
            QTableView QHeaderView::section:vertical { border-left: none; }
            #SummaryTableView QHeaderView::section { background-color: #4a90e2; color: white; }
        """)

    def load_data(self, data_frame: pd.DataFrame):
        if not isinstance(data_frame, pd.DataFrame):
            self.show_error_message("Invalid data: Expected a pandas DataFrame.")
            self.data = None; self.clear_views(); return

        self.data = data_frame.copy()
        if self.data.empty:
            self.show_empty_message("Loaded data is empty.")
        else:
            logger.info(f"DataViewer: Displaying data with shape {self.data.shape}")
            self._display_data_table()
            self._display_summary_table()
            if self.parent() and hasattr(self.parent(), 'status_label'):
                 self.parent().status_label.setText(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns.")

    def clear_views(self):
        self.data_table_view.setModel(PandasTableModel(pd.DataFrame()))
        self.summary_table.setRowCount(0); self.summary_table.setColumnCount(0)
        self.show_empty_message("Views cleared. Import data to begin.")

    def show_empty_message(self, message="No data loaded. Please import a dataset."):
        empty_df_for_model = pd.DataFrame({'Status': [message]})
        model = PandasTableModel(empty_df_for_model)
        self.data_table_view.setModel(model)
        if model.columnCount() > 0:
            self.data_table_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
        self.summary_table.setRowCount(1); self.summary_table.setColumnCount(1)
        self.summary_table.setHorizontalHeaderLabels(["Status"])
        self.summary_table.setItem(0,0, QTableWidgetItem("No data to summarize."))
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def show_error_message(self, message: str):
        error_df_for_model = pd.DataFrame({'Error': [message]})
        model = PandasTableModel(error_df_for_model)
        self.data_table_view.setModel(model)
        if model.columnCount() > 0:
            self.data_table_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        
        self.summary_table.setRowCount(1); self.summary_table.setColumnCount(1)
        self.summary_table.setHorizontalHeaderLabels(["Error"])
        item = QTableWidgetItem(message)
        item.setForeground(QColor("red"))
        self.summary_table.setItem(0,0, item)
        self.summary_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def _display_data_table(self):
        if self.data is not None and not self.data.empty:
            model = PandasTableModel(self.data)
            self.data_table_view.setModel(model)
        else:
            self.show_empty_message("No data to display in table.")

    def _display_summary_table(self):
        if self.data is None or self.data.empty:
            self.show_empty_message("No data to summarize."); return

        summary_stats = []
        try:
            for col_name in self.data.columns:
                col_data = self.data[col_name]
                stats = {'Column': col_name, 'Data Type': str(col_data.dtype)}
                stats['Null Values'] = f"{int(col_data.isnull().sum())} ({col_data.isnull().mean()*100:.1f}%)"
                
                if pd.api.types.is_numeric_dtype(col_data):
                    desc = col_data.describe()
                    stats['Count'] = int(desc.get('count', 0))
                    stats['Mean'] = f"{desc.get('mean', np.nan):.3f}"
                    stats['Std Dev'] = f"{desc.get('std', np.nan):.3f}"
                    stats['Min'] = f"{desc.get('min', np.nan):.3f}"
                    stats['25%'] = f"{desc.get('25%', np.nan):.3f}"
                    stats['Median'] = f"{desc.get('50%', np.nan):.3f}"
                    stats['75%'] = f"{desc.get('75%', np.nan):.3f}"
                    stats['Max'] = f"{desc.get('max', np.nan):.3f}"
                    stats['Skewness'] = f"{col_data.skew():.3f}" if not col_data.empty and col_data.nunique() > 1 else "N/A"
                else:
                    desc = col_data.describe(include='all')
                    stats['Count'] = int(desc.get('count', 0))
                    stats['Unique'] = int(desc.get('unique', 0))
                    stats['Top (Mode)'] = str(desc.get('top', 'N/A'))[:50] + ('...' if len(str(desc.get('top', 'N/A'))) > 50 else '')
                    stats['Freq of Top'] = int(desc.get('freq', 0))
                summary_stats.append(stats)
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}", exc_info=True)
            self.show_error_message(f"Error in summary generation: {e}"); return

        if not summary_stats:
            self.show_empty_message("Could not generate summary statistics."); return

        ordered_headers = ['Column', 'Data Type', 'Count', 'Null Values', 'Unique',
                           'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25%', '75%',
                           'Skewness', 'Top (Mode)', 'Freq of Top']
        all_possible_keys = set()
        for s in summary_stats: all_possible_keys.update(s.keys())
        final_headers = [h for h in ordered_headers if h in all_possible_keys]
        final_headers += [k for k in all_possible_keys if k not in final_headers]

        self.summary_table.setRowCount(len(summary_stats))
        self.summary_table.setColumnCount(len(final_headers))
        self.summary_table.setHorizontalHeaderLabels(final_headers)

        header_font = QFont(); header_font.setBold(True)
        self.summary_table.horizontalHeader().setFont(header_font)
        for i, header_text in enumerate(final_headers):
            header_item = self.summary_table.horizontalHeaderItem(i)
            if header_item: header_item.setToolTip(header_text)

        for row_idx, col_summary in enumerate(summary_stats):
            for col_idx, header_name in enumerate(final_headers):
                value = col_summary.get(header_name, "N/A")
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.summary_table.setItem(row_idx, col_idx, item)
        
        self.summary_table.resizeRowsToContents()

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    
    data = {
        'NumericColumnWithVeryLongNameIndeed': np.random.randn(100) * 100,
        'Another_Numeric_Col_Integer': np.random.randint(0, 1000, 100),
        'Categorical_Variable_With_Many_Levels_And_A_Long_Name': np.random.choice(['Alpha particle', 'Beta decay', 'Gamma radiation', 'Delta wave', np.nan, 'Epsilon transition', 'Zeta potential', 'Eta meson', 'Theta state', 'Iota subscript', 'Kappa coefficient'], 100),
        'Boolean_Flags_Maybe_With_Nulls': np.random.choice([True, False, None], 100, p=[0.45, 0.45, 0.1]),
        'Transaction_Date_Time_Column': pd.to_datetime(pd.date_range(start='1/1/2022', periods=100, freq='12H30min5S')),
        'Mixed_Data_Type_Column_Example': [100, 'String Value Two', 300.50, True, 'String Value Five', 600, None, 800.75, 'String Value Nine', 1000] * 10,
        'Very_Long_Descriptive_Text_Column_For_Width_Testing': ['This is an example of a very long string that might appear in a dataset column to test how the table handles width ' + str(i) for i in range(100)],
        'Column_With_All_NaN_Values': [np.nan] * 100,
        'Almost_All_NaN_With_One_Value': [np.nan]*99 + [123.456]
    }
    df = pd.DataFrame(data)
    df.iloc[np.random.choice(df.index, size=15, replace=False), 0] = np.nan 
    df.iloc[np.random.choice(df.index, size=5, replace=False), 2] = None 

    viewer = DataViewer()
    viewer.load_data(df)

    viewer.setWindowTitle("Machine Learning Studio - Data Viewer Test (NumPy 2.0 Fix)")
    viewer.setGeometry(50, 50, 1200, 750) 
    viewer.show()
    sys.exit(app.exec_())