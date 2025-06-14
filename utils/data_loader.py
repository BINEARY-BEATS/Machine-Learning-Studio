# utils/data_loader.py
# Author: Saeed Ur Rehman
from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
import chardet
import os
import logging

logger = logging.getLogger(__name__)

class DataLoaderThread(QThread):
    """
    A thread to handle data loading asynchronously, preventing UI freezes.
    Emits progress updates and either the loaded DataFrame or an error message.
    """
    progress_updated = pyqtSignal(int)
    data_loaded = pyqtSignal(object)
    loading_error = pyqtSignal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        logger.info(f"DataLoaderThread initialized for file: {self.file_path}")

    def run(self):
        try:
            self.progress_updated.emit(0)
            if not self.file_path or not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            file_extension = os.path.splitext(self.file_path)[1].lower()
            logger.info(f"Attempting to load file with extension: {file_extension}")
            self.progress_updated.emit(10)

            data = None
            if file_extension == ".csv":
                data = self._load_csv()
            elif file_extension in [".xls", ".xlsx"]:
                data = self._load_excel()
            elif file_extension == ".json":
                data = self._load_json()
            else:
                raise ValueError(f"Unsupported file type: '{file_extension}'. Please use CSV, Excel, or JSON.")

            if data is not None: # data can be an empty DataFrame
                self.progress_updated.emit(100)
                self.data_loaded.emit(data)
                if data.empty:
                    logger.warning(f"Successfully loaded '{os.path.basename(self.file_path)}' but it is empty or header-only. Shape: {data.shape}")
                else:
                    logger.info(f"Successfully loaded data from {os.path.basename(self.file_path)}. Shape: {data.shape}")
            else: # Should ideally not happen if loaders return empty DFs
                raise ValueError("Data loading resulted in an unexpected None dataset.")

        except Exception as e:
            logger.error(f"Error during data loading for '{self.file_path}': {e}", exc_info=True)
            self.progress_updated.emit(0)
            self.loading_error.emit(f"Failed to load '{os.path.basename(self.file_path)}':\n{str(e)}")

    def _detect_encoding(self) -> str | None:
        logger.debug(f"Detecting encoding for {self.file_path}...")
        try:
            with open(self.file_path, 'rb') as file:
                sample_size = min(100000, os.path.getsize(self.file_path)) # Cap sample size
                if sample_size == 0: # Empty file
                    logger.warning(f"File '{self.file_path}' is empty, cannot detect encoding.")
                    return 'utf-8' # Default for empty file
                raw_data = file.read(sample_size)

            if not raw_data: # If sample is empty for some reason
                return 'utf-8'

            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            logger.debug(f"Chardet result for {self.file_path}: encoding='{encoding}', confidence={confidence:.2f}")

            if encoding and confidence > 0.70: # Slightly more lenient confidence
                # Handle common misinterpretations or specific aliases
                if encoding.lower() in ['ascii', 'iso-8859-1', 'windows-1252'] and confidence < 0.95:
                    try:
                        raw_data.decode('utf-8') # Check if it could be UTF-8
                        logger.info(f"Detected '{encoding}' for '{self.file_path}', but it also decodes as UTF-8. Preferring UTF-8.")
                        return 'utf-8'
                    except UnicodeDecodeError:
                        logger.info(f"Sticking with detected encoding: {encoding} for '{self.file_path}'.")
                        return encoding
                return encoding
            logger.warning(f"Low confidence ({confidence:.2f}) for detected encoding '{encoding}' for '{self.file_path}'. Defaulting to utf-8.")
            return 'utf-8'
        except FileNotFoundError:
            logger.error(f"File not found during encoding detection: {self.file_path}")
            raise # Re-raise to be caught by run()
        except Exception as e:
            logger.error(f"Error detecting encoding for '{self.file_path}', defaulting to utf-8: {e}")
            return 'utf-8'

    def _load_csv(self) -> pd.DataFrame:
        encoding = self._detect_encoding()
        if encoding is None: # Should ideally not happen if _detect_encoding defaults
            raise ValueError("Could not determine file encoding.")
        logger.info(f"Loading CSV file: {self.file_path} with detected encoding: {encoding}")
        self.progress_updated.emit(20)

        try:
            # Attempt to read header first to return an empty DataFrame with columns if file is header-only
            header_df = pd.read_csv(self.file_path, encoding=encoding, nrows=0, low_memory=False)
        except pd.errors.EmptyDataError:
            logger.warning(f"CSV file '{self.file_path}' is completely empty (no header, no data).")
            return pd.DataFrame() # Return an empty DataFrame
        except Exception as header_err:
            logger.warning(f"Could not read CSV header for '{self.file_path}': {header_err}. Proceeding with full read attempt.")
            header_df = pd.DataFrame() # Fallback

        self.progress_updated.emit(30)
        chunks = []
        total_file_size = os.path.getsize(self.file_path)
        bytes_processed_estimate = 0

        try:
            chunk_size = 50000 # Adjust based on typical memory, larger can be faster for I/O
            for i, chunk in enumerate(pd.read_csv(self.file_path, chunksize=chunk_size, encoding=encoding, low_memory=False)):
                chunks.append(chunk)
                if total_file_size > 0:
                    # Estimate progress based on number of chunks if file size is large
                    # This is a rough guide as chunk memory size varies.
                    # Assuming each chunk is roughly chunk_size * avg_bytes_per_row
                    # A simpler way for progress if number of rows isn't pre-calculated:
                    progress = 30 + int( ( (i + 1) * chunk_size * 50 ) / total_file_size * 65 ) # 50 is a guess for avg row length
                    self.progress_updated.emit(min(progress, 95))
                else: # Fallback for very small files or if size is 0
                    self.progress_updated.emit(30 + (i % 14) * 5)


            if not chunks: # File might have header but no data rows
                logger.warning(f"CSV file '{self.file_path}' has a header but no data rows.")
                return header_df # Return empty DataFrame with columns from header

            full_df = pd.concat(chunks, ignore_index=True)
            logger.info(f"CSV '{self.file_path}' loaded. Shape: {full_df.shape}")
            return full_df

        except UnicodeDecodeError as ude:
            logger.error(f"UnicodeDecodeError for '{self.file_path}' with encoding '{encoding}': {ude}")
            raise ValueError(f"Encoding error with '{encoding}'. The file might be corrupted or not encoded as detected. Try verifying the file or specifying encoding if known.")
        except pd.errors.EmptyDataError: # Should be caught by header read, but as a safeguard
            logger.warning(f"CSV file '{self.file_path}' is empty.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Generic error reading CSV '{self.file_path}': {e}", exc_info=True)
            raise ValueError(f"Could not read CSV file. Ensure it's a valid CSV format.\nError: {str(e)}")

    def _load_excel(self) -> pd.DataFrame:
        logger.info(f"Loading Excel file: {self.file_path}")
        self.progress_updated.emit(25)
        try:
            excel_file = pd.ExcelFile(self.file_path, engine='openpyxl')
            self.progress_updated.emit(60)

            if not excel_file.sheet_names:
                logger.warning(f"Excel file '{self.file_path}' contains no sheets.")
                return pd.DataFrame() # Return empty if no sheets

            # Load the first sheet. This could be made configurable in the UI.
            first_sheet_name = excel_file.sheet_names[0]
            data = excel_file.parse(first_sheet_name)
            self.progress_updated.emit(95)
            logger.info(f"Excel file '{self.file_path}' (Sheet: '{first_sheet_name}') loaded. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error reading Excel file '{self.file_path}': {e}", exc_info=True)
            raise ValueError(f"Could not read Excel file. Ensure it's a valid XLSX/XLS format, not corrupted, and not password-protected.\nError: {str(e)}")

    def _load_json(self) -> pd.DataFrame:
        logger.info(f"Loading JSON file: {self.file_path}")
        self.progress_updated.emit(25)
        try:
            data = None
            # Try common JSON structures, starting with lines=True for potentially large files
            try:
                data = pd.read_json(self.file_path, orient='records', lines=True)
                logger.info(f"JSON '{self.file_path}' loaded successfully with orient='records', lines=True.")
            except ValueError:
                logger.debug(f"Failed to load JSON '{self.file_path}' with lines=True. Trying other orientations.")
                try:
                    data = pd.read_json(self.file_path, orient='records') # Common for array of objects
                    logger.info(f"JSON '{self.file_path}' loaded successfully with orient='records'.")
                except ValueError:
                    logger.debug(f"Failed to load JSON '{self.file_path}' with orient='records'. Trying default pd.read_json().")
                    # Pandas' default read_json tries to infer structure
                    data = pd.read_json(self.file_path)
                    logger.info(f"JSON '{self.file_path}' loaded successfully with default pandas inference.")
            
            if data is None: # Should not happen if one of the try blocks succeeded
                 raise ValueError("Unable to parse JSON structure.")

            self.progress_updated.emit(95)
            logger.info(f"JSON loaded from '{self.file_path}'. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error reading JSON file '{self.file_path}': {e}", exc_info=True)
            raise ValueError(f"Could not read JSON file. Ensure it's valid (e.g., records array, JSON lines, or pandas-compatible).\nError: {str(e)}")