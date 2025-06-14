import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

def handle_missing_values(
    data: pd.DataFrame,
    numeric_strategy: str = 'median',
    numeric_specific_value: float = 0.0,
    categorical_strategy: str = 'mode',
    categorical_constant_value: str = 'Unknown',
    drop_col_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Handles missing values in a DataFrame with more advanced strategies.

    Args:
        data: pandas DataFrame.
        numeric_strategy: Strategy for numeric columns.
        numeric_specific_value: Value to use if numeric_strategy is 'specific'.
        categorical_strategy: Strategy for object/categorical columns.
        categorical_constant_value: Value to use if categorical_strategy is 'constant'.
        drop_col_threshold: Threshold (0.0 to 1.0) to drop columns with missing values.

    Returns:
        pandas DataFrame with missing values handled.
    """
    df = data.copy()
    print(f"[Util] Initial shape before handling missing values: {df.shape}")

    cols_to_drop_na = df.columns[df.isnull().mean() > drop_col_threshold].tolist()
    if cols_to_drop_na:
        df.drop(columns=cols_to_drop_na, inplace=True)
        print(f"[Util] Dropped columns due to >{drop_col_threshold*100}% NA: {cols_to_drop_na}")

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            if numeric_strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif numeric_strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            elif numeric_strategy == 'mode':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else numeric_specific_value, inplace=True)
            elif numeric_strategy == 'zero':
                df[col].fillna(0, inplace=True)
            elif numeric_strategy == 'specific':
                df[col].fillna(numeric_specific_value, inplace=True)
            elif numeric_strategy == 'drop_col':
                if col in df.columns: df.drop(columns=[col], inplace=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            if categorical_strategy == 'mode':
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else categorical_constant_value, inplace=True)
            elif categorical_strategy == 'constant':
                df[col].fillna(categorical_constant_value, inplace=True)
            elif categorical_strategy == 'drop_col':
                 if col in df.columns: df.drop(columns=[col], inplace=True)

    if numeric_strategy == 'drop_row' or categorical_strategy == 'drop_row':
        df.dropna(axis=0, how='any', inplace=True)
        print(f"[Util] Dropped rows with any remaining NA values.")
    
    print(f"[Util] Shape after handling missing values: {df.shape}")
    return df

def remove_duplicate_rows(data: pd.DataFrame, keep='first') -> pd.DataFrame:
    """Removes duplicate rows from the DataFrame."""
    df = data.copy()
    initial_rows = len(df)
    df.drop_duplicates(keep=keep, inplace=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"[Util] Removed {rows_removed} duplicate row(s). Shape after: {df.shape}")
    return df

def detect_outliers_iqr(data: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detects outliers in a numeric column using the IQR method.
    Returns a boolean Series indicating outlier status.
    """
    if column not in data.columns or not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Column '{column}' must be numeric and exist in DataFrame.")
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    print(f"[Util] Outliers detected in '{column}': {outliers.sum()} rows.")
    return outliers

def treat_outliers_iqr(data: pd.DataFrame, column: str, method: str = 'cap', threshold: float = 1.5) -> pd.DataFrame:
    """
    Treats outliers in a numeric column using IQR.
    Methods: 'cap' (winsorize), 'remove' (drop rows).
    """
    df = data.copy()
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        print(f"[Util] Warning: Column '{column}' for outlier treatment is not numeric or not found. Skipping.")
        return df

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    if method == 'cap':
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        print(f"[Util] Outliers in '{column}' capped at [{lower_bound:.2f}, {upper_bound:.2f}].")
    elif method == 'remove':
        initial_rows = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"[Util] Removed {rows_removed} rows due to outliers in '{column}'.")
    else:
        raise ValueError("Invalid outlier treatment method. Choose 'cap' or 'remove'.")
    return df

def encode_categorical_features(
    data: pd.DataFrame, 
    columns_to_encode: list = None, 
    method: str = 'label',
    drop_first_dummy: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical features.

    Args:
        data: pandas DataFrame.
        columns_to_encode: List of column names to encode. If None, attempts to encode all 'object' type columns.
        method: 'label' for LabelEncoding, 'onehot' for OneHotEncoding (sklearn), 'dummy' for pd.get_dummies.
        drop_first_dummy: If True and method is 'dummy' or 'onehot', drops the first category to avoid multicollinearity.

    Returns:
        Tuple of (encoded_df, encoders_or_columns_map).
        For 'label', second element is a dict of {column_name: LabelEncoder_instance}.
        For 'onehot'/'dummy', second element is a dict of {original_column: [new_dummy_columns]}.
    """
    df = data.copy()
    output_info = {}

    if columns_to_encode is None:
        columns_to_encode = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not columns_to_encode:
        print("[Util] No categorical columns specified or found for encoding.")
        return df, output_info

    for col in columns_to_encode:
        if col not in df.columns:
            print(f"[Util] Warning: Column '{col}' not found for encoding. Skipping.")
            continue

        if method == 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            output_info[col] = le
            print(f"[Util] Label encoded column '{col}'.")
        
        elif method in ['onehot', 'dummy']:
            if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 25:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first_dummy)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                output_info[col] = dummies.columns.tolist()
                print(f"[Util] One-hot/Dummy encoded column '{col}' into {len(dummies.columns)} new columns.")
            else:
                print(f"[Util] Warning: Column '{col}' has too many unique values ({df[col].nunique()}) for one-hot encoding. Skipping or consider Label Encoding.")
        else:
            raise ValueError("Invalid encoding method. Choose 'label', 'onehot', or 'dummy'.")
            
    return df, output_info


def transform_skewed_features(
    data: pd.DataFrame, 
    columns_to_transform: list = None, 
    method: str = 'log',
    skew_threshold: float = 0.75
) -> pd.DataFrame:
    """
    Applies transformations to reduce skewness in numeric features.
    """
    df = data.copy()
    if columns_to_transform is None:
        columns_to_transform = df.select_dtypes(include=np.number).columns.tolist()

    for col in columns_to_transform:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            print(f"[Util] Warning: Column '{col}' for skew transformation is not numeric or not found. Skipping.")
            continue

        skewness = df[col].skew()
        if abs(skewness) > skew_threshold:
            print(f"[Util] Column '{col}' has skewness {skewness:.2f}. Applying '{method}' transformation.")
            if method == 'log':
                if df[col].min() <= 0:
                    df[col] = np.log1p(df[col] - df[col].min())
                    print(f"[Util] Applied log1p transformation to '{col}' after shifting.")
                else:
                    df[col] = np.log(df[col])
            elif method == 'sqrt':
                 if df[col].min() < 0:
                    print(f"[Util] Warning: Sqrt transform cannot be applied to negative values in '{col}'. Skipping.")
                    continue
                 df[col] = np.sqrt(df[col])
            elif method == 'boxcox':
                if df[col].min() <= 0:
                    print(f"[Util] Warning: Box-Cox requires positive values for '{col}'. Attempting shift or skipping.")
                    if df[col].min() < 1:
                        shifted_data = df[col] + (1 - df[col].min())
                        if shifted_data.min() > 0:
                            df[col], _ = stats.boxcox(shifted_data)
                            print(f"[Util] Applied Box-Cox to '{col}' after shifting.")
                        else:
                            print(f"[Util] Could not apply Box-Cox to '{col}' even after shift. Skipping.")
                            continue
                    else:
                        try:
                            df[col], _ = stats.boxcox(df[col])
                        except ValueError as e:
                            print(f"[Util] Box-Cox transformation failed for '{col}': {e}. Skipping.")
                            continue
                else:
                    df[col], _ = stats.boxcox(df[col])
            else:
                raise ValueError("Invalid transformation method. Choose 'log', 'sqrt', or 'boxcox'.")
        else:
            print(f"[Util] Column '{col}' skewness ({skewness:.2f}) is within threshold. No transformation applied.")
    return df

def scale_features(
    data: pd.DataFrame, 
    columns_to_scale: list = None, 
    method: str = 'standard'
) -> tuple[pd.DataFrame, dict]:
    """
    Scales numeric features.
    Returns scaled DataFrame and a dictionary of fitted scalers.
    """
    df = data.copy()
    fitted_scalers = {}

    if columns_to_scale is None:
        columns_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    
    if not columns_to_scale:
        print("[Util] No numeric columns specified or found for scaling.")
        return df, fitted_scalers

    for col in columns_to_scale:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            print(f"[Util] Warning: Column '{col}' for scaling is not numeric or not found. Skipping.")
            continue
        
        scaler = None
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', or 'robust'.")
        
        col_data = df[col].values.reshape(-1, 1)
        df[col] = scaler.fit_transform(col_data)
        fitted_scalers[col] = scaler
        print(f"[Util] Scaled column '{col}' using {method} scaler.")
        
    return df, fitted_scalers

if __name__ == '__main__':
    sample_data = {
        'Age': [25, 30, np.nan, 22, 45, 30, 28, 35, 40, 22, 120],
        'City': ['NY', 'LA', 'NY', 'SF', np.nan, 'LA', 'Boston', 'NY', 'SF', 'NY', 'LA'],
        'Salary': [50000, 60000, 55000, np.nan, 120000, 60000, 70000, 80000, 110000, 55000, 500000],
        'Experience': [2, 5, 3, 1, 15, 5, 4, 10, 12, 1, 2],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M']
    }
    test_df = pd.DataFrame(sample_data)
    test_df_original = test_df.copy()

    print("--- Original DataFrame ---")
    print(test_df.head())
    print("\n--- Missing Value Info ---")
    print(test_df.isnull().sum())

    df_cleaned_missing = handle_missing_values(
        test_df, 
        numeric_strategy='median', 
        categorical_strategy='mode',
        drop_col_threshold=0.6
    )
    print("\n--- After Handling Missing Values ---")
    print(df_cleaned_missing.head())
    print(df_cleaned_missing.isnull().sum())

    df_no_duplicates = remove_duplicate_rows(df_cleaned_missing)
    
    df_outliers_treated = treat_outliers_iqr(df_no_duplicates, 'Age', method='cap')
    print("\n--- After Treating Outliers in 'Age' (Capping) ---")
    print(df_outliers_treated[['Age']].describe())
    
    df_encoded, encoders = encode_categorical_features(df_outliers_treated, columns_to_encode=['City', 'Gender'], method='dummy')
    print("\n--- After Encoding Categorical Features (Dummy) ---")
    print(df_encoded.head())
    
    print("\n--- Salary Skewness Before Transformation ---")
    print(df_encoded['Salary'].skew())
    df_transformed = transform_skewed_features(df_encoded, columns_to_transform=['Salary'], method='log', skew_threshold=0.5)
    print("\n--- Salary Skewness After Log Transformation ---")
    print(df_transformed['Salary'].skew())
    
    numeric_cols_for_scaling = df_transformed.select_dtypes(include=np.number).columns.tolist()
    df_scaled, scalers = scale_features(df_transformed, columns_to_scale=numeric_cols_for_scaling, method='standard')
    print("\n--- After Scaling Numeric Features (StandardScaler) ---")
    print(df_scaled.head())
    
    print("\n--- Final Processed DataFrame Description ---")
    print(df_scaled.describe(include='all'))