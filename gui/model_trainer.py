import sys
import os
import logging
import json
import tempfile
import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QMessageBox, QHeaderView,
    QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem, QListWidget,
    QAbstractItemView, QProgressBar, QGroupBox, QScrollArea, QSizePolicy, QTabWidget,
    QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QFormLayout, QDialog, QMenu,
    QDialogButtonBox, QApplication, QStackedWidget, QLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QFont, QColor, QIcon, QShowEvent, QStandardItemModel, QStandardItem

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors as reportlab_colors

logger = logging.getLogger(__name__)

def _log_model_not_implemented(model_name_display, **kwargs):
    logger.error(f"Training function for '{model_name_display}' is a placeholder and not implemented.")
    from sklearn.base import BaseEstimator
    class DummyModel(BaseEstimator):
        _model_name_display = model_name_display
        def fit(self, X, y=None):
            raise NotImplementedError(f"{self._model_name_display} training not implemented.")
        def predict(self, X):
            raise NotImplementedError(f"{self._model_name_display} prediction not implemented.")
    return DummyModel()

_model_func_placeholders = {}
_model_func_placeholders["train_ridge_regression"] = lambda **kwargs: _log_model_not_implemented("Ridge Regression", **kwargs)
_model_func_placeholders["train_lasso_regression"] = lambda **kwargs: _log_model_not_implemented("Lasso Regression", **kwargs)
_model_func_placeholders["train_huber_regression"] = lambda **kwargs: _log_model_not_implemented("Huber Regression", **kwargs)
_model_func_placeholders["train_quantile_regression"] = lambda **kwargs: _log_model_not_implemented("Quantile Regression", **kwargs)
_model_func_placeholders["train_knn_regressor"] = lambda **kwargs: _log_model_not_implemented("KNN Regressor", **kwargs)
_model_func_placeholders["train_mlp_regressor"] = lambda **kwargs: _log_model_not_implemented("MLP Regressor", **kwargs)
_model_func_placeholders["train_gaussian_process_regressor"] = lambda **kwargs: _log_model_not_implemented("Gaussian Process Regressor", **kwargs)
_model_func_placeholders["train_gradient_boosting_regressor"] = lambda **kwargs: _log_model_not_implemented("Gradient Boosting Regressor", **kwargs)
_model_func_placeholders["train_bagging_regressor"] = lambda **kwargs: _log_model_not_implemented("Bagging Regressor", **kwargs)
_model_func_placeholders["train_pls_regression"] = lambda **kwargs: _log_model_not_implemented("PLSRegression", **kwargs)
_model_func_placeholders["train_isotonic_regression"] = lambda **kwargs: _log_model_not_implemented("Isotonic Regression", **kwargs)
_model_func_placeholders["train_elasticnet_regression"] = lambda **kwargs: _log_model_not_implemented("Elastic Net", **kwargs)
_model_func_placeholders["train_bayesian_ridge_regression"] = lambda **kwargs: _log_model_not_implemented("Bayesian Ridge", **kwargs)
_model_func_placeholders["train_lightgbm_regressor"] = lambda **kwargs: _log_model_not_implemented("LightGBM", **kwargs)

try:
    from utils.model_utils import (
        train_linear_regression, train_decision_tree_regressor,
        train_random_forest_regressor, train_xgboost_regressor, train_svr,
        train_ridge_regression, train_lasso_regression, train_huber_regression,
        train_quantile_regression, train_knn_regressor, train_mlp_regressor,
        train_gaussian_process_regressor, train_gradient_boosting_regressor,
        train_bagging_regressor, train_pls_regression, train_isotonic_regression,
        train_elasticnet_regression, train_bayesian_ridge_regression, train_lightgbm_regressor
    )
    from utils.visual_utils import VisualUtils
    logger.info("Successfully imported all model training functions from utils.model_utils.")
except ImportError as e:
    logger.warning(f"Partial or failed import from model_utils: {e}. Using placeholders for missing functions.")
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)

    _imported_successfully = []
    _failed_to_import = []
    _essential_models_map = {
        "train_linear_regression": "Linear Regression",
        "train_decision_tree_regressor": "Decision Tree",
        "train_random_forest_regressor": "Random Forest",
        "train_xgboost_regressor": "XGBoost",
        "train_svr": "SVR"
    }
    _additional_models_map = {
        "train_ridge_regression": "Ridge Regression",
        "train_lasso_regression": "Lasso Regression",
        "train_huber_regression": "Huber Regression",
        "train_quantile_regression": "Quantile Regression",
        "train_knn_regressor": "KNN Regressor",
        "train_mlp_regressor": "MLP Regressor",
        "train_gaussian_process_regressor": "Gaussian Process Regressor",
        "train_gradient_boosting_regressor": "Gradient Boosting Regressor",
        "train_bagging_regressor": "Bagging Regressor",
        "train_pls_regression": "PLSRegression",
        "train_isotonic_regression": "Isotonic Regression",
        "train_elasticnet_regression": "Elastic Net",
        "train_bayesian_ridge_regression": "Bayesian Ridge",
        "train_lightgbm_regressor": "LightGBM"
    }
         


    _all_models_to_try_map = {**_essential_models_map, **_additional_models_map}

    for func_name, display_name in _all_models_to_try_map.items():
        try:
            exec(f"from utils.model_utils import {func_name}", globals())
            _imported_successfully.append(func_name)
        except ImportError:
            _failed_to_import.append(func_name)
            globals()[func_name] = lambda n=display_name, **kwargs: _log_model_not_implemented(n, **kwargs)


    if _failed_to_import:
        logger.warning(f"Failed to import the following model functions (placeholders used): {', '.join(_failed_to_import)}")
    if not _imported_successfully and _essential_models_map.keys() <= set(_failed_to_import):
        logger.critical("Failed to import ANY ESSENTIAL model_utils functions even after path adjustment.")

    try:
        from utils.visual_utils import VisualUtils
    except ImportError:
        logger.critical("Failed to import visual_utils even after path adjustment. Using dummy VisualUtils.")
        class VisualUtils:
            @staticmethod
            def _create_base_figure(title, figsize):
                fig_mpl, ax_mpl = plt.subplots(figsize=figsize)
                fig_mpl.suptitle(title, fontsize=10, y=0.98)
                return fig_mpl, ax_mpl


try:
    from utils.data_cleaner import handle_missing_and_duplicates
except ImportError:
    logger.warning("'handle_missing_and_duplicates' not found. Using placeholder.")
    def handle_missing_and_duplicates(data, fill_method_numeric='mean', fill_method_object='mode', remove_duplicates=True):
        data_copy = data.copy()
        for col in data_copy.select_dtypes(include=np.number).columns:
            if data_copy[col].isnull().any():
                if fill_method_numeric == 'median': data_copy[col].fillna(data_copy[col].median(), inplace=True)
                elif fill_method_numeric == 'mean': data_copy[col].fillna(data_copy[col].mean(), inplace=True)
                else: data_copy[col].fillna(0, inplace=True)
        for col in data_copy.select_dtypes(include=['object','category']).columns:
            if data_copy[col].isnull().any():
                mode_val = data_copy[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                data_copy[col].fillna(fill_val, inplace=True)
        if remove_duplicates:
            data_copy.drop_duplicates(inplace=True, ignore_index=True)
        return data_copy

ICON_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "icons")
def get_icon(name):
    path = os.path.join(ICON_PATH_ROOT, name)
    return QIcon(path) if os.path.exists(path) else QIcon()

ICON_APP=get_icon("ai.png"); ICON_TRAIN=get_icon("play-circle.png"); ICON_SAVE=get_icon("save.png"); ICON_LOAD=get_icon("upload-cloud.png")
ICON_DELETE=get_icon("trash-2.png"); ICON_EXPORT=get_icon("file-text.png"); ICON_PREDICT_TAB=get_icon("activity.png")
ICON_PREDICT_BTN=get_icon("bar-chart-2.png"); ICON_BATCH_PREDICT_FILE=get_icon("file-plus.png"); ICON_BATCH_PREDICT_RUN=get_icon("zap.png")
ICON_BATCH_SAVE_RESULTS = get_icon("download.png"); ICON_SELECT_FEATURES=get_icon("list.png"); ICON_TAB_TRAINING=get_icon("sliders.png")
ICON_TAB_EVALUATION=get_icon("clipboard.png"); ICON_TAB_MODELS=get_icon("archive.png"); ICON_INFO=get_icon("info.png")
ICON_DOWN_ARROW_STR="chevron-down.png"; ICON_CHECKED_STR="check-square.png"; ICON_UNCHECKED_STR="square.png"; ICON_BUSY=get_icon("loader.png")


MODEL_CATEGORIES = {
    "Auto Select (via CV)": ["Auto Select (via CV)"],
    "Linear Models": ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net", "Polynomial Regression"],
    "Robust Models": ["Huber Regression", "Quantile Regression"],
    "Tree-Based Models": ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
    "Instance-Based Models": ["KNN Regressor"],
    "Kernel-Based Models": ["SVR"],
    "Bayesian Models": ["Bayesian Ridge", "Gaussian Process Regressor"],
    "Neural Networks": ["MLP Regressor"],
    "Ensemble Models": ["Gradient Boosting Regressor", "Bagging Regressor"],
    "Specialized Models": ["PLSRegression", "Isotonic Regression"]
}


class TrainingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    training_complete = pyqtSignal(object, dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, X_data: pd.DataFrame, y_data: pd.Series, model_type: str, hyperparams: dict, parent_trainer_instance):
        super().__init__(parent_trainer_instance)
        self.X_data = X_data.copy()
        self.y_data = y_data.copy()
        self.model_type_selected = model_type
        self.hyperparams_from_ui = hyperparams.copy()
        self.cv_folds = 5
        self.random_state = 42
        self.parent_trainer = parent_trainer_instance

    def _is_placeholder_function(self, func):
        """Checks if a function is one of our placeholders."""
        if func is None: return True
        if hasattr(func, '__name__') and "_log_model_not_implemented" in func.__name__:
            return True
        if hasattr(func, '__closure__') and func.__closure__:
            for cell in func.__closure__:
                if hasattr(cell.cell_contents, '__name__') and \
                   "_log_model_not_implemented" in cell.cell_contents.__name__:
                    return True
        return False

    def _run_auto_select_cv(self, X_train, y_train):
        logger.info("Starting Auto Select (via CV)...")
        candidate_models_setup = {
            "Linear Regression": (train_linear_regression, {}),
            "Ridge Regression": (train_ridge_regression, {'alpha': 1.0}),
            "Lasso Regression": (train_lasso_regression, {'alpha': 1.0}),
            "Decision Tree": (train_decision_tree_regressor, {'max_depth': 5}),
            "Random Forest": (train_random_forest_regressor, {'n_estimators': 30, 'max_depth': 10, 'n_jobs': -1}),
            "SVR": (train_svr, {'kernel': 'rbf', 'C': 1.0}),
            "KNN Regressor": (train_knn_regressor, {'n_neighbors': 5, 'n_jobs': -1}),
            "XGBoost": (train_xgboost_regressor, {'n_estimators': 30, 'max_depth': 3, 'n_jobs': -1}),
            "LightGBM": (train_lightgbm_regressor, {'n_estimators': 30, 'num_leaves':15, 'n_jobs':-1, 'verbosity':-1})
        }

        best_score = -np.inf
        best_model_name = None
        
        valid_candidate_models = {}
        for name, (model_func_ref, params) in candidate_models_setup.items():
            actual_model_func = self.parent_trainer.MODEL_MAP_TRAINING_THREAD.get(name)
            if not self._is_placeholder_function(actual_model_func):
                valid_candidate_models[name] = (actual_model_func, params)
            else:
                logger.warning(f"Auto CV: Skipping {name}, its training function is a placeholder or not found.")

        if not valid_candidate_models:
            logger.error("Auto CV: No valid (non-placeholder) candidate models found for evaluation.")
            self.error_occurred.emit("Auto CV failed: No valid candidate models available.")
            return None
            
        num_candidates_total = len(valid_candidate_models)
        processed_candidates = 0
        progress_start_auto_cv = 10

        for name, (model_func, default_params) in valid_candidate_models.items():
            processed_candidates +=1
            current_progress_percentage = (processed_candidates / num_candidates_total) * 50 if num_candidates_total > 0 else 50
            current_progress = progress_start_auto_cv + int(current_progress_percentage)

            self.progress_updated.emit(current_progress, f"Auto CV: Evaluating {name}...")
            logger.info(f"Auto CV: Evaluating {name}...")
            try:
                model_instance_params = default_params.copy()
                temp_model_instance_for_inspect = model_func() 
                if temp_model_instance_for_inspect is not None and \
                   hasattr(temp_model_instance_for_inspect, '__init__') and \
                   hasattr(temp_model_instance_for_inspect.__init__, '__code__') and \
                   'random_state' in temp_model_instance_for_inspect.__init__.__code__.co_varnames:
                    model_instance_params['random_state'] = self.random_state

                model_instance = model_func(**model_instance_params)
                pipeline = Pipeline([('scaler', StandardScaler()), ('model', model_instance)])
                kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
                mean_score = np.mean(scores)
                logger.info(f"Auto CV: {name} R2 score: {mean_score:.4f}")
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
            except NotImplementedError as nie_cv:
                logger.warning(f"Auto CV: Skipped {name} due to NotImplementedError: {nie_cv}")
            except Exception as e:
                logger.warning(f"Auto CV: Error evaluating {name}: {e}")

        if best_model_name:
            logger.info(f"Auto CV finished. Best model selected: {best_model_name} with R2: {best_score:.4f}")
            self.progress_updated.emit(progress_start_auto_cv + 50, f"Auto CV: Selected {best_model_name}. Preparing full training...")
            return best_model_name
        else:
            logger.error("Auto CV failed to select any model from the valid candidates.")
            self.error_occurred.emit("Auto CV failed: No model performed adequately among valid candidates.")
            return None

    def run(self):
        actual_model_to_train = ""
        try:
            self.progress_updated.emit(5, "Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, test_size=0.2, random_state=self.random_state)
            if X_train.empty or y_train.empty:
                self.error_occurred.emit("Not enough data for training set after split.")
                return

            actual_model_to_train = self.model_type_selected
            current_hyperparams_for_training = self.hyperparams_from_ui.copy()

            if self.model_type_selected == "Auto Select (via CV)":
                selected_model_name_by_cv = self._run_auto_select_cv(X_train, y_train)
                if not selected_model_name_by_cv: return 
                actual_model_to_train = selected_model_name_by_cv
                current_hyperparams_for_training = self.parent_trainer._get_current_hyperparams_for_model(actual_model_to_train)
                current_hyperparams_for_training['polynomial_features'] = self.hyperparams_from_ui.get('polynomial_features', False)
                current_hyperparams_for_training['poly_degree'] = self.hyperparams_from_ui.get('poly_degree', 2)
                logger.info(f"Auto-selected model for full training: {actual_model_to_train} with hyperparams: {current_hyperparams_for_training}")
                self.progress_updated.emit(65, f"Proceeding with full training for auto-selected: {actual_model_to_train}...")
            else:
                self.progress_updated.emit(10, "Setting up pipeline for selected model...")

            model_specific_hyperparams = {k: v for k, v in current_hyperparams_for_training.items() if k not in ['polynomial_features', 'poly_degree', 'auto_select_model']}
            pipeline_steps = []
            feature_names_for_model = X_train.columns.tolist()

            if current_hyperparams_for_training.get('polynomial_features', False) and actual_model_to_train != "Polynomial Regression":
                poly_degree = current_hyperparams_for_training.get('poly_degree', 2)
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
                pipeline_steps.append(('poly', poly))
                try:
                    temp_poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                    temp_poly.fit(X_train) 
                    feature_names_for_model = list(temp_poly.get_feature_names_out(X_train.columns))
                except Exception as e:
                    logger.warning(f"Could not pre-determine polynomial feature names: {e}")
            
            pipeline_steps.append(('scaler', StandardScaler()))

            progress_offset = 65 if self.model_type_selected == "Auto Select (via CV)" else 20
            self.progress_updated.emit(progress_offset, f"Initializing {actual_model_to_train}...")

            if actual_model_to_train not in self.parent_trainer.MODEL_MAP_TRAINING_THREAD:
                self.error_occurred.emit(f"Model '{actual_model_to_train}' not found in training map.")
                return
            
            model_constructor_func = self.parent_trainer.MODEL_MAP_TRAINING_THREAD[actual_model_to_train]
            if self._is_placeholder_function(model_constructor_func):
                self.error_occurred.emit(f"Training failed for {actual_model_to_train}: Model not implemented.")
                return

            final_model_kwargs = model_specific_hyperparams.copy()
            dummy_instance_for_inspect = None 
            try:
                core_kwargs = {k: v for k, v in final_model_kwargs.items() if k not in ['random_state', 'n_jobs', 'oob_score', 'warm_start']}
                dummy_instance_for_inspect = model_constructor_func(**core_kwargs)
            except TypeError: 
                try:
                    dummy_instance_for_inspect = model_constructor_func()
                except Exception as e_dummy_init:
                    logger.warning(f"Could not create dummy instance of {actual_model_to_train} for inspection (attempt 2 - no kwargs): {e_dummy_init}")
                    dummy_instance_for_inspect = None 
            except Exception as e_dummy_init_kw: 
                logger.warning(f"Could not create dummy instance of {actual_model_to_train} with core kwargs for inspection: {e_dummy_init_kw}")
                dummy_instance_for_inspect = None


            if dummy_instance_for_inspect is not None:
                if hasattr(dummy_instance_for_inspect, '__init__') and hasattr(dummy_instance_for_inspect.__init__, '__code__'):
                    model_class_init_params = dummy_instance_for_inspect.__init__.__code__.co_varnames
                    if 'random_state' in model_class_init_params:
                        final_model_kwargs['random_state'] = self.random_state
                    if 'n_jobs' in model_class_init_params:
                        if actual_model_to_train in ["Random Forest", "XGBoost", "LightGBM", "KNN Regressor", "Gradient Boosting Regressor", "Bagging Regressor"]:
                            final_model_kwargs['n_jobs'] = -1
                del dummy_instance_for_inspect 
            else:
                logger.warning(f"Could not inspect __init__ params for {actual_model_to_train} (dummy instance was None). Using provided hyperparams as is.")
                if 'random_state' not in final_model_kwargs and actual_model_to_train not in ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net", "Bayesian Ridge", "Isotonic Regression"]:
                    final_model_kwargs['random_state'] = self.random_state
                if 'n_jobs' not in final_model_kwargs and actual_model_to_train in ["Random Forest", "XGBoost", "LightGBM", "KNN Regressor", "Gradient Boosting Regressor", "Bagging Regressor"]:
                    final_model_kwargs['n_jobs'] = -1


            final_model_instance = model_constructor_func(**final_model_kwargs)
            pipeline_steps.append(('model', final_model_instance))
            pipeline = Pipeline(pipeline_steps)

            self.progress_updated.emit(progress_offset + 10, "Cross-validating main model...")
            cv_r2_scores = cross_val_score(pipeline, X_train, y_train, cv=self.cv_folds, scoring='r2', n_jobs=-1)
            cv_neg_mse_scores = cross_val_score(pipeline, X_train, y_train, cv=self.cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)

            self.progress_updated.emit(progress_offset + 40, f"Training final {actual_model_to_train} model...")
            pipeline.fit(X_train, y_train)

            final_feature_names_after_poly = list(feature_names_for_model) 
            if 'poly' in pipeline.named_steps:
                try:
                    final_feature_names_after_poly = list(pipeline.named_steps['poly'].get_feature_names_out(X_train.columns))
                except Exception as e:
                    logger.warning(f"Could not get final polynomial feature names from fitted pipeline: {e}")
                    model_step = pipeline.named_steps['model']
                    if hasattr(model_step, 'n_features_in_') and len(feature_names_for_model) == model_step.n_features_in_ :
                        final_feature_names_after_poly = feature_names_for_model
                    elif hasattr(model_step, 'n_features_in_'):
                        final_feature_names_after_poly = [f"processed_feature_{i}" for i in range(model_step.n_features_in_)]
                    else:
                        final_feature_names_after_poly = [f"feature_{i}" for i in range(X_train.shape[1])]


            metrics = {
                'train_r2': pipeline.score(X_train, y_train),
                'test_r2': pipeline.score(X_test, y_test),
                'train_mse': mean_squared_error(y_train, pipeline.predict(X_train)),
                'test_mse': mean_squared_error(y_test, pipeline.predict(X_test)),
                'cv_mean_r2': np.mean(cv_r2_scores), 'cv_std_r2': np.std(cv_r2_scores),
                'cv_mean_mse': -np.mean(cv_neg_mse_scores), 'cv_std_mse': np.std(cv_neg_mse_scores),
                'model_type': actual_model_to_train,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'hyperparams': current_hyperparams_for_training,
                'num_features_input_to_pipeline': self.X_data.shape[1],
                'num_features_to_model': len(final_feature_names_after_poly), 
                'num_samples_train': len(X_train), 'num_samples_test': len(X_test)
            }

            self.progress_updated.emit(95, "Extracting feature importances...")
            importance_dict = {}
            model_in_pipeline = pipeline.named_steps['model']
            coefficients = []
            if hasattr(model_in_pipeline, 'feature_importances_'):
                coefficients = model_in_pipeline.feature_importances_
            elif hasattr(model_in_pipeline, 'coef_'):
                coefficients = model_in_pipeline.coef_
            
            if hasattr(coefficients, 'ndim') and coefficients.ndim > 1: 
                coefficients = coefficients.ravel() 

            if len(coefficients) == len(final_feature_names_after_poly):
                importance_dict = dict(zip(final_feature_names_after_poly, coefficients))
            elif len(coefficients) > 0 : 
                logger.warning(f"Feature names ({len(final_feature_names_after_poly)}) and coefficients ({len(coefficients)}) length mismatch for {actual_model_to_train}. Using generic importance keys.")
                importance_dict = {f"importance_{i}": val for i, val in enumerate(coefficients)}

            self.progress_updated.emit(98, "Finalizing training results...")
            self.training_complete.emit(pipeline, metrics, X_train, y_train, X_test, y_test, importance_dict)
            self.progress_updated.emit(100, "Training complete!")

        except NotImplementedError as nie: 
            final_model_type_on_error = actual_model_to_train if actual_model_to_train else self.model_type_selected
            logger.error(f"TrainingThread Error (Model: {final_model_type_on_error}): {nie}", exc_info=False)
            self.error_occurred.emit(f"Training failed for {final_model_type_on_error}: Model not implemented.")
        except Exception as e:
            final_model_type_on_error = actual_model_to_train if actual_model_to_train else self.model_type_selected
            logger.error(f"TrainingThread Error (Model: {final_model_type_on_error}): {e}", exc_info=True)
            self.error_occurred.emit(f"Training failed for {final_model_type_on_error}: {str(e)[:100]}")


class EvaluationPlotThread(QThread):
    plot_generated = pyqtSignal(str, Figure); plot_error = pyqtSignal(str, str)
    def __init__(self, plot_type: str, data_dict: dict, parent=None): super().__init__(parent); self.plot_type=plot_type; self.data_dict=data_dict
    def run(self):
        try:
            fig=None; title_prefix = ""
            if self.plot_type == "feature_importance":
                title_prefix = "Feature Importances / Coefficients"; importance_dict = self.data_dict.get("importance_dict", {})
                fig, ax = VisualUtils._create_base_figure(title_prefix, figsize=(6.5, 4.8))
                if importance_dict:
                    sorted_importances=sorted(importance_dict.items(),key=lambda item:abs(item[1]),reverse=True)[:12];features=[str(item[0])[:25]+('...'if len(str(item[0]))>25 else '') for item in sorted_importances];scores=[item[1] for item in sorted_importances]
                    y_pos=np.arange(len(features));ax.barh(y_pos,scores,align='center',height=0.6,color=sns.color_palette("viridis_r",len(features)));ax.set_yticks(y_pos);ax.set_yticklabels(features,fontsize=6);ax.invert_yaxis();ax.set_xlabel("Importance / Coefficient Value",fontsize=7);ax.set_ylabel("Feature",fontsize=7);ax.tick_params(axis='x',labelsize=6)
                else: ax.text(0.5,0.5,"Feature importances or coefficients\nnot available for this model type.",horizontalalignment='center',verticalalignment='center',fontsize=8,color='grey',wrap=True)
                if fig: fig.tight_layout(pad=0.5)
            elif self.plot_type == "actual_vs_predicted":
                title_prefix = "Actual vs. Predicted Values"; y_true_test,y_pred_test=self.data_dict.get("y_test"),self.data_dict.get("y_pred_test");fig,ax=VisualUtils._create_base_figure(title_prefix+" (Test Set)",figsize=(5.5,4.6))
                if y_true_test is None or y_pred_test is None or len(y_true_test)==0 or (isinstance(y_pred_test, (pd.Series, np.ndarray)) and len(y_pred_test)==0) : ax.text(0.5,0.5,"Test data or predictions\nnot available.",ha='center',c='grey',fontsize=8,wrap=True)
                else:
                    sns.scatterplot(x=y_true_test,y=y_pred_test,ax=ax,alpha=0.45,s=18,edgecolor=None,color="#0056a3");min_val=(min(y_true_test.min(),pd.Series(y_pred_test).min()) if not y_true_test.empty and not pd.Series(y_pred_test).empty else 0);max_val=(max(y_true_test.max(),pd.Series(y_pred_test).max()) if not y_true_test.empty and not pd.Series(y_pred_test).empty else 1)
                    ax.plot([min_val,max_val],[min_val,max_val],'r--',lw=0.9,label="Ideal Fit Line");ax.set_xlabel("Actual Values",fontsize=7);ax.set_ylabel("Predicted Values",fontsize=7);ax.tick_params(labelsize=6)
                    if not y_true_test.empty: ax.legend(fontsize=6);ax.grid(True,linestyle=':',alpha=0.3)
                if fig: fig.tight_layout(pad=0.3)
            
            if fig: self.plot_generated.emit(self.plot_type,fig)
            else: self.plot_error.emit(self.plot_type,"Figure was not created (possibly due to previous error).")
        except Exception as e: logger.error(f"EvalPlot Error ({self.plot_type}): {e}",exc_info=True); self.plot_error.emit(self.plot_type,str(e)[:100])


class ModelTrainer(QWidget):
    def __init__(self, parent_window=None, data_viewer_instance=None):
        super().__init__(parent_window)
        self.setObjectName("ModelTrainerWidget")
        self.parent_window = parent_window; self.data_viewer_instance = data_viewer_instance
        self.current_model_pipeline: Pipeline | None = None; self.current_model_metrics: dict = {}
        self.X_train_data, self.y_train_data, self.X_test_data, self.y_test_data = None, None, None, None
        self.feature_importance_data: dict = {}; self.selected_features: list = []; self.selected_target: str | None = None
        self.label_encoders: dict = {}; self.batch_features_df: pd.DataFrame | None = None
        self.last_batch_prediction_results_df: pd.DataFrame | None = None
        self.training_thread: TrainingThread | None = None; self.evaluation_plot_threads: dict[str, EvaluationPlotThread] = {}
        self.model_category_tabs: QTabWidget | None = None
        self.category_comboboxes: dict[str, QComboBox] = {}
        self.current_category_selection_is_auto = False


        self.MODEL_MAP_TRAINING_THREAD = {
            "Linear Regression": train_linear_regression,
            "Ridge Regression": train_ridge_regression,
            "Lasso Regression": train_lasso_regression,
            "Elastic Net": train_elasticnet_regression,
            "Polynomial Regression": train_linear_regression, 
            "Huber Regression": train_huber_regression,
            "Quantile Regression": train_quantile_regression,
            "Decision Tree": train_decision_tree_regressor,
            "Random Forest": train_random_forest_regressor,
            "XGBoost": train_xgboost_regressor,
            "LightGBM": train_lightgbm_regressor,
            "KNN Regressor": train_knn_regressor,
            "SVR": train_svr,
            "Bayesian Ridge": train_bayesian_ridge_regression,
            "Gaussian Process Regressor": train_gaussian_process_regressor,
            "MLP Regressor": train_mlp_regressor,
            "Gradient Boosting Regressor": train_gradient_boosting_regressor,
            "Bagging Regressor": train_bagging_regressor,
            "PLSRegression": train_pls_regression,
            "Isotonic Regression": train_isotonic_regression,
        }

        self._initialize_directories()
        self._setup_ui()
        self._apply_stylesheet()
        self.refresh_saved_models()
        if parent_window and not ICON_APP.isNull(): parent_window.setWindowIcon(ICON_APP)

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        logger.debug(f"'{self.objectName()}' received showEvent. Scheduling _finalize_layouts.")
        QTimer.singleShot(0, self._finalize_layouts)

    def _finalize_layouts(self):
        logger.debug(f"'{self.objectName()}' executing _finalize_layouts (deferred).")
        if self.layout(): self.layout().activate()

        if hasattr(self, 'data_features_group') and self.data_features_group.layout():
            self.data_features_group.layout().activate()
            self.data_features_group.adjustSize()

        if hasattr(self, 'model_config_group') and self.model_config_group.layout():
            self.model_config_group.layout().activate()
            if hasattr(self, 'model_category_tabs'):
                for i in range(self.model_category_tabs.count()):
                    tab_widget = self.model_category_tabs.widget(i)
                    if tab_widget and tab_widget.layout():
                        tab_widget.layout().activate()
                    category_name = self.model_category_tabs.tabText(i)
                    if category_name in self.category_comboboxes:
                         self.category_comboboxes[category_name].adjustSize()
                self.model_category_tabs.adjustSize()
            self.model_config_group.adjustSize()
        
        if self.model_category_tabs and self.model_category_tabs.count() > 0:
            self._on_category_tab_changed(self.model_category_tabs.currentIndex())
        elif hasattr(self, 'hyperparams_stack'):
            self._update_hyperparams_ui("Auto Select (via CV)")
            self.hyperparams_stack.adjustSize()
            if hasattr(self, 'hyperparams_scroll_area'): self.hyperparams_scroll_area.updateGeometry()

        self.adjustSize()
        QApplication.processEvents()
        self.update()
        logger.debug(f"'{self.objectName()}' _finalize_layouts complete.")

    def _initialize_directories(self):
        try:
            app_data_dir=os.path.join(os.path.expanduser("~"),".MachineLearningStudioSaeedFYP"); os.makedirs(app_data_dir,exist_ok=True)
            self.saved_models_dir=os.path.join(app_data_dir,"saved_models"); self.model_reports_dir=os.path.join(app_data_dir,"model_reports")
            os.makedirs(self.saved_models_dir,exist_ok=True); os.makedirs(self.model_reports_dir,exist_ok=True); logger.info(f"User models dir: {self.saved_models_dir}")
        except OSError as e:
            logger.error(f"CRITICAL Error creating app data dirs: {e}",exc_info=True); self.saved_models_dir=os.path.join(os.getcwd(),"saved_models_mls_fallback"); self.model_reports_dir=os.path.join(os.getcwd(),"model_reports_mls_fallback")
            try: os.makedirs(self.saved_models_dir,exist_ok=True);os.makedirs(self.model_reports_dir,exist_ok=True); QMessageBox.warning(self,"Dir Warning",f"Could not create user-specific dirs: {e}.\nUsing fallback:\n{self.saved_models_dir}")
            except Exception as fallback_e: QMessageBox.critical(self,"Fatal Dir Error",f"Could not create any data dirs: {fallback_e}.\nSaving/exporting will fail.")

    def _apply_stylesheet(self): 
        icon_down_arrow_path = os.path.join(ICON_PATH_ROOT, ICON_DOWN_ARROW_STR).replace(os.sep, '/')
        icon_checked_path = os.path.join(ICON_PATH_ROOT, ICON_CHECKED_STR).replace(os.sep, '/')
        icon_unchecked_path = os.path.join(ICON_PATH_ROOT, ICON_UNCHECKED_STR).replace(os.sep, '/')
        self.setStyleSheet(f"""
            #ModelTrainerWidget {{ background-color: #f4f6f8; font-family: Arial, sans-serif; }}
            #ModelTrainerTabs::pane {{ border: 1px solid #dfe3e6; border-top: none; background: #ffffff; border-bottom-left-radius: 5px; border-bottom-right-radius: 5px; }}
            #ModelTrainerTabs QTabBar::tab {{
                background: #e9edf0; border: 1px solid #dfe3e6; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                padding: 4px 5px; margin-right: 0px;
                color: #33393e; font-weight: bold; font-size: 9px;
            }}
            #ModelTrainerTabs QTabBar::tab:selected {{ background: #0078d4; color: white; border-color: #006bc7; }}
            #ModelTrainerTabs QTabBar::tab:hover:!selected {{ background: #d8dde2; color: #1c2023; }}
            QGroupBox {{
                background-color: #ffffff; border: 1px solid #d1d9e0;
                border-radius: 5px; margin-top: 7px; padding: 8px;
                font-size: 12px; font-weight: bold; color: #102a43;
            }}
            QGroupBox#DataFeaturesGroup, QGroupBox#ModelConfigGroup, QGroupBox#TrainingActionGroup,
            QGroupBox#MetricsGroup, QGroupBox#PlotsGroup, QGroupBox#SinglePredGroup, QGroupBox#BatchPredGroup,
            QGroupBox#HistoryGroup, QGroupBox#SavedModelsGroup {{ padding-top: 20px; }}
            QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 1px 4px 3px 4px; color: #005a9e; }}
            QGroupBox#DataFeaturesGroup::title, QGroupBox#ModelConfigGroup::title, QGroupBox#TrainingActionGroup::title,
            QGroupBox#MetricsGroup::title, QGroupBox#PlotsGroup::title, QGroupBox#SinglePredGroup::title,
            QGroupBox#BatchPredGroup::title, QGroupBox#HistoryGroup::title, QGroupBox#SavedModelsGroup::title {{ background-color: #ffffff; }}
            QLabel[isHeader="true"] {{
                font-weight: bold;
                color: #005a9e;
                margin-top: 8px;
                margin-bottom: 3px;
                padding-left: 2px;
                font-size: 11px;
            }}
            QPushButton {{ background-color: #0078d4; color: white; font-size: 10.5px; font-weight: bold; padding: 6px 10px; border-radius: 3px; border: 1px solid #005a9e; min-height: 26px; }}
            QPushButton:hover {{ background-color: #005a9e; }} QPushButton:pressed {{ background-color: #004c87; }}
            QPushButton:disabled {{ background-color: #d8dcde; color: #707070; border-color: #c0c4c8; }}
            #SaveButton {{ background-color: #107c10; border-color: #0d650d; }} #SaveButton:hover {{ background-color: #0d650d; }}
            #DeleteButton {{ background-color: #d92c2c; border-color: #b82424; }} #DeleteButton:hover {{ background-color: #b82424; }}
            #SelectFeaturesButton {{ background-color: #5c2d91; border-color: #4a2474; }} #SelectFeaturesButton:hover {{ background-color: #4a2474; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background-color: #ffffff; border: 1px solid #b0b8bf; border-radius: 3px; padding: 4px 5px; font-size: 10.5px; min-height: 24px; color: #202020;}}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: #0078d4; }}
            QComboBox::down-arrow {{ image: url({icon_down_arrow_path}); width: 7px; height: 7px; right: 4px; }}
            QComboBox QAbstractItemView {{ font-size: 10px; border: 1px solid #c0c7cf; background-color: white; selection-background-color: #0078d4; selection-color: white; }}
            QCheckBox {{ font-size: 10.5px; color: #202020; padding: 1px 0; }}
            QCheckBox::indicator {{ width: 12px; height: 12px; margin-right: 3px;}}
            QCheckBox::indicator:checked {{ image: url({icon_checked_path}); }} QCheckBox::indicator:unchecked {{ image: url({icon_unchecked_path}); }}
            QListWidget, QTableWidget {{ background-color: #ffffff; border: 1px solid #d8dde2; border-radius: 3px; font-size: 10.5px; alternate-background-color: #f5f7f9; selection-background-color: #a9d0f5; selection-color: #101010; }}
            QHeaderView::section {{ background-color: #e9edf0; color: #18242f; padding: 5px; border: 1px solid #d8dde2; border-bottom-width: 2px; font-weight: bold; font-size: 10.5px;}}
            QProgressBar {{ border: 1px solid #b0b8bf; border-radius: 7px; background-color: #e8ecef; height: 17px; text-align: center; font-weight: bold; color: #102a43; font-size: 10px; }}
            QProgressBar::chunk {{ background-color: #28a745; border-radius: 6px; margin: 1px; }}
            QLabel#ProgressStatusLabel {{ font-style: italic; color: #005a9e; font-size: 10.5px; padding-top: 2px; }}
            QLabel#SelectedFeaturesLabel, QLabel#SelectedTargetLabel {{ font-size: 11px; color: #18242f; padding: 4px; border: 1px solid #c0c7cf; border-radius: 3px; background-color: #f0f4f8; min-height: 18px; }}
            QLabel#PredictionResultLabel {{ font-size: 13px; font-weight:bold; border: 1px solid #c0c7cf; border-radius: 4px; padding: 7px; qproperty-alignment: AlignCenter; min-height: 28px; }}
            QLabel#PredictionResultLabel[type="success"] {{ color: #155724; background-color: #d4edda; border-color: #c3e6cb; }}
            QLabel#PredictionResultLabel[type="error"] {{ color: #721c24; background-color: #f8d7da; border-color: #f5c6cb; }}
            QLabel#PredictionResultLabel[type="info"] {{ color: #0c5460; background-color: #d1ecf1; border-color: #bee5eb; }}
            QScrollArea {{ border: none; background-color: #ffffff; }}
            QStackedWidget {{ background-color: transparent; }}
        """)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("ModelTrainerTabs")
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        main_layout.addWidget(self.tabs)

        self._setup_training_tab()
        self._setup_evaluation_tab()
        self._setup_prediction_tab()
        self._setup_models_tab()

        self.tabs.setTabText(0, "Setup Training")
        self.tabs.setTabText(1, "Evaluation Results")
        self.tabs.setTabText(2, "Make Predictions")
        self.tabs.setTabText(3, "Manage & Compare")
        tab_icons = {0:ICON_TAB_TRAINING,1:ICON_TAB_EVALUATION,2:ICON_PREDICT_TAB,3:ICON_TAB_MODELS}
        for index, icon in tab_icons.items():
            if not icon.isNull(): self.tabs.setTabIcon(index, icon)
        
        if hasattr(self, 'mod_sel'):
            first_valid_model_index = 0
            for i in range(self.mod_sel.model().rowCount()):
                item = self.mod_sel.model().item(i)
                if item and item.isEnabled() and not item.data(Qt.UserRole + 1):
                    actual_model_name = item.data(Qt.UserRole) or item.text()
                    if actual_model_name != "Auto Select (via CV)":
                        first_valid_model_index = i
                        break
            
            if first_valid_model_index != -1 :
                self.mod_sel.setCurrentIndex(first_valid_model_index)
            elif self.mod_sel.count() > 0:
                self.mod_sel.setCurrentIndex(0)

            current_item = self.mod_sel.model().item(self.mod_sel.currentIndex())
            if current_item:
                initial_model_text = current_item.data(Qt.UserRole) or current_item.text()
                if not (not current_item.isEnabled() or current_item.data(Qt.UserRole + 1)):
                    self._update_hyperparams_ui(initial_model_text)
                else:
                    self._update_hyperparams_ui("Auto Select (via CV)")

        self._on_saved_model_selection_change()


    def _setup_training_tab(self):
        tab_page_widget = QWidget()
        tab_page_layout = QVBoxLayout(tab_page_widget)
        tab_page_layout.setContentsMargins(0,0,0,0)

        self.training_tab_scroll_area = QScrollArea()
        self.training_tab_scroll_area.setWidgetResizable(True)
        self.training_tab_scroll_area.setObjectName("TrainingTabMainScrollArea")
        self.training_tab_scroll_area.setStyleSheet("QScrollArea#TrainingTabMainScrollArea { border: none; }")
        tab_page_layout.addWidget(self.training_tab_scroll_area)

        scrollable_content_widget = QWidget()
        self.training_tab_scroll_area.setWidget(scrollable_content_widget)

        self.main_training_tab_layout = QVBoxLayout(scrollable_content_widget)
        self.main_training_tab_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        self.main_training_tab_layout.setSpacing(10)
        self.main_training_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.data_features_group = QGroupBox("1. Data Source & Feature Selection")
        self.data_features_group.setObjectName("DataFeaturesGroup")
        data_features_layout = QVBoxLayout(self.data_features_group)
        data_features_layout.setSpacing(8)
        self.select_features_button = QPushButton(" Select Features & Target Variable")
        self.select_features_button.setObjectName("SelectFeaturesButton")
        if not ICON_SELECT_FEATURES.isNull(): self.select_features_button.setIcon(ICON_SELECT_FEATURES)
        self.select_features_button.clicked.connect(self._open_feature_target_dialog)
        data_features_layout.addWidget(self.select_features_button)
        self.selected_features_label = QLabel("Features: Not selected")
        self.selected_features_label.setObjectName("SelectedFeaturesLabel"); self.selected_features_label.setWordWrap(True)
        data_features_layout.addWidget(self.selected_features_label)
        self.selected_target_label = QLabel("Target: Not selected")
        self.selected_target_label.setObjectName("SelectedTargetLabel")
        data_features_layout.addWidget(self.selected_target_label)
        fe_header_label = QLabel("Feature Engineering Options")
        fe_header_label.setProperty("isHeader", True)
        data_features_layout.addWidget(fe_header_label)
        self.feature_engineering_form_layout = QFormLayout()
        self.feature_engineering_form_layout.setSpacing(8)
        self.feature_engineering_form_layout.setLabelAlignment(Qt.AlignLeft)
        self.poly_features_checkbox = QCheckBox("Polynomial Features")
        self.poly_degree_spinbox = QSpinBox(); self.poly_degree_spinbox.setRange(2,4); self.poly_degree_spinbox.setValue(2)
        self.poly_degree_spinbox.setEnabled(False)
        self.poly_features_checkbox.toggled.connect(self.poly_degree_spinbox.setEnabled)
        self.feature_engineering_form_layout.addRow(self.poly_features_checkbox)
        self.feature_engineering_form_layout.addRow("Polynomial Degree:", self.poly_degree_spinbox)
        data_features_layout.addLayout(self.feature_engineering_form_layout)
        self.main_training_tab_layout.addWidget(self.data_features_group)

        self.model_config_group = QGroupBox("2. Model Configuration & Hyperparameters")
        self.model_config_group.setObjectName("ModelConfigGroup")
        model_config_layout = QVBoxLayout(self.model_config_group)
        model_config_layout.setSpacing(8)

        self.model_category_tabs = QTabWidget()
        self.model_category_tabs.setObjectName("ModelCategoryTabsInner")
        model_config_layout.addWidget(self.model_category_tabs)

        self.category_comboboxes = {}

        for category_name, models_in_category in MODEL_CATEGORIES.items():
            category_tab_page = QWidget()
            category_tab_layout = QVBoxLayout(category_tab_page)
            category_tab_layout.setContentsMargins(6, 6, 6, 6)
            category_tab_layout.setSpacing(6)

            if category_name == "Auto Select (via CV)":
                pass
            else:
                category_form_layout = QFormLayout()
                category_form_layout.setSpacing(8)
                category_form_layout.setLabelAlignment(Qt.AlignLeft)

                cat_combo = QComboBox()
                cat_combo.setObjectName(f"comboBox_{category_name.replace(' ', '')}")
                for model_name in models_in_category:
                    cat_combo.addItem(model_name, userData=model_name)
                
                cat_combo.currentIndexChanged.connect(self._on_category_model_selected)
                cat_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                cat_combo.setMinimumContentsLength(20)

                self.category_comboboxes[category_name] = cat_combo
                category_form_layout.addRow(f"{category_name}:", cat_combo)
                category_tab_layout.addLayout(category_form_layout)
                category_tab_layout.addStretch()

            self.model_category_tabs.addTab(category_tab_page, category_name)

        self.model_category_tabs.currentChanged.connect(self._on_category_tab_changed)

        hp_header_label = QLabel("Algorithm Hyperparameters")
        hp_header_label.setProperty("isHeader", True)
        model_config_layout.addWidget(hp_header_label)

        self.hyperparams_scroll_area = QScrollArea()
        self.hyperparams_scroll_area.setWidgetResizable(True)
        self.hyperparams_scroll_area.setMinimumHeight(150)
        self.hyperparams_scroll_area.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        self.hyperparams_stack = QStackedWidget()
        self.hyperparams_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.hyperparams_scroll_area.setWidget(self.hyperparams_stack)
        model_config_layout.addWidget(self.hyperparams_scroll_area)
        
        self.main_training_tab_layout.addWidget(self.model_config_group)

        self._init_hyperparameter_widgets()

        training_action_group = QGroupBox("3. Model Training"); training_action_group.setObjectName("TrainingActionGroup")
        training_action_layout = QVBoxLayout(training_action_group); training_action_layout.setSpacing(8)
        status_layout = QFormLayout(); status_layout.setSpacing(8); status_layout.setLabelAlignment(Qt.AlignLeft)
        self.progress_status_label = QLabel("Ready to train."); self.progress_status_label.setObjectName("ProgressStatusLabel")
        status_layout.addRow("Status:", self.progress_status_label); training_action_layout.addLayout(status_layout)
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(True); self.progress_bar.setValue(0)
        training_action_layout.addWidget(self.progress_bar)
        buttons_layout = QHBoxLayout(); buttons_layout.setSpacing(10); buttons_layout.addStretch()
        self.train_button = QPushButton(" Train Model"); self.train_button.setIcon(ICON_TRAIN if not ICON_TRAIN.isNull() else QIcon()); self.train_button.clicked.connect(self.start_training); buttons_layout.addWidget(self.train_button)
        self.save_model_button = QPushButton(" Save Trained Model"); self.save_model_button.setObjectName("SaveButton"); self.save_model_button.setIcon(ICON_SAVE if not ICON_SAVE.isNull() else QIcon()); self.save_model_button.setEnabled(False); self.save_model_button.clicked.connect(self.save_model); buttons_layout.addWidget(self.save_model_button)
        buttons_layout.addStretch(); training_action_layout.addLayout(buttons_layout)
        self.main_training_tab_layout.addWidget(training_action_group)

        self.main_training_tab_layout.addStretch(1)
        self.tabs.addTab(tab_page_widget, "")


    def _on_category_tab_changed(self, index: int):
            current_tab_text = self.model_category_tabs.tabText(index)
            logger.debug(f"Category tab changed to: {current_tab_text}")
            self.current_category_selection_is_auto = False

            if current_tab_text == "Auto Select (via CV)":
                self.current_category_selection_is_auto = True
                self._update_hyperparams_ui("Auto Select (via CV)")
            else:
                category_combo_box = self.category_comboboxes.get(current_tab_text)
                if category_combo_box and category_combo_box.count() > 0:
                    model_name = category_combo_box.currentData()
                    if model_name:
                        self._update_hyperparams_ui(model_name)
                    else:
                        self._update_hyperparams_ui(category_combo_box.currentText())
                elif category_combo_box and category_combo_box.count() == 0:
                    self._update_hyperparams_ui("Auto Select (via CV)")
                    logger.warning(f"Category tab '{current_tab_text}' selected, but its ComboBox has no models.")
                else:
                    logger.error(f"No ComboBox found for category tab: {current_tab_text}")
                    self._update_hyperparams_ui("Auto Select (via CV)")


    def _on_category_model_selected(self):
        sender_combo_box = self.sender()
        if not sender_combo_box or not isinstance(sender_combo_box, QComboBox):
            return

        model_name = sender_combo_box.currentData()
        if model_name:
            logger.debug(f"Model selected from category ComboBox: {model_name}")
            self.current_category_selection_is_auto = False
            self._update_hyperparams_ui(model_name)
        else:
            logger.warning(f"No UserData for selected item in ComboBox: {sender_combo_box.objectName()}, using currentText: {sender_combo_box.currentText()}")
            self._update_hyperparams_ui(sender_combo_box.currentText())


    def _on_mod_sel_changed(self, index):
        if index < 0: return
        item = self.mod_sel.model().item(index)
        if not item: return

        model_name = "Auto Select (via CV)" 
        if item.isEnabled() and not item.data(Qt.UserRole + 1): 
            model_name = item.data(Qt.UserRole) or item.text()
        elif item.data(Qt.UserRole + 1): 
            logger.debug(f"Category header '{item.text()}' focused. Showing Auto CV params.")
            model_name = "Auto Select (via CV)"
        
        self._update_hyperparams_ui(model_name)


    def _init_hyperparameter_widgets(self): 
        def _create_placeholder_params_widget(model_name_text: str) -> QWidget:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            label = QLabel(f"Hyperparameter UI for '{model_name_text}' not yet implemented.\nDefault parameters will be used if trained.")
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            return widget

        self.hyperparam_widgets_map = {
            "Linear Regression": self._create_lr_params_widget(),
            "Decision Tree": self._create_dt_params_widget(),
            "Random Forest": self._create_rf_params_widget(),
            "XGBoost": self._create_xgb_params_widget(),
            "SVR": self._create_svr_params_widget(),
            
            "Ridge Regression": self._create_ridge_params_widget(),
            "Lasso Regression": self._create_lasso_params_widget(),
            "Elastic Net": self._create_elasticnet_params_widget(),
            "Polynomial Regression": self._create_auto_select_params_widget(), 
            "Huber Regression": _create_placeholder_params_widget("Huber Regression"), 
            "Quantile Regression": _create_placeholder_params_widget("Quantile Regression"), 
            "LightGBM": self._create_lightgbm_params_widget(),
            "KNN Regressor": self._create_knn_params_widget(),
            "Bayesian Ridge": self._create_bayesian_ridge_params_widget(),
            "Gaussian Process Regressor": _create_placeholder_params_widget("Gaussian Process Regressor"), 
            "MLP Regressor": _create_placeholder_params_widget("MLP Regressor"), 
            "Gradient Boosting Regressor": _create_placeholder_params_widget("Gradient Boosting Regressor"), 
            "Bagging Regressor": _create_placeholder_params_widget("Bagging Regressor"), 
            "PLSRegression": _create_placeholder_params_widget("PLSRegression"), 
            "Isotonic Regression": _create_placeholder_params_widget("Isotonic Regression (often no params or data-driven)"),
            "Auto Select (via CV)": self._create_auto_select_params_widget(),
        }
        for algorithm_name, form_widget in self.hyperparam_widgets_map.items():
            self.hyperparams_stack.addWidget(form_widget)
            form_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)

    def _create_lr_params_widget(self) -> QWidget: 
        widget=QWidget();layout=QFormLayout(widget);layout.setSpacing(6);layout.setContentsMargins(2,2,2,2);layout.setLabelAlignment(Qt.AlignLeft);self.lr_fit_intercept_checkbox=QCheckBox("Fit Intercept");self.lr_fit_intercept_checkbox.setChecked(True);layout.addRow(self.lr_fit_intercept_checkbox); return widget
    def _create_dt_params_widget(self) -> QWidget: 
        widget=QWidget();layout=QFormLayout(widget);layout.setSpacing(6);layout.setContentsMargins(2,2,2,2);layout.setLabelAlignment(Qt.AlignLeft);self.dt_max_depth_spinbox=QSpinBox();self.dt_max_depth_spinbox.setRange(1,50);self.dt_max_depth_spinbox.setValue(5);self.dt_max_depth_spinbox.setToolTip("0 for unlimited.");self.dt_min_samples_split_spinbox=QSpinBox();self.dt_min_samples_split_spinbox.setRange(2,100);self.dt_min_samples_split_spinbox.setValue(2);layout.addRow("Max Depth:",self.dt_max_depth_spinbox);layout.addRow("Min Samples Split:",self.dt_min_samples_split_spinbox); return widget
    def _create_rf_params_widget(self) -> QWidget: 
        widget=QWidget();layout=QFormLayout(widget);layout.setSpacing(6);layout.setContentsMargins(2,2,2,2);layout.setLabelAlignment(Qt.AlignLeft);self.rf_n_estimators_spinbox=QSpinBox();self.rf_n_estimators_spinbox.setRange(10,500);self.rf_n_estimators_spinbox.setValue(100);self.rf_n_estimators_spinbox.setSingleStep(10);self.rf_max_depth_spinbox=QSpinBox();self.rf_max_depth_spinbox.setRange(1,50);self.rf_max_depth_spinbox.setValue(10);self.rf_max_depth_spinbox.setToolTip("0 for unlimited.");layout.addRow("Number of Estimators:",self.rf_n_estimators_spinbox);layout.addRow("Max Depth:",self.rf_max_depth_spinbox); return widget
    def _create_xgb_params_widget(self) -> QWidget: 
        widget=QWidget();layout=QFormLayout(widget);layout.setSpacing(6);layout.setContentsMargins(2,2,2,2);layout.setLabelAlignment(Qt.AlignLeft);self.xgb_n_estimators_spinbox=QSpinBox();self.xgb_n_estimators_spinbox.setRange(10,500);self.xgb_n_estimators_spinbox.setValue(100);self.xgb_n_estimators_spinbox.setSingleStep(10);self.xgb_learning_rate_spinbox=QDoubleSpinBox();self.xgb_learning_rate_spinbox.setRange(0.001,1.0);self.xgb_learning_rate_spinbox.setValue(0.1);self.xgb_learning_rate_spinbox.setSingleStep(0.01);self.xgb_learning_rate_spinbox.setDecimals(3);layout.addRow("Number of Estimators:",self.xgb_n_estimators_spinbox);layout.addRow("Learning Rate:",self.xgb_learning_rate_spinbox); return widget
    def _create_svr_params_widget(self) -> QWidget: 
        widget=QWidget();layout=QFormLayout(widget);layout.setSpacing(6);layout.setContentsMargins(2,2,2,2);layout.setLabelAlignment(Qt.AlignLeft);self.svr_kernel_combobox=QComboBox();self.svr_kernel_combobox.addItems(["rbf","linear","poly","sigmoid"]);self.svr_C_spinbox=QDoubleSpinBox();self.svr_C_spinbox.setRange(0.01,100.0);self.svr_C_spinbox.setValue(1.0);self.svr_C_spinbox.setSingleStep(0.1);self.svr_epsilon_spinbox=QDoubleSpinBox();self.svr_epsilon_spinbox.setRange(0.01,1.0);self.svr_epsilon_spinbox.setValue(0.1);self.svr_epsilon_spinbox.setSingleStep(0.01);layout.addRow("Kernel:",self.svr_kernel_combobox);layout.addRow("C (Regularization):",self.svr_C_spinbox);layout.addRow("Epsilon:",self.svr_epsilon_spinbox); return widget
    def _create_ridge_params_widget(self) -> QWidget: 
        widget = QWidget(); layout = QFormLayout(widget); layout.setSpacing(6); layout.setContentsMargins(2,2,2,2); layout.setLabelAlignment(Qt.AlignLeft)
        self.ridge_alpha_spinbox = QDoubleSpinBox(); self.ridge_alpha_spinbox.setDecimals(4); self.ridge_alpha_spinbox.setRange(0.0001, 1000.0); self.ridge_alpha_spinbox.setValue(1.0); self.ridge_alpha_spinbox.setSingleStep(0.1)
        layout.addRow("Alpha (L2 Strength):", self.ridge_alpha_spinbox)
        self.ridge_fit_intercept_checkbox = QCheckBox("Fit Intercept"); self.ridge_fit_intercept_checkbox.setChecked(True)
        layout.addRow(self.ridge_fit_intercept_checkbox)
        return widget
    def _create_lasso_params_widget(self) -> QWidget: 
        widget = QWidget(); layout = QFormLayout(widget); layout.setSpacing(6); layout.setContentsMargins(2,2,2,2); layout.setLabelAlignment(Qt.AlignLeft)
        self.lasso_alpha_spinbox = QDoubleSpinBox(); self.lasso_alpha_spinbox.setDecimals(4); self.lasso_alpha_spinbox.setRange(0.0001, 1000.0); self.lasso_alpha_spinbox.setValue(1.0); self.lasso_alpha_spinbox.setSingleStep(0.1)
        layout.addRow("Alpha (L1 Strength):", self.lasso_alpha_spinbox)
        self.lasso_fit_intercept_checkbox = QCheckBox("Fit Intercept"); self.lasso_fit_intercept_checkbox.setChecked(True)
        layout.addRow(self.lasso_fit_intercept_checkbox)
        return widget
    def _create_elasticnet_params_widget(self) -> QWidget: 
        widget = QWidget(); layout = QFormLayout(widget); layout.setSpacing(6); layout.setContentsMargins(2,2,2,2); layout.setLabelAlignment(Qt.AlignLeft)
        self.elasticnet_alpha_spinbox = QDoubleSpinBox(); self.elasticnet_alpha_spinbox.setDecimals(4); self.elasticnet_alpha_spinbox.setRange(0.0001, 1000.0); self.elasticnet_alpha_spinbox.setValue(1.0); self.elasticnet_alpha_spinbox.setSingleStep(0.1)
        layout.addRow("Alpha (Strength):", self.elasticnet_alpha_spinbox)
        self.elasticnet_l1_ratio_spinbox = QDoubleSpinBox(); self.elasticnet_l1_ratio_spinbox.setDecimals(2); self.elasticnet_l1_ratio_spinbox.setRange(0.0, 1.0); self.elasticnet_l1_ratio_spinbox.setValue(0.5); self.elasticnet_l1_ratio_spinbox.setSingleStep(0.01)
        layout.addRow("L1 Ratio (0=L2, 1=L1):", self.elasticnet_l1_ratio_spinbox)
        self.elasticnet_fit_intercept_checkbox = QCheckBox("Fit Intercept"); self.elasticnet_fit_intercept_checkbox.setChecked(True)
        layout.addRow(self.elasticnet_fit_intercept_checkbox)
        return widget
    def _create_bayesian_ridge_params_widget(self) -> QWidget: 
        widget = QWidget(); layout = QFormLayout(widget); layout.setSpacing(6); layout.setContentsMargins(2,2,2,2); layout.setLabelAlignment(Qt.AlignLeft)
        self.br_alpha_1_spinbox = QDoubleSpinBox(); self.br_alpha_1_spinbox.setDecimals(7); self.br_alpha_1_spinbox.setRange(1e-7, 1.0); self.br_alpha_1_spinbox.setValue(1e-6); self.br_alpha_1_spinbox.setSingleStep(1e-7)
        layout.addRow("Alpha 1 (Gamma prior shape):", self.br_alpha_1_spinbox)
        self.br_alpha_2_spinbox = QDoubleSpinBox(); self.br_alpha_2_spinbox.setDecimals(7); self.br_alpha_2_spinbox.setRange(1e-7, 1.0); self.br_alpha_2_spinbox.setValue(1e-6); self.br_alpha_2_spinbox.setSingleStep(1e-7)
        layout.addRow("Alpha 2 (Gamma prior rate):", self.br_alpha_2_spinbox)
        self.br_lambda_1_spinbox = QDoubleSpinBox(); self.br_lambda_1_spinbox.setDecimals(7); self.br_lambda_1_spinbox.setRange(1e-7, 1.0); self.br_lambda_1_spinbox.setValue(1e-6); self.br_lambda_1_spinbox.setSingleStep(1e-7)
        layout.addRow("Lambda 1 (Gamma prior shape):", self.br_lambda_1_spinbox)
        self.br_lambda_2_spinbox = QDoubleSpinBox(); self.br_lambda_2_spinbox.setDecimals(7); self.br_lambda_2_spinbox.setRange(1e-7, 1.0); self.br_lambda_2_spinbox.setValue(1e-6); self.br_lambda_2_spinbox.setSingleStep(1e-7)
        layout.addRow("Lambda 2 (Gamma prior rate):", self.br_lambda_2_spinbox)
        self.br_fit_intercept_checkbox = QCheckBox("Fit Intercept"); self.br_fit_intercept_checkbox.setChecked(True)
        layout.addRow(self.br_fit_intercept_checkbox)
        return widget
    def _create_lightgbm_params_widget(self) -> QWidget: 
        widget = QWidget(); layout = QFormLayout(widget); layout.setSpacing(6); layout.setContentsMargins(2,2,2,2); layout.setLabelAlignment(Qt.AlignLeft)
        self.lgbm_n_estimators_spinbox = QSpinBox(); self.lgbm_n_estimators_spinbox.setRange(10, 2000); self.lgbm_n_estimators_spinbox.setValue(100); self.lgbm_n_estimators_spinbox.setSingleStep(10)
        layout.addRow("Number of Estimators:", self.lgbm_n_estimators_spinbox)
        self.lgbm_learning_rate_spinbox = QDoubleSpinBox(); self.lgbm_learning_rate_spinbox.setDecimals(3); self.lgbm_learning_rate_spinbox.setRange(0.001, 1.0); self.lgbm_learning_rate_spinbox.setValue(0.1); self.lgbm_learning_rate_spinbox.setSingleStep(0.01)
        layout.addRow("Learning Rate:", self.lgbm_learning_rate_spinbox)
        self.lgbm_num_leaves_spinbox = QSpinBox(); self.lgbm_num_leaves_spinbox.setRange(2, 256); self.lgbm_num_leaves_spinbox.setValue(31)
        layout.addRow("Number of Leaves:", self.lgbm_num_leaves_spinbox)
        self.lgbm_max_depth_spinbox = QSpinBox(); self.lgbm_max_depth_spinbox.setRange(-1, 100); self.lgbm_max_depth_spinbox.setValue(-1); self.lgbm_max_depth_spinbox.setToolTip("-1 for no limit.")
        layout.addRow("Max Depth:", self.lgbm_max_depth_spinbox)
        return widget
    def _create_knn_params_widget(self) -> QWidget: 
        widget = QWidget(); layout = QFormLayout(widget); layout.setSpacing(6); layout.setContentsMargins(2,2,2,2); layout.setLabelAlignment(Qt.AlignLeft)
        self.knn_n_neighbors_spinbox = QSpinBox(); self.knn_n_neighbors_spinbox.setRange(1, 100); self.knn_n_neighbors_spinbox.setValue(5)
        layout.addRow("Number of Neighbors (k):", self.knn_n_neighbors_spinbox)
        self.knn_weights_combobox = QComboBox(); self.knn_weights_combobox.addItems(["uniform", "distance"])
        layout.addRow("Weights:", self.knn_weights_combobox)
        self.knn_algorithm_combobox = QComboBox(); self.knn_algorithm_combobox.addItems(["auto", "ball_tree", "kd_tree", "brute"])
        layout.addRow("Algorithm:", self.knn_algorithm_combobox)
        self.knn_p_spinbox = QSpinBox(); self.knn_p_spinbox.setRange(1, 5); self.knn_p_spinbox.setValue(2); self.knn_p_spinbox.setToolTip("1 for Manhattan, 2 for Euclidean distance")
        layout.addRow("Minkowski Power (p):", self.knn_p_spinbox)
        return widget
    def _create_auto_select_params_widget(self) -> QWidget: 
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel("Hyperparameters will be determined by Auto CV during training.\n"
                       "Cross-validation will be performed on a selection of models\n"
                       "to choose the best performing one with its default settings.\n\n"
                       "Polynomial features choice (above) will be respected if checked.")
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        return widget

    def _update_hyperparams_ui(self, model_type: str):
        logger.debug(f"Updating hyperparams UI for: {model_type} (using QStackedWidget).")
        target_widget = self.hyperparam_widgets_map.get(model_type)
        
        if not target_widget:
            if model_type == "Auto Select (via CV)" or model_type.startswith("--- ") or not model_type:
                target_widget = self.hyperparam_widgets_map.get("Auto Select (via CV)")
            else:
                logger.error(f"Model type '{model_type}' has no defined UI. Defaulting to 'Auto Select' UI.")
                target_widget = self.hyperparam_widgets_map.get("Auto Select (via CV)", QWidget())

        if hasattr(self, 'hyperparams_stack') and target_widget:
            if self.hyperparams_stack.currentWidget() != target_widget:
                self.hyperparams_stack.setCurrentWidget(target_widget)
            if target_widget.layout(): target_widget.layout().activate()
            target_widget.adjustSize()
            self.hyperparams_stack.adjustSize()
        elif hasattr(self, 'hyperparams_stack'):
            logger.warning(f"Target_widget could not be determined for model_type: '{model_type}' in hyperparams_stack.")
        
        if model_type == "Polynomial Regression":
            if hasattr(self, 'poly_features_checkbox'): self.poly_features_checkbox.setChecked(True)
            if hasattr(self, 'poly_degree_spinbox'): self.poly_degree_spinbox.setEnabled(True)

        if hasattr(self, 'hyperparams_scroll_area'):
            self.hyperparams_scroll_area.updateGeometry()
        if hasattr(self, 'model_config_group') and self.model_config_group.layout():
            self.model_config_group.layout().activate()
            self.model_config_group.adjustSize()
        QApplication.processEvents()
        logger.debug(f"Hyperparams UI (QStackedWidget) update for {model_type} complete.")


    def _get_current_hyperparams(self) -> dict:
        params = {
            'polynomial_features': self.poly_features_checkbox.isChecked(),
            'poly_degree': self.poly_degree_spinbox.value() if self.poly_features_checkbox.isChecked() else None
        }
        
        model_type_to_get_params_for = ""

        if self.current_category_selection_is_auto or \
           (self.model_category_tabs and self.model_category_tabs.tabText(self.model_category_tabs.currentIndex()) == "Auto Select (via CV)"):
            params['auto_select_model'] = True
            return params
        else:
            if self.model_category_tabs:
                current_category_tab_text = self.model_category_tabs.tabText(self.model_category_tabs.currentIndex())
                active_category_combo = self.category_comboboxes.get(current_category_tab_text)
                if active_category_combo:
                    model_type_to_get_params_for = active_category_combo.currentData()
                    if not model_type_to_get_params_for:
                        model_type_to_get_params_for = active_category_combo.currentText()
                else:
                    logger.warning(f"Could not find ComboBox for active category tab: {current_category_tab_text}. Defaulting to Auto Select params logic.")
                    params['auto_select_model'] = True
                    return params
            else:
                logger.error("_get_current_hyperparams: model_category_tabs not found. Defaulting.")
                params['auto_select_model'] = True
                return params

        if not model_type_to_get_params_for:
            logger.warning("_get_current_hyperparams: model_type_to_get_params_for is empty. Defaulting to Auto Select logic.")
            params['auto_select_model'] = True
            return params

        if model_type_to_get_params_for == "Polynomial Regression":
            params['polynomial_features'] = True 
            params['poly_degree'] = self.poly_degree_spinbox.value() if self.poly_degree_spinbox.isEnabled() else 2
        elif model_type_to_get_params_for == "Linear Regression": params.update({'fit_intercept': self.lr_fit_intercept_checkbox.isChecked()})
        elif model_type_to_get_params_for == "Ridge Regression": params.update({'alpha': self.ridge_alpha_spinbox.value(), 'fit_intercept': self.ridge_fit_intercept_checkbox.isChecked()})
        elif model_type_to_get_params_for == "Lasso Regression": params.update({'alpha': self.lasso_alpha_spinbox.value(), 'fit_intercept': self.lasso_fit_intercept_checkbox.isChecked()})
        elif model_type_to_get_params_for == "Elastic Net": params.update({'alpha': self.elasticnet_alpha_spinbox.value(), 'l1_ratio': self.elasticnet_l1_ratio_spinbox.value(), 'fit_intercept': self.elasticnet_fit_intercept_checkbox.isChecked()})
        elif model_type_to_get_params_for == "Decision Tree": params.update({'max_depth': self.dt_max_depth_spinbox.value() if self.dt_max_depth_spinbox.value() > 0 else None, 'min_samples_split': self.dt_min_samples_split_spinbox.value()})
        elif model_type_to_get_params_for == "Random Forest": params.update({'n_estimators': self.rf_n_estimators_spinbox.value(),'max_depth': self.rf_max_depth_spinbox.value() if self.rf_max_depth_spinbox.value() > 0 else None})
        elif model_type_to_get_params_for == "XGBoost": params.update({'n_estimators': self.xgb_n_estimators_spinbox.value(),'learning_rate': self.xgb_learning_rate_spinbox.value()})
        elif model_type_to_get_params_for == "LightGBM": params.update({'n_estimators': self.lgbm_n_estimators_spinbox.value(), 'learning_rate': self.lgbm_learning_rate_spinbox.value(), 'num_leaves': self.lgbm_num_leaves_spinbox.value(), 'max_depth': self.lgbm_max_depth_spinbox.value() if self.lgbm_max_depth_spinbox.value() != -1 else None })
        elif model_type_to_get_params_for == "KNN Regressor": params.update({'n_neighbors': self.knn_n_neighbors_spinbox.value(), 'weights': self.knn_weights_combobox.currentText(), 'algorithm': self.knn_algorithm_combobox.currentText(), 'p': self.knn_p_spinbox.value()})
        elif model_type_to_get_params_for == "SVR": params.update({'kernel': self.svr_kernel_combobox.currentText(),'C': self.svr_C_spinbox.value(),'epsilon': self.svr_epsilon_spinbox.value()})
        elif model_type_to_get_params_for == "Bayesian Ridge": params.update({'alpha_1': self.br_alpha_1_spinbox.value(), 'alpha_2': self.br_alpha_2_spinbox.value(), 'lambda_1': self.br_lambda_1_spinbox.value(), 'lambda_2': self.br_lambda_2_spinbox.value(), 'fit_intercept': self.br_fit_intercept_checkbox.isChecked()})
        else:
            logger.info(f"Getting params for '{model_type_to_get_params_for}' which may have a placeholder UI. Only general FE params included explicitly.")
        return params
    
    def _get_current_hyperparams_for_model(self, model_type_for_defaults: str) -> dict:
        logger.debug(f"Getting default UI hyperparams for (auto-selected) model: {model_type_for_defaults}")
        original_tab_index = self.model_category_tabs.currentIndex()
        original_combo_indices = {cat: combo.currentIndex() for cat, combo in self.category_comboboxes.items()}

        target_category_tab_name = None
        target_model_index_in_combo = -1
        target_combo_box_to_set = None

        for category_name, combo_box in self.category_comboboxes.items():
            for i in range(combo_box.count()):
                if combo_box.itemData(i) == model_type_for_defaults:
                    target_category_tab_name = category_name
                    target_model_index_in_combo = i
                    target_combo_box_to_set = combo_box
                    break
            if target_category_tab_name:
                break
        
        params = {}
        if target_category_tab_name and target_combo_box_to_set is not None:
            self.model_category_tabs.blockSignals(True)
            target_combo_box_to_set.blockSignals(True)

            for i in range(self.model_category_tabs.count()):
                if self.model_category_tabs.tabText(i) == target_category_tab_name:
                    self.model_category_tabs.setCurrentIndex(i)
                    break
            
            target_combo_box_to_set.setCurrentIndex(target_model_index_in_combo)
            
            self._update_hyperparams_ui(model_type_for_defaults) 
            QApplication.processEvents()

            params = self._get_current_hyperparams()
            params.pop('auto_select_model', None)

            target_combo_box_to_set.setCurrentIndex(original_combo_indices.get(target_category_tab_name, 0))
            self.model_category_tabs.setCurrentIndex(original_tab_index)
            
            target_combo_box_to_set.blockSignals(False)
            self.model_category_tabs.blockSignals(False)

            self._on_category_tab_changed(original_tab_index)

        else:
            logger.warning(f"Could not find UI form for auto-selected model: {model_type_for_defaults} to get its default params. Using only poly features if set.")
            params = {
                'polynomial_features': self.poly_features_checkbox.isChecked(),
                'poly_degree': self.poly_degree_spinbox.value() if self.poly_features_checkbox.isChecked() else None
            }
        return params

    def _update_ui_after_load(self, metadata: dict):
        loaded_model_type = metadata.get('metrics', {}).get('model_type')
        if not loaded_model_type:
            logger.error("Loaded metadata is missing model_type. Cannot update UI.")
            self._show_message("Load Error", "Model metadata is incomplete (missing model type).", "critical")
            return

        logger.info(f"Updating UI after loading model: {loaded_model_type}")

        found_tab_idx = -1
        found_combo_idx = -1
        target_category_name = None

        if loaded_model_type == "Auto Select (via CV)":
            for i in range(self.model_category_tabs.count()):
                if self.model_category_tabs.tabText(i) == "Auto Select (via CV)":
                    found_tab_idx = i
                    break
        else:
            for cat_idx in range(self.model_category_tabs.count()):
                category_name = self.model_category_tabs.tabText(cat_idx)
                if category_name in self.category_comboboxes:
                    combo = self.category_comboboxes[category_name]
                    for item_idx in range(combo.count()):
                        if combo.itemData(item_idx) == loaded_model_type:
                            found_tab_idx = cat_idx
                            found_combo_idx = item_idx
                            target_category_name = category_name
                            break
                if found_tab_idx != -1:
                    break
        
        if found_tab_idx != -1:
            self.model_category_tabs.blockSignals(True)
            self.model_category_tabs.setCurrentIndex(found_tab_idx)
            self.current_category_selection_is_auto = (self.model_category_tabs.tabText(found_tab_idx) == "Auto Select (via CV)")
            self.model_category_tabs.blockSignals(False)

            if target_category_name and found_combo_idx != -1:
                combo_to_set = self.category_comboboxes.get(target_category_name)
                if combo_to_set:
                    combo_to_set.blockSignals(True)
                    combo_to_set.setCurrentIndex(found_combo_idx)
                    combo_to_set.blockSignals(False)
            self._update_hyperparams_ui(loaded_model_type)
        else:
            logger.warning(f"Loaded model type '{loaded_model_type}' not found in any category tab. Defaulting UI.")
            self.model_category_tabs.setCurrentIndex(0)
            self._update_hyperparams_ui("Auto Select (via CV)")


        hp_conf = metadata.get('hyperparams_config', {})
        is_poly_reg_loaded = (loaded_model_type == "Polynomial Regression")
        self.poly_features_checkbox.setChecked(hp_conf.get('polynomial_features', is_poly_reg_loaded))
        if self.poly_features_checkbox.isChecked():
            self.poly_degree_spinbox.setValue(hp_conf.get('poly_degree', 2))
            self.poly_degree_spinbox.setEnabled(True)
        else:
            self.poly_degree_spinbox.setEnabled(False)

        if loaded_model_type == "Linear Regression": self.lr_fit_intercept_checkbox.setChecked(hp_conf.get('fit_intercept', True))
        elif loaded_model_type == "Ridge Regression": self.ridge_alpha_spinbox.setValue(hp_conf.get('alpha', 1.0)); self.ridge_fit_intercept_checkbox.setChecked(hp_conf.get('fit_intercept', True))
        elif loaded_model_type == "Bayesian Ridge": self.br_alpha_1_spinbox.setValue(hp_conf.get('alpha_1', 1e-6)); self.br_alpha_2_spinbox.setValue(hp_conf.get('alpha_2', 1e-6)); self.br_lambda_1_spinbox.setValue(hp_conf.get('lambda_1', 1e-6)); self.br_lambda_2_spinbox.setValue(hp_conf.get('lambda_2', 1e-6)); self.br_fit_intercept_checkbox.setChecked(hp_conf.get('fit_intercept', True))


        sf_text = (', '.join(map(str, self.selected_features[:3])) + ('...' if len(self.selected_features) > 3 else '')) if self.selected_features else "N/A"
        self.selected_features_label.setText(f"<b>Features ({len(self.selected_features)}) (Loaded):</b> {sf_text}")
        self.selected_target_label.setText(f"<b>Target (Loaded):</b> {self.selected_target or 'N/A'}")
        self._update_evaluation_metrics_table(self.current_model_metrics)
        self._clear_evaluation_plots_display("Model loaded. Plots not regenerated until re-training or specific action.")
        self._update_prediction_inputs_ui(self.selected_features)
        self.prediction_result_label.setText("Model loaded. Ready for predictions.")
        self.prediction_result_label.setProperty("type", "info")
        if self.style(): self.prediction_result_label.style().polish(self.prediction_result_label)

    def _setup_evaluation_tab(self): 
        tab_widget=QWidget();layout=QVBoxLayout(tab_widget);layout.setSpacing(10);layout.setContentsMargins(8,8,8,8)
        metrics_group=QGroupBox("Model Performance Metrics");metrics_group.setObjectName("MetricsGroup");metrics_layout=QVBoxLayout(metrics_group);metrics_layout.setContentsMargins(10,5,10,10);self.metrics_table=QTableWidget();self.metrics_table.setColumnCount(6);self.metrics_table.setHorizontalHeaderLabels(["Metric","Train Value","Test Value","CV Mean","CV Std Dev","Test-Train Diff."]);self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.metrics_table.setAlternatingRowColors(True);self.metrics_table.setMinimumHeight(120);metrics_layout.addWidget(self.metrics_table);layout.addWidget(metrics_group)
        plots_group=QGroupBox("Evaluation Visualizations");plots_group.setObjectName("PlotsGroup");plots_layout=QHBoxLayout(plots_group);plots_layout.setContentsMargins(10,5,10,10);plots_layout.setSpacing(10)
        fi_plot_group=QGroupBox("Feature Importances / Coefficients");fi_plot_layout=QVBoxLayout(fi_plot_group);fi_plot_layout.setContentsMargins(5,5,5,5);self.feature_importance_canvas=FigureCanvas(Figure(figsize=(6.0,4.5)));self.feature_importance_canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding);fi_plot_layout.addWidget(self.feature_importance_canvas);plots_layout.addWidget(fi_plot_group,1)
        ap_plot_group=QGroupBox("Actual vs. Predicted Values");ap_plot_layout=QVBoxLayout(ap_plot_group);ap_plot_layout.setContentsMargins(5,5,5,5);self.actual_predicted_canvas=FigureCanvas(Figure(figsize=(5.5,4.5)));self.actual_predicted_canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding);ap_plot_layout.addWidget(self.actual_predicted_canvas);plots_layout.addWidget(ap_plot_group,1)
        layout.addWidget(plots_group);layout.addStretch(1);self.tabs.addTab(tab_widget,"");self._clear_evaluation_plots_display()

    def _clear_evaluation_plots_display(self, message="Train or load a model to view evaluation results."): 
        for cvs,txt in [(self.feature_importance_canvas,"Train model for Feature Importances."),(self.actual_predicted_canvas,message)]:
            if cvs:cvs.figure.clear();ax=cvs.figure.add_subplot(111);ax.text(0.5,0.5,txt,ha='center',va='center',fontsize=9,color='grey',wrap=True);ax.set_axis_off();cvs.draw_idle()

    def _setup_prediction_tab(self): 
        tab_widget=QWidget();layout=QVBoxLayout(tab_widget);layout.setSpacing(10);layout.setContentsMargins(8,8,8,8)
        single_pred_group=QGroupBox("Single Instance Prediction");single_pred_group.setObjectName("SinglePredGroup");single_pred_layout=QVBoxLayout(single_pred_group);single_pred_layout.setContentsMargins(10,5,10,10);single_pred_layout.setSpacing(8)
        self.prediction_inputs_scroll=QScrollArea();self.prediction_inputs_scroll.setWidgetResizable(True);self.prediction_inputs_scroll.setMinimumHeight(120);self.prediction_inputs_scroll.setMaximumHeight(200)
        pred_inputs_container=QWidget();self.prediction_form_layout=QFormLayout(pred_inputs_container);self.prediction_form_layout.setSpacing(8);self.prediction_form_layout.setLabelAlignment(Qt.AlignLeft);self.prediction_inputs_scroll.setWidget(pred_inputs_container);single_pred_layout.addWidget(self.prediction_inputs_scroll)
        self.prediction_result_label=QLabel("Prediction will appear here.");self.prediction_result_label.setObjectName("PredictionResultLabel");self.prediction_result_label.setProperty("type","info");single_pred_layout.addWidget(self.prediction_result_label,0,Qt.AlignCenter)
        pred_btn_layout=QHBoxLayout();pred_btn_layout.addStretch();self.predict_button=QPushButton(" Make Prediction");self.predict_button.setIcon(ICON_PREDICT_BTN if not ICON_PREDICT_BTN.isNull() else QIcon());self.predict_button.clicked.connect(self.make_prediction);pred_btn_layout.addWidget(self.predict_button);pred_btn_layout.addStretch();single_pred_layout.addLayout(pred_btn_layout);layout.addWidget(single_pred_group)
        batch_pred_group=QGroupBox("Batch Prediction from File");batch_pred_group.setObjectName("BatchPredGroup");batch_pred_layout=QVBoxLayout(batch_pred_group);batch_pred_layout.setContentsMargins(10,5,10,10);batch_pred_layout.setSpacing(8)
        load_batch_layout=QHBoxLayout();load_batch_layout.setSpacing(8);self.load_batch_features_button=QPushButton(" Load Features File (.csv, .xlsx)");self.load_batch_features_button.setIcon(ICON_BATCH_PREDICT_FILE if not ICON_BATCH_PREDICT_FILE.isNull() else QIcon());self.load_batch_features_button.clicked.connect(self._load_batch_features_file);load_batch_layout.addWidget(self.load_batch_features_button)
        self.batch_file_label=QLabel("No file loaded for batch prediction.");self.batch_file_label.setStyleSheet("font-style:italic;color:grey;font-size:10px;padding-left:5px;");load_batch_layout.addWidget(self.batch_file_label,1);batch_pred_layout.addLayout(load_batch_layout)
        batch_actions_layout=QHBoxLayout();batch_actions_layout.setSpacing(8);batch_actions_layout.addStretch()
        self.run_batch_predict_button=QPushButton(" Run Batch Prediction");self.run_batch_predict_button.setIcon(ICON_BATCH_PREDICT_RUN if not ICON_BATCH_PREDICT_RUN.isNull() else QIcon());self.run_batch_predict_button.setEnabled(False);self.run_batch_predict_button.clicked.connect(self._run_batch_prediction);batch_actions_layout.addWidget(self.run_batch_predict_button)
        self.save_batch_results_button=QPushButton(" Save Batch Results");self.save_batch_results_button.setIcon(ICON_BATCH_SAVE_RESULTS if not ICON_BATCH_SAVE_RESULTS.isNull() else QIcon());self.save_batch_results_button.setEnabled(False);self.save_batch_results_button.clicked.connect(self._save_batch_prediction_results);batch_actions_layout.addWidget(self.save_batch_results_button)
        batch_actions_layout.addStretch();batch_pred_layout.addLayout(batch_actions_layout)
        self.batch_predictions_table=QTableWidget();self.batch_predictions_table.setObjectName("BatchPredictionsTable");self.batch_predictions_table.setMinimumHeight(150);self.batch_predictions_table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.batch_predictions_table.setAlternatingRowColors(True);self.batch_predictions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive);self.batch_predictions_table.horizontalHeader().setDefaultSectionSize(100);batch_pred_layout.addWidget(self.batch_predictions_table);layout.addWidget(batch_pred_group)
        layout.addStretch(1);self.tabs.addTab(tab_widget,"")

    def _setup_models_tab(self): 
        tab_widget=QWidget();layout=QVBoxLayout(tab_widget);layout.setSpacing(10);layout.setContentsMargins(8,8,8,8)
        history_group=QGroupBox("Model Training History (Current Session)");history_group.setObjectName("HistoryGroup");history_layout=QVBoxLayout(history_group);history_layout.setContentsMargins(10,5,10,10);self.model_comparison_table=QTableWidget();self.model_comparison_table.setColumnCount(8);self.model_comparison_table.setHorizontalHeaderLabels(["Run ID","Algorithm","# Input Feats","# Train Samples","Train R²","Test R²","CV Mean R²","Timestamp"]);self.model_comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents);self.model_comparison_table.horizontalHeader().setStretchLastSection(True);self.model_comparison_table.setSelectionBehavior(QAbstractItemView.SelectRows);self.model_comparison_table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.model_comparison_table.setAlternatingRowColors(True);self.model_comparison_table.setMinimumHeight(150);history_layout.addWidget(self.model_comparison_table);layout.addWidget(history_group,1)
        saved_models_group=QGroupBox("Manage Saved Models");saved_models_group.setObjectName("SavedModelsGroup");saved_models_layout=QVBoxLayout(saved_models_group);saved_models_layout.setContentsMargins(10,5,10,10);saved_models_layout.setSpacing(8);self.saved_models_list=QListWidget();self.saved_models_list.setSelectionMode(QAbstractItemView.SingleSelection);self.saved_models_list.setContextMenuPolicy(Qt.CustomContextMenu);self.saved_models_list.customContextMenuRequested.connect(self._show_saved_models_context_menu);self.saved_models_list.itemSelectionChanged.connect(self._on_saved_model_selection_change);saved_models_layout.addWidget(self.saved_models_list)
        model_actions_layout=QHBoxLayout();model_actions_layout.setSpacing(8);self.load_model_button=QPushButton(" Load Selected Model");self.delete_model_button=QPushButton(" Delete Selected Model");self.export_report_button=QPushButton(" Export Model Report")
        self.load_model_button.setIcon(ICON_LOAD if not ICON_LOAD.isNull() else QIcon());self.delete_model_button.setObjectName("DeleteButton");self.delete_model_button.setIcon(ICON_DELETE if not ICON_DELETE.isNull() else QIcon());self.export_report_button.setIcon(ICON_EXPORT if not ICON_EXPORT.isNull() else QIcon())
        model_actions_layout.addStretch();model_actions_layout.addWidget(self.load_model_button);model_actions_layout.addWidget(self.delete_model_button);model_actions_layout.addWidget(self.export_report_button);model_actions_layout.addStretch()
        self.export_report_button.clicked.connect(self.export_report_for_current_model);self.delete_model_button.clicked.connect(self.delete_selected_model);self.load_model_button.clicked.connect(self.load_selected_model)
        saved_models_layout.addLayout(model_actions_layout);layout.addWidget(saved_models_group,1);self.tabs.addTab(tab_widget,"")

    def _update_training_progress(self, value: int, message: str): 
        if hasattr(self, 'progress_bar'): self.progress_bar.setValue(value)
        if hasattr(self, 'progress_status_label'): self.progress_status_label.setText(message)
        logger.debug(f"Training Progress: {value}% - {message}")

    def _on_training_thread_finished(self): 
        logger.debug("TrainingThread process finished.")
        if hasattr(self, 'train_button') and not self.train_button.isEnabled():
            if hasattr(self, 'save_model_button') and not self.save_model_button.isEnabled():
                self.train_button.setEnabled(True); self.train_button.setText(" Train Model")
                if not ICON_TRAIN.isNull(): self.train_button.setIcon(ICON_TRAIN)
                logger.info("Training thread finished, possibly due to error or interruption. Train button re-enabled.")


    def _on_saved_model_selection_change(self): 
        is_item_selected = hasattr(self, 'saved_models_list') and self.saved_models_list.currentItem() is not None and \
                                 self.saved_models_list.currentItem().text() != "No saved models found in directory."
        if hasattr(self, 'load_model_button'): self.load_model_button.setEnabled(is_item_selected)
        if hasattr(self, 'delete_model_button'): self.delete_model_button.setEnabled(is_item_selected)
        if hasattr(self, 'export_report_button'): self.export_report_button.setEnabled(self.current_model_pipeline is not None and bool(self.current_model_metrics))


    def _show_saved_models_context_menu(self, position): 
        selected_item = self.saved_models_list.itemAt(position)
        if not selected_item or selected_item.text() == "No saved models found in directory.": return
        self.saved_models_list.setCurrentItem(selected_item); context_menu = QMenu(self)
        load_action = context_menu.addAction("Load This Model", self.load_selected_model)
        if not ICON_LOAD.isNull(): load_action.setIcon(ICON_LOAD)
        delete_action = context_menu.addAction("Delete This Model", self.delete_selected_model)
        if not ICON_DELETE.isNull(): delete_action.setIcon(ICON_DELETE)
        context_menu.addSeparator()
        details_action = context_menu.addAction("View Model Details", lambda: self._show_model_details(selected_item.text()))
        if not ICON_INFO.isNull(): details_action.setIcon(ICON_INFO)
        context_menu.exec_(self.saved_models_list.mapToGlobal(position))

    def _show_model_details(self, model_name_base: str): 
        meta_file_path = os.path.join(self.saved_models_dir, f"{model_name_base}.json")
        if not os.path.exists(meta_file_path): self._show_message("Error", f"Metadata file for '{model_name_base}' not found.", "critical"); return
        try:
            with open(meta_file_path, 'r') as f: metadata = json.load(f, object_hook=decode_np_array)
            features_list = metadata.get('features_input_to_pipeline', [])
            features_str = (', '.join(map(str, features_list[:5])) + ('...' if len(features_list) > 5 else '')) if features_list else 'N/A'
            hp_items = "".join([f"<li style='margin-bottom:2px;'><b>{k.replace('_',' ').title()}:</b> {v}</li>" for k,v in metadata.get('hyperparams_config',{}).items()])
            metrics_items = "".join([f"<li style='margin-bottom:2px;'><b>{k.replace('_',' ').title()}:</b> {v:.4f if isinstance(v,float) else v}</li>" for k,v in metadata.get('metrics',{}).items() if k in ['train_r2','test_r2','cv_mean_r2','test_mse', 'num_samples_train', 'num_features_to_model']])
            details_html = f"""<div style='font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 1.5;'><p><b>Model Name:</b> {metadata.get('model_name','N/A')}</p><p><b>Algorithm:</b> {metadata.get('model_type','N/A')}</p><p><b>Saved On:</b> {metadata.get('timestamp','N/A')}</p><hr style='border:0; border-top:1px solid #eee; margin: 8px 0;'/><p><b>Input Features ({len(features_list if features_list else [])}):</b> {features_str}</p><p><b>Target Variable:</b> {metadata.get('target','N/A')}</p><hr style='border:0; border-top:1px solid #eee; margin: 8px 0;'/><p><b>Hyperparameters Configured:</b></p><ul style='margin-left:15px; padding-left:0;'>{hp_items if hp_items else "<li>Default or not specified</li>"}</ul><hr style='border:0; border-top:1px solid #eee; margin: 8px 0;'/><p><b>Key Performance Metrics:</b></p><ul style='margin-left:15px; padding-left:0;'>{metrics_items if metrics_items else "<li>Not available</li>"}</ul></div>"""
            details_msg_box = QMessageBox(self); details_msg_box.setWindowTitle(f"Details for: {model_name_base}"); details_msg_box.setTextFormat(Qt.RichText) ; details_msg_box.setText(details_html)
            details_msg_box.setIcon(QMessageBox.Information); details_msg_box.setStandardButtons(QMessageBox.Ok); details_msg_box.exec_()
        except Exception as e: self._show_message("Error", f"Could not load or display model details: {e}", "critical"); logger.error(f"Error showing model details for '{model_name_base}': {e}", exc_info=True)

    def _open_feature_target_dialog(self): 
        if not self.data_viewer_instance or self.data_viewer_instance.data is None or self.data_viewer_instance.data.empty: self._show_message("No Data Loaded", "Please import a dataset before selecting features and target.", "warning"); return
        all_columns = self.data_viewer_instance.data.columns.tolist(); numeric_columns = self.data_viewer_instance.data.select_dtypes(include=np.number).columns.tolist()
        dialog = FeatureTargetDialog(all_columns, numeric_columns, self.selected_features, self.selected_target, self)
        if dialog.exec_():
            selected_f, selected_t = dialog.get_selections()
            if not selected_f or not selected_t: self._show_message("Selection Incomplete", "Both features (at least one) and a target variable must be selected.", "warning"); return
            self.selected_features = selected_f; self.selected_target = selected_t
            self.selected_features_label.setText(f"<b>Features ({len(self.selected_features)}):</b> {', '.join(self.selected_features[:3])}{'...' if len(self.selected_features)>3 else ''}")
            self.selected_target_label.setText(f"<b>Target:</b> {self.selected_target}")
            if self.parent_window and hasattr(self.parent_window, 'status_label'): self.parent_window.status_label.setText(f"Features and target selected. Ready to configure model.")
        else:
            if self.parent_window and hasattr(self.parent_window, 'status_label'): self.parent_window.status_label.setText(f"Feature and target selection cancelled.")

    def _validate_training_inputs(self) -> bool: 
        if not self.data_viewer_instance or self.data_viewer_instance.data is None or self.data_viewer_instance.data.empty: self._show_message("No Data", "Cannot start training. Load a dataset first.", "critical"); return False
        if not self.selected_features: self._show_message("Input Error", "No features selected. Select input features (X).", "warning"); return False
        if not self.selected_target: self._show_message("Input Error", "No target variable selected. Select a target (y).", "warning"); return False
        if self.selected_target in self.selected_features: self._show_message("Input Error", "Target variable cannot also be an input feature.", "warning"); return False
        if self.selected_target in self.data_viewer_instance.data.columns and self.data_viewer_instance.data[self.selected_target].isnull().any():
            reply = QMessageBox.question(self,"Missing Target Values",f"Target '{self.selected_target}' has missing values. Rows with missing target will be excluded.\nClean first or proceed?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if reply == QMessageBox.No: return False
        return True

    def _prepare_training_data(self) -> tuple[pd.DataFrame|None, pd.Series|None, list|None]: 
        if not self.data_viewer_instance or self.data_viewer_instance.data is None: return None,None,None
        data_copy=self.data_viewer_instance.data.copy()
        try:
            relevant_columns=self.selected_features+[self.selected_target]; data_subset=data_copy[relevant_columns]
            data_processed=handle_missing_and_duplicates(data_subset.copy(),fill_method_numeric='median',fill_method_object='mode',remove_duplicates=True)
            data_processed.dropna(subset=[self.selected_target],inplace=True)
            if data_processed.empty: self._show_message("Data Preparation Error","No data remains after handling missing target values.","critical"); return None,None,None
            
            X=data_processed[self.selected_features].copy(); y=data_processed[self.selected_target].copy(); 
            self.label_encoders={}
            
            for col in X.select_dtypes(include=['object','category']).columns: 
                le=LabelEncoder();
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown').astype(str)
                X[col]=le.fit_transform(X[col]);
                self.label_encoders[col]=le
            
            if y.dtype=='object' or pd.api.types.is_categorical_dtype(y): 
                logger.warning(f"Target variable '{self.selected_target}' is categorical/object. Attempting to encode for regression. This may not be appropriate.")
                le_target=LabelEncoder();
                y = y.fillna(y.mode()[0] if not y.mode().empty else 'Unknown_Target_Placeholder').astype(str)
                y=le_target.fit_transform(y);
                self.label_encoders[self.selected_target]=le_target
            
            if X.empty or len(y)==0: self._show_message("Data Preparation Error","Not enough data after preprocessing.","critical"); return None,None,None
            if len(X) != len(y): self._show_message("Data Preparation Error","X and y have mismatched lengths after preprocessing.","critical"); return None,None,None
            
            return X,y,self.selected_features
        except Exception as e: 
            logger.error(f"Data preparation error: {e}",exc_info=True);
            self._show_message("Data Preparation Error",f"Failed to prepare data: {str(e)[:150]}","critical"); 
            return None,None,None


    def start_training(self):
        if not self._validate_training_inputs(): return
        
        X_prepared, y_prepared, _ = self._prepare_training_data()
        if X_prepared is None or y_prepared is None or X_prepared.empty:
            logger.error("Data prep failed. Training aborted.")
            self.train_button.setEnabled(True); self.train_button.setText(" Train Model"); self.train_button.setIcon(ICON_TRAIN if not ICON_TRAIN.isNull() else QIcon())
            return

        current_model_type_str = "Auto Select (via CV)"
        if self.model_category_tabs:
            active_tab_text = self.model_category_tabs.tabText(self.model_category_tabs.currentIndex())
            if active_tab_text == "Auto Select (via CV)":
                current_model_type_str = "Auto Select (via CV)"
            elif active_tab_text in self.category_comboboxes:
                combo = self.category_comboboxes[active_tab_text]
                current_model_type_str = combo.currentData() or combo.currentText()
            else:
                logger.warning(f"Could not determine model from active tab '{active_tab_text}'. Defaulting to Auto Select.")
        else:
            logger.error("model_category_tabs not initialized. Defaulting to Auto Select for training.")
            
        logger.info(f"Starting training for model type: {current_model_type_str}")
        current_hyperparams = self._get_current_hyperparams()

        self.train_button.setEnabled(False); self.train_button.setText(" Training..."); self.train_button.setIcon(ICON_BUSY if not ICON_BUSY.isNull() else QIcon()); self.save_model_button.setEnabled(False)
        self.progress_bar.setValue(0); self.progress_status_label.setText("Initializing training..."); self._clear_evaluation_plots_display("Training in progress...")
        
        if self.training_thread and self.training_thread.isRunning(): 
            logger.info("Terminating previous training thread."); 
            self.training_thread.requestInterruption()
            self.training_thread.quit()
            if not self.training_thread.wait(1500):
                logger.warning("Training thread did not terminate gracefully, forcing termination.")
                self.training_thread.terminate()
                self.training_thread.wait(500)
        
        self.training_thread = TrainingThread(X_prepared, y_prepared, current_model_type_str, current_hyperparams, self)
        self.training_thread.progress_updated.connect(self._update_training_progress)
        self.training_thread.training_complete.connect(self._on_training_complete)
        self.training_thread.error_occurred.connect(self._on_training_error)
        self.training_thread.finished.connect(self._on_training_thread_finished)
        self.training_thread.start()

    def _on_training_error(self, error_message: str): 
        self._show_message("Training Error", error_message, "critical"); 
        self.train_button.setEnabled(True); self.train_button.setText(" Train Model"); self.train_button.setIcon(ICON_TRAIN if not ICON_TRAIN.isNull() else QIcon())
        self.progress_status_label.setText("Training failed. See message."); self.progress_bar.setValue(0); self._clear_evaluation_plots_display("Training failed.")

    def _on_training_complete(self, model_pipeline, metrics, X_train, y_train, X_test, y_test, feature_importance_dict): 
        self.current_model_pipeline=model_pipeline; self.current_model_metrics=metrics.copy(); self.X_train_data,self.y_train_data=X_train,y_train; self.X_test_data,self.y_test_data=X_test,y_test; self.feature_importance_data=feature_importance_dict
        run_id=f"{metrics['model_type'].replace(' ','')}_Run{self.model_comparison_table.rowCount()+1}"; self.current_model_metrics['model_name']=run_id
        self._update_evaluation_metrics_table(self.current_model_metrics); self._update_model_comparison_table(self.current_model_metrics); self._update_prediction_inputs_ui(self.selected_features)
        self.save_model_button.setEnabled(True); self.train_button.setEnabled(True); self.train_button.setText(" Train Model"); self.train_button.setIcon(ICON_TRAIN if not ICON_TRAIN.isNull() else QIcon())
        self.progress_status_label.setText("Training complete! Generating plots..."); self.progress_bar.setValue(100)
        status_msg=f"Model '{run_id}' trained. Evaluation results available."; (self.parent_window.status_label.setText(status_msg) if self.parent_window and hasattr(self.parent_window,'status_label') else self._show_message("Training Successful",status_msg,"information"))
        self.tabs.setCurrentIndex(1)
        if self.feature_importance_data: self._start_evaluation_plot_generation("feature_importance",{"importance_dict":self.feature_importance_data})
        else: self._update_specific_evaluation_plot("feature_importance",None,"Feature importances not available for this model.")
        if self.X_test_data is not None and not self.X_test_data.empty and self.y_test_data is not None and not self.y_test_data.empty:
            try: y_pred_plot=self.current_model_pipeline.predict(self.X_test_data); self._start_evaluation_plot_generation("actual_vs_predicted",{"y_test":self.y_test_data,"y_pred_test":y_pred_plot})
            except Exception as e: logger.error(f"Predict for plot error: {e}"); self._update_specific_evaluation_plot("actual_vs_predicted",None,f"Plot error: {str(e)[:50]}")
        else: self._update_specific_evaluation_plot("actual_vs_predicted",None,"Test data N/A for plot.")
        self.export_report_button.setEnabled(True)

    def _start_evaluation_plot_generation(self, plot_type: str, data_dict: dict): 
        if plot_type in self.evaluation_plot_threads and self.evaluation_plot_threads[plot_type].isRunning(): logger.info(f"Plot '{plot_type}' busy. Terminating."); self.evaluation_plot_threads[plot_type].terminate(); self.evaluation_plot_threads[plot_type].wait(200)
        thread=EvaluationPlotThread(plot_type,data_dict,self); thread.plot_generated.connect(self._update_specific_evaluation_plot); thread.plot_error.connect(self._handle_evaluation_plot_error)
        thread.finished.connect(lambda pt=plot_type:self._on_eval_plot_finished(pt)); self.evaluation_plot_threads[plot_type]=thread
        canvas=self.feature_importance_canvas if plot_type=="feature_importance" else self.actual_predicted_canvas
        if canvas: canvas.figure.clear();ax=canvas.figure.add_subplot(111);ax.text(0.5,0.5,f"Generating {plot_type.replace('_',' ')}...",ha='center',va='center',fontsize=9,c='grey');ax.set_axis_off();canvas.draw_idle()
        thread.start()

    def _on_eval_plot_finished(self, plot_type:str): 
        logger.info(f"Eval plot thread '{plot_type}' finished.");
        if plot_type in self.evaluation_plot_threads:
            del self.evaluation_plot_threads[plot_type]
        if not self.evaluation_plot_threads and hasattr(self,'progress_status_label'): self.progress_status_label.setText("Evaluation plots ready.")

    def _update_specific_evaluation_plot(self, plot_name: str, fig: Figure | None, error_message: str | None = None): 
        canvas_map={"feature_importance":self.feature_importance_canvas,"actual_vs_predicted":self.actual_predicted_canvas}; target_canvas=canvas_map.get(plot_name)
        if not target_canvas: logger.warning(f"Canvas for '{plot_name}' not found."); return
        target_canvas.figure.clear()
        if fig: target_canvas.figure=fig; target_canvas.draw_idle(); logger.info(f"Eval plot '{plot_name}' updated.")
        elif error_message: self._handle_evaluation_plot_error(plot_name,error_message)
        else: ax=target_canvas.figure.add_subplot(111);ax.text(0.5,0.5,f"{plot_name.replace('_',' ')} N/A.",ha='center',va='center',c='grey',fontsize=9,wrap=True);ax.set_axis_off();target_canvas.draw_idle()

    def _handle_evaluation_plot_error(self, plot_name: str, error_message: str): 
        logger.error(f"Error in eval plot '{plot_name}': {error_message}")
        canvas_map={"feature_importance":self.feature_importance_canvas,"actual_vs_predicted":self.actual_predicted_canvas}; target_canvas=canvas_map.get(plot_name)
        if target_canvas: target_canvas.figure.clear();ax=target_canvas.figure.add_subplot(111);ax.text(0.5,0.5,f"Error: {plot_name.replace('_',' ')}\n{error_message[:80]}",ha='center',va='center',fontsize=8,c='darkred',wrap=True);ax.set_axis_off();target_canvas.draw_idle()

    def _update_evaluation_metrics_table(self, metrics: dict): 
        self.metrics_table.setRowCount(0)
        disp_order=[('R² Score (Coefficient of Determination)','train_r2','test_r2','cv_mean_r2','cv_std_r2'),('Mean Squared Error (MSE)','train_mse','test_mse','cv_mean_mse','cv_std_mse')]
        for i,(name,tr_k,ts_k,cv_m_k,cv_s_k) in enumerate(disp_order):
            self.metrics_table.insertRow(i); self.metrics_table.setItem(i,0,QTableWidgetItem(name))
            tr_v,ts_v,cv_m_v,cv_s_v=metrics.get(tr_k),metrics.get(ts_k),metrics.get(cv_m_k),metrics.get(cv_s_k)
            for c,val in enumerate([tr_v,ts_v,cv_m_v,cv_s_v]): self.metrics_table.setItem(i,c+1,QTableWidgetItem(f"{val:.4f}" if val is not None and isinstance(val,(float,np.floating)) else "N/A"))
            diff=(ts_v-tr_v) if ts_v is not None and tr_v is not None else None; item=QTableWidgetItem(f"{diff:.4f}" if diff is not None else "N/A")
            if name.startswith('R²') and diff is not None and diff<-0.05: item.setForeground(QColor("darkRed")); item.setToolTip("Test R² is significantly lower than Train R², potential overfitting.")
            if name.startswith('Mean Squared Error') and diff is not None and tr_v is not None and tr_v > 1e-9 and (diff/tr_v)>0.2: item.setForeground(QColor("darkOrange")); item.setToolTip("Test MSE is notably higher than Train MSE.")
            self.metrics_table.setItem(i,5,item)
        self.metrics_table.resizeRowsToContents(); self.metrics_table.resizeColumnsToContents()

    def _update_model_comparison_table(self, metrics: dict): 
        r=self.model_comparison_table.rowCount(); self.model_comparison_table.insertRow(r)
        items=[metrics.get('model_name',f"Run-{r+1}"),metrics.get('model_type',"N/A"),str(metrics.get('num_features_input_to_pipeline','?')),str(metrics.get('num_samples_train','?')),f"{metrics.get('train_r2',0):.3f}",f"{metrics.get('test_r2',0):.3f}",f"{metrics.get('cv_mean_r2',0):.3f} (±{metrics.get('cv_std_r2',0):.3f})",metrics.get('timestamp',"N/A")]
        for c,t in enumerate(items):self.model_comparison_table.setItem(r,c,QTableWidgetItem(t))
        self.model_comparison_table.sortItems(7,Qt.DescendingOrder); self.model_comparison_table.resizeColumnsToContents()

    def _update_prediction_inputs_ui(self, feature_names: list): 
        while self.prediction_form_layout.rowCount()>0: self.prediction_form_layout.removeRow(0)
        self.prediction_input_widgets={};
        if not self.current_model_pipeline: self.prediction_form_layout.addRow(QLabel("Train or load a model to enable predictions.")); return
        if not feature_names: self.prediction_form_layout.addRow(QLabel("Model has no specified input features.")); return
        original_df=self.data_viewer_instance.data if self.data_viewer_instance and self.data_viewer_instance.data is not None else None
        for fn in feature_names:
            lbl=QLabel(f"{str(fn)[:25]}:"); lbl.setToolTip(str(fn)); widget=None
            if fn in self.label_encoders and hasattr(self.label_encoders[fn],'classes_'):
                widget=QComboBox(); classes=self.label_encoders[fn].classes_.astype(str); widget.addItems(classes)
                if original_df is not None and fn in original_df.columns: mode_v=original_df[fn].mode(); (widget.setCurrentText(str(mode_v[0])) if not mode_v.empty and str(mode_v[0]) in classes else None)
            elif original_df is not None and fn in original_df.columns and pd.api.types.is_numeric_dtype(original_df[fn]):
                widget=QDoubleSpinBox(); widget.setRange(-1e9,1e9); widget.setDecimals(4)
                med_v=original_df[fn].median(); (widget.setValue(float(med_v)) if pd.notna(med_v) else widget.setValue(0.0))
            else: widget=QLineEdit(); widget.setPlaceholderText(f"Val for {str(fn)[:15]}...")
            if widget: widget.setToolTip(f"Value for: {fn}"); self.prediction_form_layout.addRow(lbl,widget); self.prediction_input_widgets[fn]=widget
        if hasattr(self.prediction_inputs_scroll, 'widget') and self.prediction_inputs_scroll.widget():
            self.prediction_inputs_scroll.widget().adjustSize()


    def make_prediction(self): 
        if not self.current_model_pipeline: self._show_message("No Model Loaded","Train or load a model.","critical"); self.prediction_result_label.setText("Error: No model."); self.prediction_result_label.setProperty("type","error"); self.style().polish(self.prediction_result_label) if self.style() else None; return
        in_vals={};
        try:
            for name,widget in self.prediction_input_widgets.items():
                if isinstance(widget,QDoubleSpinBox):in_vals[name]=widget.value()
                elif isinstance(widget,QComboBox):txt=widget.currentText();in_vals[name]=self.label_encoders[name].transform([txt])[0] if name in self.label_encoders else txt
                elif isinstance(widget,QLineEdit):
                    v=widget.text().strip()
                    if not v:
                        is_numeric = False
                        if self.data_viewer_instance and self.data_viewer_instance.data is not None and \
                           name in self.data_viewer_instance.data.columns and \
                           pd.api.types.is_numeric_dtype(self.data_viewer_instance.data[name]):
                            is_numeric = True
                        if is_numeric: in_vals[name] = 0.0; logger.warning(f"Empty input for numeric feature '{name}', using 0.0.")
                        else: raise ValueError(f"Empty input for feature '{name}'.")
                    else:
                        try: 
                            in_vals[name]=float(v)
                        except ValueError:
                            if name in self.label_encoders:
                                try:
                                    in_vals[name] = self.label_encoders[name].transform([v])[0]
                                except ValueError:
                                    raise ValueError(f"Value '{v}' for '{name}' not in known categories.")
                            else:
                                raise ValueError(f"Invalid numeric input '{v}' for feature '{name}'.")
            
            in_df=pd.DataFrame([in_vals])[self.selected_features];
            pred=self.current_model_pipeline.predict(in_df)
            predicted_value_display = pred[0]
            if self.selected_target and self.selected_target in self.label_encoders and hasattr(self.label_encoders[self.selected_target], 'inverse_transform'):
                try:
                    predicted_value_display = self.label_encoders[self.selected_target].inverse_transform(pred)[0]
                    self.prediction_result_label.setText(f"Predicted Value: {predicted_value_display} (raw: {pred[0]:.4f})")
                except Exception as e_inv:
                    logger.warning(f"Could not inverse_transform prediction: {e_inv}. Displaying raw value.")
                    self.prediction_result_label.setText(f"Predicted Value: {pred[0]:.4f}")
            else:
                self.prediction_result_label.setText(f"Predicted Value: {pred[0]:.4f}")
            self.prediction_result_label.setProperty("type","success")
        except KeyError as ke: self._show_message("Prediction Error",f"Feature mismatch: {ke}. Ensure model's features are correctly provided.","critical");self.prediction_result_label.setText("Error: Feature mismatch.");self.prediction_result_label.setProperty("type","error");logger.error(f"KeyError prediction: {ke}",exc_info=True)
        except ValueError as ve: self._show_message("Input Error",f"Invalid input: {ve}","warning");self.prediction_result_label.setText("Error: Invalid input.");self.prediction_result_label.setProperty("type","error");logger.warning(f"ValueError prediction: {ve}",exc_info=True)
        except Exception as e: self._show_message("Prediction Error",f"Error during prediction: {e}","critical");self.prediction_result_label.setText("Error: Prediction failed!");self.prediction_result_label.setProperty("type","error");logger.error(f"Unexpected prediction error: {e}",exc_info=True)
        if self.style(): self.prediction_result_label.style().polish(self.prediction_result_label)

    def _load_batch_features_file(self): 
        if not self.current_model_pipeline or not self.selected_features: self._show_message("Model Required","Train/load a model first.","warning"); return
        fPath,_=QFileDialog.getOpenFileName(self,"Load Batch Features File","","Data Files (*.csv *.xlsx);;All Files (*.*)");
        if not fPath: return
        try:
            df=pd.read_csv(fPath) if fPath.lower().endswith('.csv') else pd.read_excel(fPath)
            missing=[f for f in self.selected_features if f not in df.columns]
            if missing: self.batch_features_df=None; self.run_batch_predict_button.setEnabled(False); self.save_batch_results_button.setEnabled(False); self.batch_file_label.setText("Error: File missing features!"); self.batch_file_label.setStyleSheet("color:red;font-style:italic;font-size:10px;padding-left:5px;"); self._show_message("Feature Mismatch",f"Batch file missing model features: {', '.join(missing)}","error"); return
            self.batch_features_df=df; self.batch_file_label.setText(os.path.basename(fPath)); self.batch_file_label.setStyleSheet("color:#333;font-style:normal;font-size:10.5px;padding-left:5px;"); self.run_batch_predict_button.setEnabled(True)
            self.save_batch_results_button.setEnabled(False); self.last_batch_prediction_results_df=None; self.batch_predictions_table.clearContents(); self.batch_predictions_table.setRowCount(0)
            if self.parent_window and hasattr(self.parent_window,'status_label'): self.parent_window.status_label.setText(f"Batch file '{os.path.basename(fPath)}' loaded.")
        except Exception as e: self.batch_features_df=None; self.run_batch_predict_button.setEnabled(False); self.save_batch_results_button.setEnabled(False); self.batch_file_label.setText("Error loading file!"); self._show_message("File Load Error",f"Failed to load batch file: {e}","critical"); logger.error(f"Error loading batch file '{fPath}': {e}",exc_info=True)

    def _prepare_batch_data_for_prediction(self, df_batch_input: pd.DataFrame) -> pd.DataFrame | None: 
        if df_batch_input is None or not self.selected_features: logger.error("Batch data/selected features missing."); return None
        try:
            df_proc=df_batch_input[self.selected_features].copy()
            for fn,enc in self.label_encoders.items():
                if fn in df_proc.columns and (pd.api.types.is_object_dtype(df_proc[fn]) or pd.api.types.is_categorical_dtype(df_proc[fn])):
                    known_cls=list(enc.classes_.astype(str))
                    df_proc[fn]=df_proc[fn].astype(str).apply(lambda x:enc.transform([x])[0] if x in known_cls else np.nan)
                    if df_proc[fn].isnull().any(): 
                        logger.warning(f"Unknown categories in batch for '{fn}' marked NaN & will be imputed.")
            
            for col in df_proc.columns:
                if df_proc[col].isnull().any():
                    fill_val = 0
                    if pd.api.types.is_numeric_dtype(df_proc[col]):
                        fill_val = 0.0 
                    else:
                        fill_val = 0 
                    df_proc[col].fillna(fill_val,inplace=True)
                    logger.info(f"Batch NaNs in '{col}' imputed with {fill_val} before prediction.")
            return df_proc
        except KeyError as ke: logger.error(f"Batch data prep - Missing feature: {ke}",exc_info=True); self._show_message("Batch Data Error",f"Batch file missing required feature: '{ke}'.","critical"); return None
        except Exception as e: logger.error(f"Batch data prep error: {e}",exc_info=True); self._show_message("Batch Prep Error",f"Error preparing batch data: {e}","critical"); return None

    def _run_batch_prediction(self):
        if not self.current_model_pipeline: self._show_message("No Model","No model loaded/trained.","critical"); return
        if self.batch_features_df is None or self.batch_features_df.empty: self._show_message("No Batch Data","No batch data loaded or file is empty.","critical"); return
        self.run_batch_predict_button.setEnabled(False); self.save_batch_results_button.setEnabled(False); self.progress_status_label.setText("Running batch predictions..."); QApplication.processEvents()
        try:
            prep_df=self._prepare_batch_data_for_prediction(self.batch_features_df)
            if prep_df is None or prep_df.empty: self.progress_status_label.setText("Batch predict: Data prep error."); self.run_batch_predict_button.setEnabled(True); return
            preds=self.current_model_pipeline.predict(prep_df)
            res_df_display=self.batch_features_df.copy()
            pred_col_base="Predicted_Value"; pred_col=pred_col_base; i=0
            while pred_col in res_df_display.columns: i+=1; pred_col=f"{pred_col_base}_{i}"
            if self.selected_target and self.selected_target in self.label_encoders and hasattr(self.label_encoders[self.selected_target], 'inverse_transform'):
                try: res_df_display[pred_col] = self.label_encoders[self.selected_target].inverse_transform(preds)
                except Exception as e_inv_batch: logger.warning(f"Could not inverse_transform batch predictions: {e_inv_batch}. Using raw."); res_df_display[pred_col] = preds
            else: res_df_display[pred_col]=preds
            self.last_batch_prediction_results_df=res_df_display.copy()
            self.batch_predictions_table.clearContents(); self.batch_predictions_table.setRowCount(res_df_display.shape[0]); self.batch_predictions_table.setColumnCount(res_df_display.shape[1])
            self.batch_predictions_table.setHorizontalHeaderLabels(res_df_display.columns.astype(str))
            for r_idx in range(res_df_display.shape[0]):
                for c_idx in range(res_df_display.shape[1]): val=res_df_display.iloc[r_idx,c_idx]; self.batch_predictions_table.setItem(r_idx,c_idx,QTableWidgetItem(f"{val:.4f}" if isinstance(val,(float,np.floating)) else str(val)))
            self.batch_predictions_table.resizeColumnsToContents(); self.progress_status_label.setText(f"Batch predict done: {len(preds)} predictions.")
            self._show_message("Batch Prediction Success","Batch predictions generated.","information"); self.save_batch_results_button.setEnabled(True)
        except Exception as e: 
            logger.error(f"Batch predict run error: {e}",exc_info=True)
            self.progress_status_label.setText("Batch predict failed.")
            self._show_message("Batch Error",f"Failed: {str(e)[:200]}","critical")
            self.last_batch_prediction_results_df=None
        finally: 
            self.run_batch_predict_button.setEnabled(True)

    def _save_batch_prediction_results(self):
        if self.last_batch_prediction_results_df is None or self.last_batch_prediction_results_df.empty: self._show_message("No Results","No batch prediction results to save.","warning"); return
        original_file_name="batch_predictions"; current_batch_label=self.batch_file_label.text()
        if current_batch_label and not current_batch_label.startswith("Error") and not current_batch_label.startswith("No file"): original_file_name=os.path.splitext(current_batch_label)[0]+"_with_predictions"
        default_path=os.path.join(os.getcwd(),f"{original_file_name}.csv")
        file_path,selected_filter=QFileDialog.getSaveFileName(self,"Save Batch Prediction Results",default_path,"CSV Files (*.csv);;Excel Files (*.xlsx)")
        if not file_path: return
        try:
            if not file_path.lower().endswith((".csv", ".xlsx")):
                 if "csv" in selected_filter.lower(): file_path += ".csv"
                 elif "xlsx" in selected_filter.lower(): file_path += ".xlsx"
                 else: file_path += ".csv"

            if file_path.lower().endswith(".csv"):
                self.last_batch_prediction_results_df.to_csv(file_path,index=False)
            elif file_path.lower().endswith(".xlsx"):
                self.last_batch_prediction_results_df.to_excel(file_path,index=False,engine='openpyxl')
            else:
                self._show_message("Save Error", "Unsupported file type selected for saving.", "critical")
                return

            self._show_message("Save Successful",f"Batch results saved to:\n{os.path.basename(file_path)}","information")
            if self.parent_window and hasattr(self.parent_window,'status_label'): self.parent_window.status_label.setText(f"Batch results saved: {os.path.basename(file_path)}.")
        except Exception as e: self._show_message("Save Error",f"Failed to save batch results: {str(e)[:200]}","critical"); logger.error(f"Error saving batch results to '{file_path}': {e}",exc_info=True)

    def save_model(self):
        if not self.current_model_pipeline: self._show_message("No Model to Save","No trained model available.","warning");return
        default_name=self.current_model_metrics.get('model_name',f"{self.current_model_metrics.get('model_type','UnknownModel').replace(' ','')}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
        path,_=QFileDialog.getSaveFileName(self,"Save Trained Model",os.path.join(self.saved_models_dir,default_name),"Joblib Model Files (*.joblib)");
        if not path: return
        
        if not path.lower().endswith(".joblib"): path += ".joblib"

        base=os.path.splitext(os.path.basename(path))[0];m_path=os.path.join(os.path.dirname(path),f"{base}.joblib");meta_path=os.path.join(os.path.dirname(path),f"{base}.json")
        try:
            joblib.dump(self.current_model_pipeline,m_path)
            meta={'model_name':base,'model_type':self.current_model_metrics.get('model_type'),'timestamp':self.current_model_metrics.get('timestamp'),
                  'features_input_to_pipeline':self.selected_features,'target':self.selected_target,'hyperparams_config':self.current_model_metrics.get('hyperparams'),
                  'metrics':self.current_model_metrics,'label_encoders':{c:enc.classes_.astype(str).tolist() for c,enc in self.label_encoders.items() if hasattr(enc,'classes_')}}
            with open(meta_path,'w') as f:json.dump(meta,f,indent=4,cls=NpEncoder)
            self._show_message("Model Saved",f"Model '{base}' saved.","information");self.refresh_saved_models()
        except Exception as e:self._show_message("Save Error",f"Failed to save model: {str(e)[:200]}","critical");logger.error(f"Error saving model '{base}': {e}",exc_info=True)

    def refresh_saved_models(self):
        self.saved_models_list.clear();files=[]
        if hasattr(self, 'saved_models_dir') and os.path.exists(self.saved_models_dir): files=[os.path.splitext(f)[0] for f in os.listdir(self.saved_models_dir) if f.endswith('.joblib')]
        if not files: self.saved_models_list.addItem("No saved models found in directory.")
        else:
            try: files.sort(key=lambda name:name.split('_')[-1] if '_' in name and name.split('_')[-1].isdigit() else name,reverse=True)
            except: files.sort(reverse=True)
            self.saved_models_list.addItems(files)
        self._on_saved_model_selection_change()

    def load_selected_model(self):
        item=self.saved_models_list.currentItem()
        if not item or item.text()=="No saved models found in directory.": self._show_message("No Model Selected","Select a model to load.","info");return
        name=item.text();m_path=os.path.join(self.saved_models_dir,f"{name}.joblib");meta_path=os.path.join(self.saved_models_dir,f"{name}.json")
        if not os.path.exists(m_path) or not os.path.exists(meta_path): self._show_message("File Not Found",f"Files for '{name}' missing.","critical");self.refresh_saved_models();return
        try:
            self.current_model_pipeline=joblib.load(m_path)
            with open(meta_path,'r') as f:meta=json.load(f,object_hook=decode_np_array)
            self.current_model_metrics=meta.get('metrics',{}); self.current_model_metrics['model_name'] = name
            self.selected_features=meta.get('features_input_to_pipeline',[]);self.selected_target=meta.get('target',None)
            self.label_encoders={};loaded_le_meta=meta.get('label_encoders',{})
            for col,classes_list in loaded_le_meta.items(): le=LabelEncoder();le.classes_=np.array(classes_list,dtype='object');self.label_encoders[col]=le
            self._update_ui_after_load(meta);self._show_message("Model Loaded",f"Model '{name}' loaded.","information")
            self.tabs.setCurrentIndex(2);self.save_model_button.setEnabled(False);self.export_report_button.setEnabled(True)
        except Exception as e: self._show_message("Load Error",f"Failed to load model: {str(e)[:200]}","critical");logger.error(f"Error loading model '{name}': {e}",exc_info=True);self.current_model_pipeline=None;self.export_report_button.setEnabled(False)


    def delete_selected_model(self):
        item=self.saved_models_list.currentItem()
        if not item or item.text()=="No saved models found in directory.": self._show_message("No Model Selected","Select a model to delete.","info");return
        name=item.text();reply=QMessageBox.warning(self,"Confirm Deletion",f"Permanently delete '{name}'?",QMessageBox.Yes|QMessageBox.Cancel,QMessageBox.Cancel)
        if reply==QMessageBox.Yes:
            try:
                m_file=os.path.join(self.saved_models_dir,f"{name}.joblib");meta_file=os.path.join(self.saved_models_dir,f"{name}.json")
                deleted_something = False
                if os.path.exists(m_file):os.remove(m_file); deleted_something=True
                if os.path.exists(meta_file):os.remove(meta_file); deleted_something=True
                if deleted_something: self._show_message("Deletion Successful",f"Model '{name}' deleted.","information")
                else: self._show_message("Deletion Info",f"No files found for model '{name}'.","info")
                self.refresh_saved_models()
                if self.current_model_pipeline and self.current_model_metrics.get('model_name')==name:
                    self.current_model_pipeline=None;self.current_model_metrics={};self.export_report_button.setEnabled(False)
                    self._clear_evaluation_plots_display("Current model deleted.");self._update_prediction_inputs_ui([])
                    self.selected_features_label.setText("Features: Not selected");self.selected_target_label.setText("Target: Not selected");self.metrics_table.setRowCount(0)
                    if hasattr(self, 'prediction_result_label'): self.prediction_result_label.setText("Prediction will appear here.");self.prediction_result_label.setProperty("type", "info");
                    if self.style(): self.style().polish(self.prediction_result_label)
            except Exception as e:self._show_message("Deletion Error",f"Failed to delete '{name}': {str(e)[:200]}","critical");logger.error(f"Error deleting '{name}': {e}",exc_info=True)

    def export_report_for_current_model(self):
        if not self.current_model_pipeline:
            self._show_message("No Model Available","No model trained/loaded.","warning")
            return
        
        if self.X_test_data is None or self.y_test_data is None:
            reply = QMessageBox.question(self, "Limited Report Data",
                                         "Original test data (X_test, y_test) not available for the currently "
                                         "loaded/trained model. Report plots might be limited or missing.\nProceed?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        name_rep = self.current_model_metrics.get('model_name', 
                                                self.current_model_metrics.get('model_type','TrainedModel').replace(' ',''))
        path_def = os.path.join(self.model_reports_dir, 
                                f"{name_rep}_Report_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pdf")
        
        path, _ = QFileDialog.getSaveFileName(self, "Export Model Report", path_def, "PDF Files (*.pdf)")
        
        if not path: 
            return

        if not path.lower().endswith(".pdf"):
            path += ".pdf"
            
        try:
            self._generate_pdf_report(path, self.current_model_pipeline, self.current_model_metrics,
                                      self.X_test_data, self.y_test_data, 
                                      self.selected_features, self.selected_target,
                                      self.feature_importance_data)
            self._show_message("Report Exported", f"Report exported to:\n{os.path.basename(path)}", "information")
            if self.parent_window and hasattr(self.parent_window, 'status_label'):
                self.parent_window.status_label.setText(f"Report for '{name_rep}' exported.")
        except Exception as e:
            self._show_message("Report Export Error", f"Failed to generate report: {str(e)[:200]}", "critical")
            logger.error(f"Report generation error for '{name_rep}': {e}", exc_info=True)

    def _generate_pdf_report(self, file_path, model_pipeline, metrics_dict, X_test, y_test, 
                             feature_list, target_variable_name, feature_importance_values):
        doc = SimpleDocTemplate(file_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.7*inch, rightMargin=0.7*inch)
        styles = getSampleStyleSheet()
        story = []

        styles['h1'].alignment = TA_CENTER
        styles['h1'].fontSize = 16
        styles['h2'].fontSize = 13
        styles['h2'].spaceBefore = 10
        styles['h2'].spaceAfter = 2
        styles['h3'].fontSize = 11
        styles['h3'].spaceBefore = 8
        styles['h3'].spaceAfter = 1
        styles['Normal'].fontSize = 9
        styles['Normal'].leading = 11
        styles['Italic'].fontSize = 8 

        story.append(Paragraph(f"<b>Model Performance Report: {metrics_dict.get('model_name','N/A')}</b>", styles['h1']))
        story.append(Paragraph(f"<font size=8><i>Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Machine Learning Studio (Saeed Ur Rehman)</i></font>", styles['Italic']))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("<u>I. Model Overview & Configuration</u>", styles['h2']))
        overview_txt = (f"<font size=9><b>Algorithm Trained:</b> {metrics_dict.get('model_type','N/A')}<br/>"
                        f"<b>Training Timestamp:</b> {metrics_dict.get('timestamp','N/A')}<br/>"
                        f"<b># Input Features to Pipeline:</b> {metrics_dict.get('num_features_input_to_pipeline','N/A')}<br/>"
                        f"<b># Features to Model (after poly):</b> {metrics_dict.get('num_features_to_model','N/A')}<br/>"
                        f"<b>Selected Target Variable:</b> {target_variable_name or 'N/A'}<br/>"
                        f"<b>Training Samples:</b> {metrics_dict.get('num_samples_train','N/A')} | <b>Test Samples:</b> {metrics_dict.get('num_samples_test','N/A')}</font>")
        story.append(Paragraph(overview_txt, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph("Hyperparameters Used:", styles['h3']))
        hp_list = [f"&nbsp;&nbsp;• <b>{k.replace('_',' ').title()}:</b> {v}" for k,v in metrics_dict.get('hyperparams',{}).items()]
        hp_txt = "<br/>".join(hp_list) if hp_list else "&nbsp;&nbsp;• Default parameters used or not specified."
        story.append(Paragraph(f"<font size=8.5>{hp_txt}</font>", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))

        story.append(Paragraph("<u>II. Key Performance Metrics</u>", styles['h2']))
        met_data = [[Paragraph(col, styles['Normal']) for col in ["<b>Metric</b>","<b>Train Set</b>","<b>Test Set</b>","<b>CV Mean</b>","<b>CV Std Dev</b>"]]]
        met_to_rep = [('R² Score','train_r2','test_r2','cv_mean_r2','cv_std_r2'),
                      ('Mean Squared Error (MSE)','train_mse','test_mse','cv_mean_mse','cv_std_mse')]
        for name,tr_k,ts_k,cv_m_k,cv_s_k in met_to_rep:
            row = [name] + [f"{metrics_dict.get(k, 0.0):.4f}" if metrics_dict.get(k) is not None else "N/A" for k in [tr_k,ts_k,cv_m_k,cv_s_k]]
            met_data.append([Paragraph(str(cell), styles['Normal']) for cell in row])
        
        met_tbl_style = TableStyle([
            ('BACKGROUND',(0,0),(-1,0),reportlab_colors.HexColor("#005a9e")),
            ('TEXTCOLOR',(0,0),(-1,0),reportlab_colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8),
            ('BOTTOMPADDING',(0,0),(-1,0),5),
            ('TOPPADDING',(0,0),(-1,0),3),
            ('BACKGROUND',(0,1),(-1,-1),reportlab_colors.HexColor("#f0f4f8")),
            ('GRID',(0,0),(-1,-1),0.5,reportlab_colors.darkgrey),
            ('BOX',(0,0),(-1,-1),0.5,reportlab_colors.black)
        ])
        met_tbl=Table(met_data, colWidths=[1.7*inch,1.1*inch,1.1*inch,1.1*inch,1.1*inch])
        met_tbl.setStyle(met_tbl_style)
        story.append(met_tbl)
        story.append(Spacer(1,0.2*inch))
        
        plots_to_del = []
        
        if X_test is not None and not X_test.empty and y_test is not None and not y_test.empty and model_pipeline is not None:
            story.append(Paragraph("<u>III. Evaluation Visualizations (on Test Set)</u>", styles['h2']))
            try:
                y_pred_rep = model_pipeline.predict(X_test)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_scatter:
                    fig_s, ax_s = VisualUtils._create_base_figure("Actual vs. Predicted (Test Set)", figsize=(4.5,3.6))
                    sns.scatterplot(x=y_test, y=y_pred_rep, ax=ax_s, alpha=0.4, s=15, color="#0056a3", edgecolor='none')
                    min_d = min(y_test.min(), pd.Series(y_pred_rep).min()) if not y_test.empty and not pd.Series(y_pred_rep).empty else 0
                    max_d = max(y_test.max(), pd.Series(y_pred_rep).max()) if not y_test.empty and not pd.Series(y_pred_rep).empty else 1
                    ax_s.plot([min_d,max_d],[min_d,max_d],'r--',lw=1.0,label="Ideal Fit")
                    ax_s.set_xlabel("Actual Values",fontsize=7); ax_s.set_ylabel("Predicted Values",fontsize=7)
                    ax_s.tick_params(labelsize=6); ax_s.legend(fontsize=6); ax_s.grid(True, linestyle=':', alpha=0.3)
                    fig_s.tight_layout(pad=0.2)
                    fig_s.savefig(tmp_scatter.name, dpi=150); plt.close(fig_s)
                    story.append(Image(tmp_scatter.name, width=3.0*inch, height=2.4*inch))
                    plots_to_del.append(tmp_scatter.name)
                story.append(Spacer(1,0.05*inch))
            except Exception as e_avp: 
                logger.error(f"Report AVP plot error: {e_avp}")
                story.append(Paragraph(f"<font color='red' size=8><i>Actual vs. Predicted plot generation error: {str(e_avp)[:50]}</i></font>", styles['Normal']))
        else:
            story.append(Paragraph("<font size=8><i>Test set visualizations (e.g., Actual vs. Predicted) omitted due to missing test data or model.</i></font>", styles['Normal']))
            story.append(Spacer(1,0.1*inch))

        if feature_importance_values:
            story.append(Paragraph("Top Feature Importances / Coefficients:", styles['h3']))
            try:
                sorted_fi = sorted(feature_importance_values.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
                fi_feats = [str(item[0])[:20] + ('...' if len(str(item[0])) > 20 else '') for item in sorted_fi]
                fi_scores = [item[1] for item in sorted_fi]
                
                if fi_feats:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_fi:
                        fig_fi, ax_fi = VisualUtils._create_base_figure("Feature Importances", figsize=(5.0, max(2.5, len(fi_feats)*0.3))) 
                        y_pos_fi = np.arange(len(fi_feats))
                        ax_fi.barh(y_pos_fi, fi_scores, align='center', height=0.6, color=sns.color_palette("viridis_r", len(fi_feats)))
                        ax_fi.set_yticks(y_pos_fi); ax_fi.set_yticklabels(fi_feats, fontsize=6)
                        ax_fi.invert_yaxis() 
                        ax_fi.set_xlabel("Importance / Coefficient Value", fontsize=7)
                        ax_fi.set_ylabel("Feature", fontsize=7)
                        ax_fi.tick_params(axis='x', labelsize=6)
                        fig_fi.tight_layout(pad=0.3)
                        fig_fi.savefig(tmp_fi.name, dpi=150); plt.close(fig_fi)
                        story.append(Image(tmp_fi.name, width=4.0*inch, height=max(2.0, len(fi_feats)*0.24)*inch)) 
                        plots_to_del.append(tmp_fi.name)
                else:
                    story.append(Paragraph("<font size=8><i>No feature importances to display.</i></font>", styles['Normal']))

            except Exception as e_fi:
                logger.error(f"Report Feature Importance plot error: {e_fi}")
                story.append(Paragraph(f"<font color='red' size=8><i>Feature Importance plot generation error: {str(e_fi)[:50]}</i></font>", styles['Normal']))
        else:
            story.append(Paragraph("<font size=8><i>Feature importances not available or not applicable for this model.</i></font>", styles['Normal']))
        
        story.append(Spacer(1,0.2*inch))
        story.append(Paragraph("--- End of Report ---", styles['Normal']))

        try:
            doc.build(story)
        finally:
            for plot_file in plots_to_del:
                try:
                    if os.path.exists(plot_file):
                        os.remove(plot_file)
                except Exception as e_cleanup:
                    logger.warning(f"Could not delete temporary report plot file '{plot_file}': {e_cleanup}")

    def _show_message(self, title: str, message: str, level: str = "info"):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        if level == "info": msg_box.setIcon(QMessageBox.Information)
        elif level == "warning": msg_box.setIcon(QMessageBox.Warning)
        elif level == "critical": msg_box.setIcon(QMessageBox.Critical)
        else: msg_box.setIcon(QMessageBox.NoIcon)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def set_data(self, data_frame: pd.DataFrame):
        if self.data_viewer_instance:
            self.data_viewer_instance.load_data(data_frame) 
            self.selected_features = []
            self.selected_target = None
            self.selected_features_label.setText("Features: Not selected (New data loaded)")
            self.selected_target_label.setText("Target: Not selected (New data loaded)")
            self.current_model_pipeline = None
            self.current_model_metrics = {}
            self._clear_evaluation_plots_display("New data loaded. Train a model.")
            self._update_prediction_inputs_ui([])
            self.save_model_button.setEnabled(False)
            self.export_report_button.setEnabled(False)
            if hasattr(self.parent_window, 'status_label'):
                self.parent_window.status_label.setText("New dataset loaded. Select features and target.")
            logger.info(f"ModelTrainer received new data, shape: {data_frame.shape}")
        else:
            logger.warning("ModelTrainer: data_viewer_instance is None. Cannot set data.")

class NpEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.integer):return int(obj)
        if isinstance(obj,np.floating):return float(obj)
        if isinstance(obj,np.ndarray):return {"__ndarray__":True,"values":obj.tolist(),"dtype":str(obj.dtype)}
        if isinstance(obj,(datetime.date,datetime.datetime)):return obj.isoformat()
        return super(NpEncoder,self).default(obj)

def decode_np_array(dct):
    if "__ndarray__" in dct and dct["__ndarray__"]:return np.array(dct["values"],dtype=dct.get("dtype","object"))
    return dct

class FeatureTargetDialog(QDialog):
    def __init__(self, all_columns:list, numeric_columns:list, current_features:list|None=None, current_target:str|None=None, parent=None):
        super().__init__(parent); self.setWindowTitle("Select Features (X) and Target (y)")
        if not ICON_SELECT_FEATURES.isNull(): self.setWindowIcon(ICON_SELECT_FEATURES)
        self.setMinimumWidth(480); self.setMinimumHeight(420); self.setObjectName("FeatureTargetDialog")
        self.all_columns_list = all_columns; self.numeric_columns_list = numeric_columns
        self.selected_features_list = current_features if current_features else []; self.selected_target_variable = current_target
        main_dialog_layout = QVBoxLayout(self); main_dialog_layout.setSpacing(12); main_dialog_layout.setContentsMargins(12,12,12,12)
        features_group = QGroupBox("Input Features (X)"); features_layout = QVBoxLayout(features_group); features_layout.setContentsMargins(10,5,10,10); features_layout.setSpacing(8)
        instruction_label_features = QLabel("<i>Select one or more input features for the model.<br/><b>Tip:</b> Use Ctrl+Click or Shift+Click for multiple selections in the list below.</i>"); instruction_label_features.setWordWrap(True); instruction_label_features.setStyleSheet("font-size: 10.5px; color: #444;"); features_layout.addWidget(instruction_label_features)
        self.features_listwidget = QListWidget(); self.features_listwidget.setSelectionMode(QAbstractItemView.ExtendedSelection); self.features_listwidget.addItems(self.all_columns_list)
        if self.selected_features_list:
            for ft in self.selected_features_list: items = self.features_listwidget.findItems(ft,Qt.MatchExactly); (items[0].setSelected(True) if items else None)
        features_layout.addWidget(self.features_listwidget); main_dialog_layout.addWidget(features_group)
        target_group = QGroupBox("Target Variable (y)"); target_layout = QVBoxLayout(target_group); target_layout.setContentsMargins(10,5,10,10); target_layout.addWidget(QLabel("<i>Select a single <b>numeric</b> column as the target variable for regression.</i>"))
        self.target_combobox = QComboBox(); self.target_combobox.addItem("-- Select Target --"); target_cands = self.numeric_columns_list if self.numeric_columns_list else self.all_columns_list; self.target_combobox.addItems(target_cands)
        if self.selected_target_variable and self.target_combobox.findText(self.selected_target_variable) >=0 : self.target_combobox.setCurrentText(self.selected_target_variable)
        target_layout.addWidget(self.target_combobox); main_dialog_layout.addWidget(target_group)
        self.dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel); self.dialog_buttons.accepted.connect(self.accept_selection); self.dialog_buttons.rejected.connect(self.reject); main_dialog_layout.addWidget(self.dialog_buttons)
        self._apply_dialog_stylesheet()
    def _apply_dialog_stylesheet(self): self.setStyleSheet("""#FeatureTargetDialog{background-color:#f8f9fa;font-family:Arial;} QGroupBox{font-weight:bold;margin-top:5px;padding:8px;border:1px solid #c8cfd6;border-radius:5px;background-color:#fff;} QGroupBox::title{left:10px;padding:0 4px 2px 4px;color:#005a9e;background-color:#fff;} QListWidget,QComboBox{border:1px solid #ced4da;border-radius:4px;padding:6px;min-height:28px;font-size:11px;background-color:white;} QListWidget{min-height:130px;} QLabel{margin-bottom:3px;font-size:11px;color:#333;} QPushButton{padding:7px 14px;min-width:75px;font-size:11px;background-color:#0078d4;color:white;border-radius:4px;} QPushButton:hover{background-color:#005a9e;}""")
    def accept_selection(self): self.selected_features_list=[item.text() for item in self.features_listwidget.selectedItems()]; self.selected_target_variable=self.target_combobox.currentText() if self.target_combobox.currentIndex()>0 else None; (QMessageBox.warning(self,"Input Error","Please select at least one input feature (X).") if not self.selected_features_list else (QMessageBox.warning(self,"Input Error","Please select a target variable (y).") if not self.selected_target_variable else (QMessageBox.warning(self,"Input Error","Target variable cannot also be an input feature.") if self.selected_target_variable in self.selected_features_list else self.accept())))
    def get_selections(self): return self.selected_features_list, self.selected_target_variable


if __name__ == '__main__':
    app = QApplication(sys.argv); logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    class DummyDataViewer:
        def __init__(self): 
            self.data = pd.DataFrame({
                'NumericA':np.random.rand(100)*10,
                'NumericB':np.random.randn(100)*5+20,
                'CategoryX':np.random.choice(['P','Q','R','S'],100),
                'CategoryY':np.random.choice(['Low','Medium','High','Very High'],100),
                'TargetNumeric':np.random.rand(100)*100+50,
                'DateColumn':pd.to_datetime(pd.date_range(start='2023-01-01',periods=100,freq='D'))
            })
            self.data.loc[self.data.sample(frac=0.05).index,'NumericA']=np.nan
            self.data.loc[self.data.sample(frac=0.03).index,'CategoryX']=np.nan
        def load_data(self,df_new): 
            self.data=df_new
            logger.info(f"DummyDataViewer: Data updated, shape {df_new.shape}")

    class DummyParentWindow(QWidget):
        def __init__(self): 
            super().__init__()
            self.status_label=QLabel("Main Window Status: Ready")
            layout=QVBoxLayout(self)
            layout.addWidget(self.status_label)
            self.setLayout(layout)
            self.setWindowTitle("Dummy Parent Window") 
            self.setGeometry(50, 50, 300, 100) 

    dummy_dv_instance = DummyDataViewer()
    dummy_pw_instance = DummyParentWindow() 

    trainer_widget = ModelTrainer(parent_window=dummy_pw_instance, data_viewer_instance=dummy_dv_instance)
    trainer_widget.setWindowTitle("ML Studio - Model Trainer") 
    trainer_widget.setGeometry(100, 100, 1200, 800) 
    trainer_widget.show()
    
    sys.exit(app.exec_())