from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluates a regression model's performance.
    Returns a dictionary containing R-squared (r2), Mean Squared Error (mse),
    Root Mean Squared Error (rmse), and Mean Absolute Error (mae).
    """
    if not isinstance(y_true, (np.ndarray, pd.Series)) or not isinstance(y_pred, (np.ndarray, pd.Series)):
        logger.error("y_true and y_pred must be numpy arrays or pandas Series.")
        return {'r2': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan}

    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning("Empty true or predicted values passed for evaluation. Returning NaN metrics.")
        return {'r2': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan}
    if len(y_true) != len(y_pred):
        logger.error("Length of y_true and y_pred must be the same for evaluation.")
        return {'r2': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan}

    try:
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        logger.info(f"Model Evaluation: R2={r2:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        return {'r2': r2, 'mse': mse, 'rmse': rmse, 'mae': mae}
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        return {'r2': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mae': np.nan}


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple | None:
    """
    Splits features (X) and target (y) into training and testing sets.
    Returns (X_train, X_test, y_train, y_test) or None on error.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        logger.error("X must be a pandas DataFrame and y must be a pandas Series.")
        return None
    if len(X) != len(y):
        logger.error("X and y must have the same number of samples.")
        return None
    if len(X) == 0:
        logger.error("Input data (X and y) is empty, cannot split.")
        return None
    if not (0 < test_size < 1):
        logger.error(f"test_size must be between 0 and 1 (exclusive). Got {test_size}")
        return None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}", exc_info=True)
        return None


def train_linear_regression(fit_intercept: bool = True, **kwargs) -> LinearRegression:
    logger.debug(f"Instantiating LinearRegression with fit_intercept={fit_intercept}, kwargs={kwargs}")
    return LinearRegression(fit_intercept=fit_intercept, **kwargs)

def train_ridge_regression(alpha: float = 1.0, fit_intercept: bool = True, random_state: int | None = None, **kwargs) -> Ridge:
    logger.debug(f"Instantiating Ridge with alpha={alpha}, fit_intercept={fit_intercept}, random_state={random_state}, kwargs={kwargs}")
    kwargs.pop('n_jobs', None)
    return Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=random_state, **kwargs)

def train_lasso_regression(alpha: float = 1.0, fit_intercept: bool = True, random_state: int | None = None, **kwargs) -> Lasso:
    logger.debug(f"Instantiating Lasso with alpha={alpha}, fit_intercept={fit_intercept}, random_state={random_state}, kwargs={kwargs}")
    kwargs.pop('n_jobs', None)
    return Lasso(alpha=alpha, fit_intercept=fit_intercept, random_state=random_state, **kwargs)

def train_elasticnet_regression(alpha: float = 1.0, l1_ratio: float = 0.5, fit_intercept: bool = True, random_state: int | None = None, **kwargs) -> ElasticNet:
    logger.debug(f"Instantiating ElasticNet with alpha={alpha}, l1_ratio={l1_ratio}, fit_intercept={fit_intercept}, random_state={random_state}, kwargs={kwargs}")
    kwargs.pop('n_jobs', None)
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, random_state=random_state, **kwargs)

def train_bayesian_ridge_regression(alpha_1: float = 1e-6, alpha_2: float = 1e-6, lambda_1: float = 1e-6, lambda_2: float = 1e-6, fit_intercept: bool = True, **kwargs) -> BayesianRidge:
    logger.debug(f"Instantiating BayesianRidge with alpha_1={alpha_1}, alpha_2={alpha_2}, lambda_1={lambda_1}, lambda_2={lambda_2}, kwargs={kwargs}")
    kwargs.pop('n_jobs', None)
    return BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, fit_intercept=fit_intercept, **kwargs)


def train_decision_tree_regressor(
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42,
    **kwargs
) -> DecisionTreeRegressor:
    logger.debug(f"Instantiating DecisionTreeRegressor with max_depth={max_depth}, min_samples_split={min_samples_split}, random_state={kwargs.get('random_state', random_state)}, other_kwargs={kwargs}")
    kwargs.pop('n_jobs', None)
    return DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=kwargs.pop('random_state', random_state),
        **kwargs
    )

def train_random_forest_regressor(
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> RandomForestRegressor:
    logger.debug(f"Instantiating RandomForestRegressor with n_estimators={n_estimators}, max_depth={max_depth}, random_state={kwargs.get('random_state', random_state)}, n_jobs={n_jobs}, other_kwargs={kwargs}")
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=kwargs.pop('random_state', random_state),
        n_jobs=n_jobs,
        **kwargs
    )

def train_xgboost_regressor(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 42,
    n_jobs: int = -1,
    objective: str = 'reg:squarederror',
    **kwargs
) -> xgb.XGBRegressor:
    logger.debug(f"Instantiating XGBRegressor with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, objective={objective}, random_state={kwargs.get('random_state', random_state)}, n_jobs={n_jobs}, other_kwargs={kwargs}")
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        objective=objective,
        random_state=kwargs.pop('random_state', random_state),
        n_jobs=n_jobs,
        **kwargs
    )

def train_lightgbm_regressor(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    max_depth: int = -1,
    random_state: int | None = None,
    n_jobs: int = -1,
    **kwargs
) -> lgb.LGBMRegressor:
    logger.debug(f"Instantiating LGBMRegressor with n_estimators={n_estimators}, learning_rate={learning_rate}, num_leaves={num_leaves}, max_depth={max_depth}, random_state={random_state}, n_jobs={n_jobs}, kwargs={kwargs}")
    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )


def train_svr(kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1, **kwargs) -> SVR:
    logger.debug(f"Instantiating SVR with kernel={kernel}, C={C}, epsilon={epsilon}, kwargs={kwargs}")
    kwargs.pop('n_jobs', None)
    return SVR(kernel=kernel, C=C, epsilon=epsilon, **kwargs)

def train_kneighbors_regressor(
    n_neighbors: int = 5,
    weights: str = 'uniform',
    algorithm: str = 'auto',
    leaf_size: int = 30,
    p: int = 2,
    n_jobs: int = -1,
    **kwargs
) -> KNeighborsRegressor:
    logger.debug(f"Instantiating KNeighborsRegressor with n_neighbors={n_neighbors}, weights='{weights}', algorithm='{algorithm}', n_jobs={n_jobs}, kwargs={kwargs}")
    return KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        n_jobs=n_jobs,
        **kwargs
    )


def perform_grid_search(
    estimator,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = 'r2',
    n_jobs: int = -1
) -> tuple[object, dict, float] | None:
    """
    Performs GridSearchCV to find the best hyperparameters for an estimator.
    Returns (best_estimator, best_params, best_score) or None on error.
    """
    logger.info(f"Performing GridSearchCV for {estimator.__class__.__name__} with param_grid: {param_grid}, scoring: {scoring}")
    try:
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = float(grid_search.best_score_)

        logger.info(f"GridSearchCV complete. Best parameters: {best_params}")
        logger.info(f"Best cross-validated score ({scoring}): {best_score:.4f}")

        return best_estimator, best_params, best_score
    except Exception as e:
        logger.error(f"Error during GridSearchCV for {estimator.__class__.__name__}: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    rng = np.random.RandomState(42)
    X_dummy = pd.DataFrame(rng.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    y_dummy = pd.Series(
        2 * X_dummy['feature_0'] -
        1.5 * X_dummy['feature_1']**2 +
        0.5 * X_dummy['feature_2'] * X_dummy['feature_3'] +
        rng.randn(100) * 0.5 + 5
    )
    split_result = split_data(X_dummy, y_dummy)
    if split_result:
        X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = split_result

        logger.info("\n--- Ridge Regression Example ---")
        ridge_model = train_ridge_regression(alpha=0.5)
        ridge_model.fit(X_train_dummy, y_train_dummy)
        y_pred_ridge = ridge_model.predict(X_test_dummy)
        eval_ridge = evaluate_regression_model(y_test_dummy, y_pred_ridge)
        print(f"Ridge Eval: {eval_ridge}")

        logger.info("\n--- KNN Regressor Example ---")
        knn_model = train_kneighbors_regressor(n_neighbors=3)
        knn_model.fit(X_train_dummy, y_train_dummy)
        y_pred_knn = knn_model.predict(X_test_dummy)
        eval_knn = evaluate_regression_model(y_test_dummy, y_pred_knn)
        print(f"KNN Eval: {eval_knn}")

        logger.info("\n--- LightGBM Regressor Example ---")
        try:
            lgbm_model = train_lightgbm_regressor(n_estimators=50, random_state=42, n_jobs=1)
            lgbm_model.fit(X_train_dummy, y_train_dummy)
            y_pred_lgbm = lgbm_model.predict(X_test_dummy)
            eval_lgbm = evaluate_regression_model(y_test_dummy, y_pred_lgbm)
            print(f"LGBM Eval: {eval_lgbm}")
        except Exception as e_lgbm:
             print(f"Could not run LightGBM example: {e_lgbm}")
    logger.info("\n--- Script execution finished ---")