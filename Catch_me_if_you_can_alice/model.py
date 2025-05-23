import polars as pl
import polars.selectors as cs
import polars_ds as ds
import polars_xdt as xdt
import pickle
from datetime import date
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


def model_pipeline(X_train, y_train, modeltype: str):
    if modeltype == "xgboost":
        model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="auc",
            scale_pos_weight=1,
            random_state=42,
        )

        param_grid = {
            "max_depth": [3, 6],
            "learning_rate": [0.1, 0.01],
            "n_estimators": [100, 300],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    elif modeltype == "catboost":
        model = CatBoostClassifier(
            auto_class_weights="Balanced", verbose=0, random_seed=42, eval_metric="AUC"
        )

        param_grid = {
            "depth": [6, 10],
            "learning_rate": [0.1, 0.01],
            "iterations": [100, 300],
            "l2_leaf_reg": [1, 3],
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train.to_pandas(), y_train.to_pandas())
        return grid_search.best_estimator_

    else:
        raise ValueError(f"Modelltypen '{modeltype}' støttes ikke.")
