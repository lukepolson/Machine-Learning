import pandas as pd
import numpy as np
from hf import *
import lightgbm as lgb
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn import impute
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb
import warnings


## ALL CLASSES

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_hourday = True):
        self.add_hourday = add_hourday
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.add_hourday:
            return X.assign(tm_day_of_week=((X.timestamp // 24) % 7), tm_hour_of_day=(X.timestamp % 24))
        else:
            return X
        
class CatSplitRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, col):
        self.model = model
        self.col = col

    def fit(self, X, y):
        self.fitted = {}
        importances = []
        for val in X[self.col].unique():
            X1 = X[X[self.col] == val].drop(columns=[self.col])
            self.fitted[val] = clone(self.model).fit(X1, y.reindex_like(X1))
            importances.append(self.fitted[val].feature_importances_)
            del X1
        fi = np.average(importances, axis=0)
        col_index = list(X.columns).index(self.col)
        self.feature_importances_ = [*fi[:col_index], 0, *fi[col_index:]]
        return self
    
    def transform(self, X):
        return X

    def predict(self, X):
        result = np.zeros(len(X))
        for val in X[self.col].unique():
            ix = np.nonzero((X[self.col] == val).to_numpy())
            predictions = self.fitted[val].predict(X.iloc[ix].drop(columns=[self.col]))
            result[ix] = predictions
        return result
    
categorical_columns = [
    "building_id", "meter", "site_id", "primary_use", "had_air_temperature", "had_cloud_coverage",
    "had_dew_temperature", "had_precip_depth_1_hr", "had_sea_level_pressure", "had_wind_direction",
    "had_wind_speed", "tm_day_of_week", "tm_hour_of_day"
]


class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, categorical_feature=None, **params):
        self.model = lgb.LGBMRegressor(**params)
        self.categorical_feature = categorical_feature

    def fit(self, X, y):
        with warnings.catch_warnings():
            cats = None if self.categorical_feature is None else list(
                X.columns.intersection(self.categorical_feature))
            warnings.filterwarnings("ignore",
                                    "categorical_feature in Dataset is overridden".lower())
            self.model.fit(X, y, **({} if cats is None else {"categorical_feature": cats}))
            self.feature_importances_ = self.model.feature_importances_
            return self
        
    def transform(self, X):
        return X

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {**self.model.get_params(deep), "categorical_feature": self.categorical_feature}

    def set_params(self, **params):
        ctf = params.pop("categorical_feature", None)
        if ctf is not None: self.categorical_feature = ctf
        self.model.set_params(**params)
        
