import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pdb
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s -%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeekdayImputer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='dteday', weekday_column = 'weekday'):
        self.date_column = date_column
        self.weekday_column = weekday_column

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        weeks=X[X[self.weekday_column].isnull()].index
        X.loc[weeks,self.weekday_column] = X.loc[weeks,self.date_column].dt.day_name().str[:3]
        X.drop(self.date_column, axis=1, inplace=True)
        return X
    

class WeathersitImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column="weathersit"):
        self.column = column
        self.most_frequent = None

    def fit(self, X, y=None):
        self.most_frequent = X[self.column].mode()[0]  
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].fillna(self.most_frequent)
        return X


class Mapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables:str, mappings:dict):
        self.mappings = mappings
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X


class CustomOutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, variable:str):
        if not isinstance(variable, str):
            raise ValueError("variables should be a string")
        self.variable = variable


    def fit(self, X: pd.DataFrame, y: pd.Series = None):
    
        q1 = X.describe()[self.variable].loc['25%']
        q3 = X.describe()[self.variable].loc['75%']
        iqr = q3 - q1
        self.lower_bound = q1 - (1.5 * iqr)
        self.upper_bound = q3 + (1.5 * iqr)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for i in X.index:
            if X.loc[i,self.variable] > self.upper_bound:
                X.loc[i,self.variable]= self.upper_bound
            if X.loc[i,self.variable] < self.lower_bound:
                X.loc[i,self.variable]= self.lower_bound
        return X

  
# Not being used
# Changed this with mapper
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column="weekday"):
        """
        One-hot encodes the specified weekday column.

        :param column: Name of the column to encode (default is "weekday").
        """
        self.column = column
        self.unique_values = None

    def fit(self, X, y=None):
        """Identify unique weekday values."""
        self.unique_values = sorted(X[self.column].dropna().unique())  # Get unique weekday values
        return self

    def transform(self, X):
        """Perform one-hot encoding on the weekday column."""
        X = X.copy()
        
        # One-hot encode weekday column
        one_hot_encoded = pd.get_dummies(X[self.column], prefix=self.column)

        # Drop original column and merge one-hot columns
        X = X.drop(columns=[self.column])  # ✅ Make sure to drop the original column
        X = X.join(one_hot_encoded)

        return X
