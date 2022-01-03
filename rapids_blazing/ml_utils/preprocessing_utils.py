from sklearn.base import TransformerMixin, BaseEstimator


class SubsetTransformer(BaseEstimator, TransformerMixin):
    """An sklearn-compatible class for fitting and transforming on
    a subset of features
    
    This allows a transformation to be applied to only data in a
    specific column of a dataframe or only data of a particular dtype.
    
    Credit: https://gist.github.com/wphicks/6233758c1d83014f54891ba617300b48
    """
    def __init__(self,
                 transformer,
                 include_dtypes=None,
                 exclude_dtypes=None,
                 columns=None,
                 copy=True):
        self.transformer = transformer
        self.include_dtypes = include_dtypes
        self.exclude_dtypes = exclude_dtypes
        self.columns = columns
        self.copy = copy
        
    def _get_subset(self, X):
        subset = X
        if self.columns is not None:
            subset = X[self.columns]
        if self.include_dtypes or self.exclude_dtypes:
            subset = subset.select_dtypes(
                include=self.include_dtypes,
                exclude=self.exclude_dtypes
            )
        return subset
        
    def fit(self, X, y=None):
        subset = self._get_subset(X)
        try:
            self.transformer.fit(subset, y=y)
        except TypeError:  # https://github.com/rapidsai/cuml/issues/3053
            self.transformer.fit(subset)
        return self
    
    def transform(self, X, y=None):
        if self.copy:
            X = X.copy()
        subset = self._get_subset(X)
        try:
            X[subset.columns] = self.transformer.transform(subset, y=y)
        except TypeError:  # https://github.com/rapidsai/cuml/issues/3053
            X[subset.columns] = self.transformer.transform(subset)
        
        return X
    
    def fit_transform(self, X, y=None):
        if self.copy:
            X = X.copy()
        subset = self._get_subset(X)
        try:
            X[subset.columns] = self.transformer.fit_transform(subset, y=y)
        except TypeError:  # https://github.com/rapidsai/cuml/issues/3053
            X[subset.columns] = self.transformer.fit_transform(subset)
        
        return X
    
    

class PerFeatureTransformer(BaseEstimator, TransformerMixin):
    """An sklearn-compatible class for fitting and transforming on
    each feature independently
    
    Some preprocessing algorithms need to be applied independently to
    each feature. This wrapper facilitates that process.
    
    Credit: https://gist.github.com/wphicks/6233758c1d83014f54891ba617300b48
    """
    def __init__(self,
                 transformer_class,
                 transformer_args=(),
                 transformer_kwargs={},
                 copy=True):
        self.transformer_class = transformer_class
        self.transformer_args = transformer_args
        self.transformer_kwargs = transformer_kwargs
        self.transformers = {}
        self.copy = copy
        
    def fit(self, X, y=None):
        for col in X.columns:
            self.transformers[col] = self.transformer_class(
                *self.transformer_args,
                **self.transformer_kwargs
            )
            try:
                self.transformers[col].fit(X[col], y=y)
            except TypeError:  # https://github.com/rapidsai/cuml/issues/3053
                self.transformers[col].fit(X[col])
        return self
    
    def transform(self, X, y=None):
        if self.copy:
            X = X.copy()
        for col in X.columns:
            try:
                X[col] = self.transformers[col].transform(X[col], y=y)
            except TypeError:  # https://github.com/rapidsai/cuml/issues/3053
                X[col] = self.transformers[col].transform(X[col])
            
        return X
    
    def fit_transform(self, X, y=None):
        for col in X.columns:
            self.transformers[col] = self.transformer_class(
                *self.transformer_args,
                **self.transformer_kwargs
            )
            try:
                X[col] = self.transformers[col].fit_transform(X[col], y=y)
            except TypeError:  # https://github.com/rapidsai/cuml/issues/3053
                X[col] = self.transformers[col].fit_transform(X[col])
        return X