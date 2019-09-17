from datetime import datetime
from sklearn.linear_model import Ridge
import pandas as pd

import NonlinearML.lib.utils as utils

#-------------------------------------------------------------------------------
# Model class
#-------------------------------------------------------------------------------
class LinearRank:
    """ Linear regression + ranking model. 
    Attributes:
        model: tf.keras.Sequential model
        params: MOdel parameters given in dictionary.
    """
    def __init__(self, n_classes, class_names):
        """ Initialize variables."""
        print("Building model..")
        # parameter set
        self.model = Ridge()
        self.n_classes = n_classes
        self.class_names = class_names

    def set_params(self, **params):
        """ Set model parameters. """
        self.model.set_params(**params)
        return self

    def fit(self, X, y):
        """ Train model."""
        self.model.fit(X, y)
        return self

    def predict(self, X, date):
        """ Make prediction."""
        pred = self.model.predict(X)
        if type(date) == pd.Series:
            df = pd.DataFrame(date).set_index(date.name)
            df['pred'] = pred
            df = utils.discretize_variables_by_month(
                df=df,
                variables=['pred'],
                n_classes=self.n_classes,
                class_names=self.class_names,
                suffix="discrete", month=date.name)
        elif date==None:
            df = pd.DataFrame(pred)
            df['pred_discrete'] = df.transform(
                lambda x: pd.qcut(x, self.n_classes, self.class_names))
        return df['pred_discrete'].values

