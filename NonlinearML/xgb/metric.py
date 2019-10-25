import numpy as np
import xgboost as xgb

#-------------------------------------------------------------------------------
# Custom metric function
#-------------------------------------------------------------------------------
def mse(y_pred, y_true):
    """ Regular MSE to validate other custom metrics"""
    return 'custom_mse', 1/2*pow(y_pred-y_true,2)

def mape(y_pred, y_true):
    """ Mean absolute percentage error.
        MAPE = abs(y_pred - y_true) / max(abs(y_true) + epsilon)"""
    epsilon = 1e-7
    return 'PyMAPE', (100 * abs(y_true - y_pred) / np.maximum(abs(y_true), epsilon)).mean()

def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix):
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))

def log_abs_error(y_pred, y_true):
    """ Log absolute error with an offset of 1.
        LAE = ln(abs(y_pred - y_true) +1)"""
    return 'custom_lae', np.log1p(abs(y_pred - y_true))



def log_square_error(y_pred, y_true):
    """ Log square error with an offset of 1.
        LSE = ln((y_pred - y_true)^2 + 1)"""
    return 'custom_lse', np.log1p(pow(y_pred-y_true,2))







