import numpy as np
import xgboost as xgb

#-------------------------------------------------------------------------------
# Custom objective function
#-------------------------------------------------------------------------------

def mse(y_true, y_pred):
    """ Standard MSE to validate other custom metrics"""
    #eps = 1e-16
    grad = y_pred - y_true
    #hess = np.maximum(y_pred *  (1-y_pred), eps)
    hess = [1]*len(y_true)
    return grad, hess

def squared_log(y_true, y_pred):
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    y_pred[y_pred < -1] = -1 + 1e-6
    grad = (np.log1p(y_pred) - np.log1p(y_true)) / (y_pred + 1)
    hess = ((-np.log1p(y_pred) + np.log1p(y_true) + 1) / np.power(y_pred + 1, 2))
    return grad, hess    

def log_square_error(y_true, y_pred):
    """ Log square error with an offset of 1.
        LSE = ln((y_pred - y_true)^2 + 1)"""
    sqer = np.power(y_pred-y_true, 2)
    grad = 2*(y_pred-y_true) / (sqer+1)
    hess = (-2 * (sqer-1)) / np.power(sqer+1, 2)
    return grad, hess



def mape(y_true, y_pred):
    """ Mean absolute percentage error.
        MAPE = abs(y_pred - y_true) / max(abs(y_true) + epsilon)"""
    eps = 1e-7
    grad = (100 * np.sign(y_pred - y_true) / np.maximum(abs(y_true), eps))
    hess = y_true*0
    return grad, hess

def log_abs_error(y_true, y_pred):
    """ Log absolute error with an offset of 1.
        LAE = ln(abs(y_pred - y_true) +1)"""
    grad = np.sign(y_pred - y_true) / (abs(y_pred-y_true) + 1)
    hess = -1 / pow(abs(y_pred - y_true) + 1, 2)
    return grad, hess

