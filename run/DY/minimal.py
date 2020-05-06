# Import xgboost
""" You need xgboost 0.90 version. Please specify version when you install the
library.
    pip install xgboost==0.90
"""
from xgboost.sklearn import XGBRegressor

# Load model
MODEL_PATH = "/mnt/mainblob/nonlinearML/NonlinearML/model/DY/DY_dmed_PO_dmed/v1/enhanced_DY_xgboost.v1"
model = XGBRegressor()
model.load_model(MODEL_PATH)

# Import factors
""" Model expects DY and PO features in (m x 2) matrix where each row
represents samples, and two columns represent DY and PO. The order of columns
must be DY, PO.

You can also use Pandas dataframe or numpy array."""
# Dummy for testing purpose: 3 samples.
features = [
        [0.1, 0.2],
        [0.3, 0.5],
        [0.9, 1.4]]

# Instruction for preprocessing DY and PO factors
""" For each DY and PO, 
1. Winzorize the factor to 3% and 97% within MSCIEM.
2. Calculate median and standard devation within MSCIEM.
3. Standardize (demedian) the factor using median and std calculated above.
4. Winsorzie the factor between -3 and 3.
"""

# Make inference
""" Make prediction to generate Enhanced dividend yield factor.
'validate_features' checks if input feature names match with feature names used
in training. We need to disable this when we use array as input."""
EDY = model.predict(features, validate_features=False)





