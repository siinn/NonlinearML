from Asset_growth.lib.utils import *


class HeuristicModel_SortAG():
    ''' Class representing a simple rule-based model for predicting return class using only single factor.
    Attributes:
        df: input dataframe
        tertile_boundary: Boundary of tertile class
        feature: Name of feature. ex. "AG"
    '''
    def __init__(self, df, feature, month="eom"):
        # Initialize variables
        self.df = df.copy()
        self.feature = feature
        # get tertile boundary
        self.tertile_boundary =  get_tertile_boundary(df, [feature])
    def predict(self, df):
        ''' Make prediction based on pre-defined rule. Dataset defined in the initialization is used for prediction.'''
        # Get lower and upper bound
        ag_lb = self.tertile_boundary[self.feature][0]
        ag_ub = self.tertile_boundary[self.feature][1]
        # Determine class
        def get_class(x, lb, ub):
            if x < lb:
                return 0
            elif x < ub:
                return 1
            else:
                return 2
        return np.array(df[self.feature].apply(get_class, lb=ag_lb, ub=ag_ub))

class HeuristicModel_AG_HighFCFA():
    ''' Class representing a simple rule-based model for predicting return class using two factors.
    Attributes:
        tertile_boundary: Boundary of tertile class
        ag, fcfa: two features used in this model
    '''
    def __init__(self, df, ag, fcfa, month="eom"):
        # Initialize variables and get tertile boundary
        self.tertile_boundary =  get_tertile_boundary(df, [ag, fcfa])
        self.ag = ag
        self.fcfa = fcfa
        self.ag_tertile = ag+"_tertile"
        self.fcfa_tertile = fcfa+"_tertile"
    def predict(self, df):
        ''' Make prediction based on pre-defined rule.'''
        # Get lower and upper bound
        ag_lb = self.tertile_boundary[self.ag][0]
        ag_ub = self.tertile_boundary[self.ag][1]
        fcfa_lb = self.tertile_boundary[self.fcfa][0]
        fcfa_ub = self.tertile_boundary[self.fcfa][1]
        # Determine class
        def get_class(x, ag_lb, ag_ub, fcfa_lb, fcfa_ub):
            if (x[self.ag] < ag_lb) & (x[self.fcfa] < fcfa_ub):
                return 0
            elif (x[self.ag] > ag_ub) & (x[self.fcfa] < fcfa_ub):
                return 2
            else:
                return 1
        return np.array(df[[self.fcfa, self.ag]].apply(get_class, ag_lb=ag_lb, ag_ub=ag_ub, fcfa_lb=fcfa_lb, fcfa_ub=fcfa_ub, axis=1))

    def predict_exact(self, df):
        ''' Make prediction using exact tertile label.'''
        # Determine class
        def get_class(x):
            if (x[self.ag_tertile] == 0) & (x[self.fcfa_tertile] != 0):
                return 0
            elif (x[self.ag_tertile] == 2) & (x[self.fcfa_tertile] != 0):
                return 2
            else:
                return 1
        df["pred"] = df[[self.fcfa_tertile, self.ag_tertile]].apply(get_class, axis=1)
        return df
