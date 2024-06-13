# Third-party libraries
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from utils.popphy_io import get_stat_dict, get_stat
from sklearn.metrics import roc_curve

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class AdaBoost():
    def __init__(self, config, n_estimators=2000, learning_rate=0.01):
        self.num_trees = int(config.get('Benchmark', 'NumberTrees'))

        self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=1000, learning_rate=learning_rate)

        self.num_valid_models = int(config.get('Benchmark', 'ValidationModels'))
        self.feature_importance = []
        self.features = []
        
    def train(self, train, seed=42):
        x, y = train
        self.model.fit(x, y)

        return

    def test(self, test):
        x, y = test

        y_pred = self.model.predict(x)

        y_state = np.eye(2)[y]
        y_pred_state = np.eye(2)[y_pred]

        stat= get_stat_dict(y_state, y_pred_state)


        return y_pred, stat
