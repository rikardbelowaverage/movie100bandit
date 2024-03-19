import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from agent import Agent
import random
random.seed(42)

class TETS(Agent):
    def __init__(self, context_size,
                use_cuda=False, epsilon=0.05,
                max_depth=6, n_estimators=100,
                eta=0.3, gamma=100.0, xgb_lambda=1.0):
        print("initializing agent")
        self.t = 1
        self.context_size = context_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.observations = []
        self.rewards = []
        self.feature_types = ['c' for _ in range(25)]

        self.model_parameters = {'booster': 'gbtree', 'tree_method': 'hist', 'objective': 'reg:squarederror', 'max_depth': max_depth, 'gamma': gamma, 'learning_rate': eta, 'reg_lambda': xgb_lambda, 'min_child_weight': 2}

        Agent.__init__(self)

    def train_model(self, observations, rewards):
        print("fitting model")
        print(observations)
        print(rewards)
        X = np.array(observations)
        y = np.array(rewards).reshape(len(rewards),1)
        print(X)
        print(y)
        print(X.shape)
        print(y.shape)
        print(self.feature_types)
        print(len(self.feature_types))
        print(X[0].shape)
        Xy = xgb.DMatrix(X[0].reshape(len(rewards),26), y, feature_types=self.feature_types, enable_categorical=True)
        #self.model = xgb.train(params = self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)
        print("model fitted")
        
    def update_observations(self, observation, action, reward):
        print("updating observations")
        self.observations.append(observation)
        self.rewards.append(reward)
        # add some if-statement to avoid training model at each iteration
        self.train_model(self.observations,self.rewards)

    def predict(self, X):
        print("predicting outputs")
        pass

    def pick_action(self, observations):
        print("selecting actions")
        if self.t<10:
            arms, features = observations.shape
            print(arms,features)
            return random.randint(0,arms-1)
        else:
            return 1

