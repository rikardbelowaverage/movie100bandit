import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,train_test_split
from agent import Agent
import random
random.seed(42)
import time

class TETS(Agent):
    def __init__(self, use_cuda=False, epsilon=0.05,
                max_depth=6, n_estimators=100,
                eta=0.3, gamma=0, xgb_lambda=1.0):
        print("initializing agent")
        self.t = 1
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.observations = []
        self.rewards = []
        self.feature_types = ['q' for _ in range(1)]+['c' for _ in range(22)]

        self.model_parameters = {'booster': 'gbtree', 'tree_method': 'hist', 
                                 'objective': 'reg:squarederror', 'max_depth': max_depth, 
                                 'gamma': gamma, 'learning_rate': eta, 
                                 'reg_lambda': xgb_lambda, 'min_child_weight': 2}
        Agent.__init__(self)
    def __str__(self):
        return "This is the TETS contextual multi-armed bandit"
    
    def train_model(self, observations, rewards):
        #print("training model.....")
        X = np.array(observations)
        y = np.array(rewards).reshape(len(rewards),1)
        #print("Observations shape : {} rewards shape : {}".format(X.shape,y.shape))
        start = time.time()
        Xy = xgb.DMatrix(X, y, feature_types=self.feature_types, enable_categorical=True)
        self.model = xgb.train(params = self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)
        end = time.time()
        #print("model trained in {:.3} seconds".format(end-start))
        
    def update_observations(self, observation, action, reward):
        self.observations.append(observation)
        self.rewards.append(reward)
        # add some if-statement to avoid training model at each iteration
        if self.t==9 or self.t%100==0:
            self.train_model(self.observations,self.rewards)
        self.t += 1

    def get_samples(self, X):
        samples = self.model.predict(xgb.DMatrix(X))
        return samples

    def pick_action(self, observations):
        arms, _ = observations.shape
        if (self.t<10) or (random.uniform(0,1)<(1/self.t)):
            return random.randint(0,arms-1)
        else:
            return np.random.choice(np.flatnonzero(self.get_samples(observations)==self.get_samples(observations).max()))

