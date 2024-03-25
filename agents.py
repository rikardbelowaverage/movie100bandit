import logging
import time
import json
import random
random.seed(42)
import pandas as pd
import numpy as np
import xgboost as xgb
from agent import Agent
from utils.utils_xgboost import get_leafs




class TETS(Agent):
    def __init__(self, use_cuda=False, epsilon=0.05,
                 exploration_factor=1,
                max_depth=6, n_estimators=100,
                eta=0.3, gamma=0, xgb_lambda=1.0):
        print("initializing agent")
        self.t = 1
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.observations = []
        self.rewards = []
        self.feature_types = ['q' for _ in range(1)]+['c' for _ in range(22)]
        self.lr = eta
        self.actions = []
        self.exploration_factor = exploration_factor
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

        leaf_scores = self.model.get_dump(with_stats=True, dump_format='json')
        
        #gets residuals for first tree which has constant prediction s
        residuals_array = np.array(y - 0.5, dtype="float64")
        
        json_trees = [json.loads(leaf_scores[i]) for i in range(self.n_estimators)]
        self.leaves_per_tree = []
        X = xgb.DMatrix(X, feature_types=self.feature_types, enable_categorical=True)

        # Iterate through all individual trees in the sample and collect the leavs
        for idx, tree in enumerate(json_trees):
            #print('len(tree)', len(tree))
            if not self.leaves_per_tree:  #always append leafs of first tree (even if only one leaf)
                self.leaves_per_tree.append(get_leafs(tree))
            elif 'leaf' in tree.keys():  #if first node is leaf, we don't have any more trees
                break
            else:
                self.leaves_per_tree.append(get_leafs(tree))
                """
                try:
                    print(get_leafs(tree).__sizeof__())
                    print(get_leafs(tree)[1])#.__sizeof__())
                except:
                    print("Exception: get_leafs(tree)")
                """
            pred = self.model.predict(X, iteration_range=(0,idx+1)).reshape((len(self.rewards),1))
            logging.info('Staged pred ' + str(idx) + ': ' + str(pred))
            logging.info('Target ' + str(idx) + ': ' + str(self.rewards))
            residuals_array = np.append(residuals_array, self.rewards - pred, axis=1)
        
        individual_preds = self.model.predict(X, pred_leaf=True)
        leaf_of_data = np.array(individual_preds)

        for i in range(len(self.leaves_per_tree)):
            df2 = pd.DataFrame({'leaf':leaf_of_data[:,i], 'residual':residuals_array[:,i]})
            group2 = df2.groupby(by='leaf').var()
        
            for row_idx, row in group2.iterrows():
                self.leaves_per_tree[i][row_idx]["row_idx"] = row_idx
                #print('row_idx:'+str(row_idx))
                self.leaves_per_tree[i][row_idx]["leaf_variance"] = row['residual'] * self.lr**2
                #print("leaf_variance:"+str(row['residual'] * self.lr**2))

        end = time.time()
        print("model trained in {:.3} seconds".format(end-start))
            
    def update_observations(self, observation, action, reward):
        self.observations.append(observation)
        self.rewards.append(reward)
        # add some if-statement to avoid training model at each iteration
        if self.t==9 or self.t%100==0:
            self.train_model(self.observations,self.rewards)
        self.t += 1

    def get_samples(self, X):
        samples = self.predict(xgb.DMatrix(X))
        return samples

    def predict(self, X):
        mu_hat = self.model.predict(X)
        individual_preds = self.model.predict(X, pred_leaf=True)
        leaf_assignments = np.array(individual_preds)
        n_samples_total = len(self.rewards)
        variance_of_mean_per_assigned_leaf = np.array([self.leaves_per_tree[i][leaf_assignments[0][i]]["leaf_variance"] / self.leaves_per_tree[i][leaf_assignments[0][i]]["cover"] for i in range(len(self.leaves_per_tree))])
        total_variance_of_mean = np.sum(variance_of_mean_per_assigned_leaf)

        return mu_hat + np.sqrt(total_variance_of_mean) * np.random.randn()     

    def pick_action(self, observations):
        arms, _ = observations.shape
        if (self.t<10) or (random.uniform(0,1)<(1/self.t)): #initial random pulls
            return random.randint(0,arms-1)
        else: # sample rewards for all possible actions, np.random.choice break ties if more than one option is considered optimal
            samples = self.get_samples(observations)
            print(samples)
            action = np.random.choice(np.flatnonzero(samples==samples.max()))
            self.actions.append(action)
            if self.t%9000==0:
                print(self.actions)
            return action

class XGBGreedy(Agent):
    def __init__(self, use_cuda=False, epsilon=0.05,
                max_depth=6, n_estimators=100,
                gamma=0, xgb_lambda=1.0, eta=0.3):
        print("initializing agent")
        self.t = 1
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.observations = []
        self.rewards = []
        self.feature_types = ['q' for _ in range(1)]+['c' for _ in range(22)]
        self.actions = []
        self.model_parameters = {'booster': 'gbtree', 'tree_method': 'hist', 
                                 'objective': 'reg:squarederror', 'max_depth': max_depth, 
                                 'gamma': gamma, 'learning_rate': eta, 
                                 'reg_lambda': xgb_lambda, 'min_child_weight': 2}
        Agent.__init__(self)
    def __str__(self):
        return "This is the XGBGreedy contextual multi-armed bandit"
    
    def train_model(self, observations, rewards):
        X = np.array(observations)
        y = np.array(rewards).reshape(len(rewards),1)
        Xy = xgb.DMatrix(X, y, feature_types=self.feature_types, enable_categorical=True)
        self.model = xgb.train(params = self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)
            
    def update_observations(self, observation, action, reward):
        self.observations.append(observation)
        self.rewards.append(reward)
        # add some if-statement to avoid training model at each iteration
        if self.t==9 or self.t%100==0:
            self.train_model(self.observations,self.rewards)
        self.t += 1

    def get_samples(self, X):
        samples = self.predict(xgb.DMatrix(X))
        return samples

    def predict(self, X):
        mu_hat = self.model.predict(X)
        return mu_hat

    def pick_action(self, observations):
        arms, _ = observations.shape
        if (self.t<10) or (random.uniform(0,1)<(1/self.t)): #initial random pulls
            return random.randint(0,arms-1)
        else: # sample rewards for all possible actions, np.random.choice break ties if more than one option is considered optimal
            samples = self.get_samples(observations)
            action = np.random.choice(np.flatnonzero(samples==samples.max()))
            self.actions.append(action)
            return action