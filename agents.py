import time
import json
import random
random.seed(42)
import pandas as pd
import numpy as np
import xgboost as xgb
from agent import Agent
from utils.utils_xgboost import get_leafs
from utils.setup_logger import logger

class TETS(Agent):
    def __init__(self, feature_types, use_cuda=False, epsilon=0.05,
                 exploration_factor=1,
                max_depth=6, n_estimators=100,
                eta=0.03, gamma=0, xgb_lambda=0):
        logger.info("initializing agent")
        self.t = 1
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.observation_history = []
        self.rewards = []
        self.feature_types = feature_types
        logger.debug("self.feature_types="+str(self.feature_types))
        logger.debug("self.feature_types length="+str(len(self.feature_types)))
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
        self.X = np.array(observations)
        self.y = np.array(rewards).reshape(len(rewards),1)
        logger.debug("Observations shape : {} rewards shape : {}".format(self.X.shape,self.y.shape))
        start = time.time()
        Xy = xgb.DMatrix(self.X, self.y, feature_types=self.feature_types, enable_categorical=True)
        self.model = xgb.train(params = self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)

        leaf_scores = self.model.get_dump(with_stats=True, dump_format='json')
        
        #gets residuals for first tree which has constant prediction some constant value
        residuals_array = np.array(self.y - 0.5, dtype="float64")
        
        json_trees = [json.loads(leaf_scores[i]) for i in range(self.n_estimators)]
        self.leaves_per_tree = []
        self.X_d = xgb.DMatrix(self.X, feature_types=self.feature_types, enable_categorical=True)

        # Iterate through all individual trees in the sample and collect the leavs
        for idx, tree in enumerate(json_trees):
            #print('len(tree)', len(tree))
            if not self.leaves_per_tree:  #always append leafs of first tree (even if only one leaf)
                self.leaves_per_tree.append(get_leafs(tree))
            elif 'leaf' in tree.keys():  #if first node is leaf, we don't have any more trees
                break
            else:
                self.leaves_per_tree.append(get_leafs(tree))
            pred = self.model.predict(self.X_d, iteration_range=(0,idx+1)).reshape((len(self.rewards),1))
            logger.debug('Staged pred ' + str(idx) + ': ' + str(pred))
            logger.debug('Target ' + str(idx) + ': ' + str(self.rewards))
            residuals_array = np.append(residuals_array, self.rewards - pred, axis=1)
        
        individual_preds = self.model.predict(self.X_d, pred_leaf=True)
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
        logger.info("model trained in {:.3} seconds in timestep {}".format((end-start),self.t))
            
    def update_observations(self, observation, action, reward):
        logger.debug("Observation is"+str(observation))
        logger.debug('Reward is'+str(reward))
        self.observation_history.append(observation)
        self.rewards.append(reward)
        # add some if-statement to avoid training model at each iteration
        if (self.t==14 or self.t%100==0) and self.t<5000:
            self.train_model(self.observation_history,self.rewards)
        # TODO dump array of observations and rewards.
        if self.t==25000:
            np.save('X.dat', self.X)
            np.save('y.dat',self.y)
        self.t += 1

    def get_samples(self, X):
        samples = self.predict(xgb.DMatrix(X))
        return samples

    def predict(self, X):
        mu_hat = self.model.predict(X)
        individual_preds = self.model.predict(X, pred_leaf=True)
        leaf_assignments = np.array(individual_preds)

        logger.debug('Individual predictions shape'+str(individual_preds.shape))
        logger.debug('Leaf assignments'+str(leaf_assignments))
        logger.debug(str(np.shape(leaf_assignments)))
        n_samples_total = len(self.rewards)
        logger.debug(str(self.leaves_per_tree))
        logger.debug(str(np.shape(self.leaves_per_tree)))
        logger.debug(type(self.leaves_per_tree))

        variance_of_mean_per_assigned_leaf = [np.array([self.leaves_per_tree[i][leaf_assignments[j,i]].get('leaf_variance') 
                                  for j in range(len(leaf_assignments))]) for i in range(len(self.leaves_per_tree))]
        
        covers_of_mean_per_assigned_leaf = [np.array([self.leaves_per_tree[i][leaf_assignments[j,i]].get('cover') 
                                  for j in range(len(leaf_assignments))]) for i in range(len(self.leaves_per_tree))]

        logger.debug('Variance term:'+str(np.sqrt(np.array(variance_of_mean_per_assigned_leaf)/np.array(covers_of_mean_per_assigned_leaf))))
        logger.debug('Variance:'+str(np.array(variance_of_mean_per_assigned_leaf)))
        logger.debug('Covers:'+str(np.array(covers_of_mean_per_assigned_leaf)))
        logger.debug('Variance shape'+str(np.array(variance_of_mean_per_assigned_leaf).shape))
        logger.debug('Covers shape'+str(np.array(covers_of_mean_per_assigned_leaf).shape))
        logger.debug('mu_hat shape'+str(mu_hat.shape))
        logger.debug('predict'+str(mu_hat + np.sqrt(np.sum(np.array(variance_of_mean_per_assigned_leaf)/np.array(covers_of_mean_per_assigned_leaf),axis=0).T)*np.random.randn(mu_hat.shape[0])))
        return mu_hat + np.sqrt(np.sum(np.array(variance_of_mean_per_assigned_leaf)/np.array(covers_of_mean_per_assigned_leaf),axis=0).T)*np.random.randn(mu_hat.shape[0])      

    def pick_action(self, observations):
        arms, _ = observations.shape
        if (self.t<15): #initial random pulls
            return random.randint(0,arms-1)
        else: # sample rewards for all possible actions, np.random.choice break ties if more than one option is considered optimal
            logger.debug('observation shape'+str(observations.shape))
            samples = self.get_samples(observations)
            
            logger.debug('samples shape:'+str(samples.shape))
            logger.debug('samples:'+str(samples))
            action = np.random.choice(np.flatnonzero(samples==samples.max()))
            logger.debug('action:'+str(action))
            self.actions.append(action)
            return action

class XGBGreedy(Agent):
    def __init__(self, use_cuda=False, epsilon=0.05,
                max_depth=6, n_estimators=100,
                gamma=0, xgb_lambda=1.0, eta=0.3):
        logger.info("initializing agent")
        self.t = 1
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.observation_history = []
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
        assert self.X.shape[1] == len(self.feature_types)
        y = np.array(rewards).reshape(len(rewards),1)
        logging.debug('X_observations:'+str(X))
        logging.debug('y-reshaped:'+str(y))
        Xy = xgb.DMatrix(X, y, feature_types=self.feature_types, enable_categorical=True)
        self.model = xgb.train(params = self.model_parameters, dtrain=Xy, num_boost_round=self.n_estimators)
            
    def update_observations(self, observation, action, reward):
        self.observation_history.append(observation)
        self.rewards.append(reward)
        # add some if-statement to avoid training model at each iteration
        if self.t==9 or self.t%100==0:
            logger.info('Training model')
            logger.info('Observation history'+str(len(self.observation_history)))
            logger.info('Length rewards'+str(len(self.rewards)))
            self.train_model(self.observation_history,self.rewards)
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