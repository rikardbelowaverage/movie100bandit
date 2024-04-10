from read_data import get_mushroom_data
import numpy as np
import pandas as pd
import xgboost as xgb
import random
from helpers import get_binary_reward
from xgboost import XGBRegressor
from agents import TETS,XGBGreedy
from utils.setup_logger import logger
from utils.create_graphs import regret_plot
from sklearn.model_selection import train_test_split
import argparse

#TODO make config handler
parser = argparse.ArgumentParser()
parser.add_argument('--result_path', help='result_path', type=str)
parser.add_argument('--time_steps', help='time_steps', type=int)
args, _ = parser.parse_known_args()

if args.result_path is not None:
    result_path = args.result_path
if args.time_steps is not None:
    timesteps = args.time_steps
else:
    timesteps = 100

X_history = []
y_history = []
y_optimal = []
regrets = []

if __name__ == "__main__":
    #return movie_df, rating_df, user_df
    data, feature_types = get_mushroom_data()
    cr_data = data.sample(n=timesteps,random_state=42,replace=True).reset_index(drop=True)
    model = TETS(feature_types=feature_types)

    features = (cr_data.loc[:,cr_data.columns != 'poisonous']).to_numpy()
    actions = (cr_data.loc[:,cr_data.columns == 'poisonous']).to_numpy()

    for t in range(timesteps):
        context = features[t,:]
        target = actions[t]
        logger.debug('Context:'+str(context))
        logger.debug('target:'+str(target))
        #X=np.concatenate((np.tile(context,(2,1)),np.array([[0],[1]])),axis=1)
        X=np.concatenate((np.array([[0],[1]]),np.tile(context,(2,1))),axis=1)
        logger.debug('X.shape is:'+str(X.shape))
        logger.debug(str(X))
        action = model.pick_action(observations=X)
        logger.debug('action is:'+str(action)+' correct prediction was:'+str(target))

        reward = get_binary_reward(target,action)
        regrets.append(1-reward)
        model.update_observations(observation=X[action,:],action=action,reward=reward) #not sure if we should give target

    regret_plot(timesteps,regrets,result_path)

