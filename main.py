from read_data import get_movie_data
import numpy as np
import pandas as pd
import xgboost as xgb
import random
from helpers import get_regret
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
    timesteps = 10000

X_history = []
y_history = []
y_optimal = []
regrets = []

if __name__ == "__main__":
    df_movies, df_user_ratings, df_user_data, feature_types = get_movie_data()
    sampled_users = df_user_data.sample(n=timesteps, random_state=1, replace=True)
    model = TETS(feature_types=feature_types)

    for t in range(0,timesteps):
        if t%100==0:
            logger.info("timestep: "+str(t))
        # Get contextual data regarding the current user
        while True:
            current_user = sampled_users.iloc[t]
            current_user_ratings = df_user_ratings.loc[df_user_ratings['user_id']==current_user.user_id]
            current_user_rated_movies = df_movies[df_movies['movie_id'].isin(current_user_ratings.movie_id.values)]
            logger.debug(current_user_rated_movies)
            logger.debug(df_user_ratings)
            logger.debug(df_movies)
            

            if (len(current_user_rated_movies)<=1):
                logger.info("Resampling a new user")
                #TODO drop the current user from sampled_users df
                t = random.randint(0,len(df_user_data))
            else:
                break
        user_features = current_user.drop(index=('user_id'))

        # Get contextual data regarding all movies rated by current_user
        action_features = current_user_rated_movies.drop(columns=['release_date'])
        # Gets all ratings which will serve as the ground truth when comparing the recommendations
        ratings = current_user_ratings.rating.values
        rows, columns = action_features.shape
        #logger.info(action_features.columns)
        #logger.info(user_features.columns)
        # X presents the possible actions to the agent
        X = np.concatenate((action_features,np.tile(user_features,(rows,1))), axis=1)
        #logger.info(X[0,:])
        X[:, [1, 25]] = X[:, [25, 1]]
        #logger.info('X-datatypes:'+str(X.dtype))
        #logger.info(X[0,:])
        #X = np.concatenate((np.tile(user_features,(rows,1)),action_features), axis=1)
        # y are the rewards for each actions, not presented to the agent.
        y = ratings.reshape(rows,1)
        logger.debug(str(y))
        logger.debug("X.shape is:"+str(X.shape))
        logger.debug('y-shape is:'+str(y.shape))
        logger.debug(X[0,:])
        logger.debug(user_features)
        logger.debug(action_features)
        #X[:,0]=y.ravel()
        action = model.pick_action(observations=X)
        logger.debug('Action:'+str(action))
        model.update_observations(observation=X[action,:],action=action,reward=y[action])

        max_reward = ratings[ratings.argmax(axis=0)]
        y_optimal.append(max_reward)
        regret = get_regret(max_reward,y[action])
        regrets.append(regret)
        #print("Recommended movie had rating {}, most liked movie had rating {}, regret for timestep {} is therefore {}".format(y[action],max_reward,t,regret))
        #remove the recommended movie, movie_id, from rated movies by user_id
        recommended_movie_id = current_user_rated_movies.iloc[action].movie_id
        df_user_ratings = df_user_ratings.loc[~((df_user_ratings['movie_id'] == recommended_movie_id) & (df_user_ratings['user_id']==current_user.user_id))] 

    regret_plot(timesteps,regrets,result_path)

    arr = np.load('sample_data.npy')
    data = arr[:,:-1]
    labels = arr[:,-1]
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    X_d = xgb.DMatrix(data_test, feature_types=model.feature_types, enable_categorical=True)
    pred = model.predict(X_d)
    MSE = np.square(np.subtract(labels_test,pred)).mean()
    logger.info('MSE:'+str(MSE))
    RMSE = np.sqrt(MSE)
    logger.info('RMSE:'+str(RMSE))