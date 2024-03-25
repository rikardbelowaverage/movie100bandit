from read_data import get_data
import numpy as np
import pandas as pd
import random
from helpers import get_regret
from xgboost import XGBRegressor
from agents import TETS
import matplotlib.pyplot as plt
import argparse
import logging

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

logging.info("Initializing experiment for " +str(timesteps) + " timesteps")

X_history = []
y_history = []
y_optimal = []
regrets = []

if __name__ == "__main__":
    #return movie_df, rating_df, user_df
    agentz = [TETS()]
    for a in agentz:
        df_movies, df_user_ratings, df_user_data = get_data()
        sampled_users = df_user_data.sample(n=timesteps, random_state=1, replace=True)
        model = a

        for t in range(0,timesteps):
            if t%100==0:
                print("timestep: "+str(t))
            # Get contextual data regarding the current user
            while True:
                current_user = sampled_users.iloc[t]
                current_user_ratings = df_user_ratings.loc[df_user_ratings['user_id']==current_user.user_id]
                current_user_rated_movies = df_movies[df_movies['movie_id'].isin(current_user_ratings.movie_id.values)]

                if (len(current_user_rated_movies)<=1):
                    #TODO drop the current user from sampled_users df
                    logging.info("User has no more rated movies, sampling a new user")
                    t = random.randint(0,len(df_user_data))
                else:
                    break
            user_features = current_user.drop(index=('user_id'))

            # Get contextual data regarding all movies rated by current_user
            action_features = current_user_rated_movies.drop(columns=['movie_title','release_date'])
            # Gets all ratings which will serve as the ground truth when comparing the recommendations
            ratings = current_user_ratings.rating.values
            rows, columns = action_features.shape

            # X presents the possible actions to the agent
            X = np.concatenate((np.tile(user_features,(rows,1)),action_features), axis=1)
            # y are the rewards for each actions, not presented to the agent.
            y = ratings.reshape(rows,1)

            action = model.pick_action(observations=X)
            model.update_observations(observation=X[action],action=action,reward=y[action])

            max_reward = ratings[ratings.argmax(axis=0)]
            y_optimal.append(max_reward)
            print("Recommended movie had rating {}, most liked movie had rating {}, regret for timestep {} is therefore {}".format(y[action],max_reward,t,max_reward-y[action]))
            regret = get_regret(max_reward,y[action])
            regrets.append(regret)

            #remove the recommended movie, movie_id, from rated movies by user_id
            recommended_movie_id = current_user_rated_movies.iloc[action].movie_id
            df_user_ratings = df_user_ratings.loc[~((df_user_ratings['movie_id'] == recommended_movie_id) & (df_user_ratings['user_id']==current_user.user_id))] 


        ts = range(0,timesteps)
        plt.plot(ts,regrets,'.',linestyle='')
        plt.savefig(result_path+'/instant_regret.png')
        plt.clf()

        plt.plot(ts,np.cumsum(regrets))
        plt.savefig(result_path+'/cum_regret.png')