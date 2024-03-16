from read_data import get_data
import numpy as np
import pandas as pd
import random
from helpers import get_regret
from xgboost import XGBRegressor
#from agents import NormalGraphXGBoostAgent

timesteps = 10
X_history = []
y_history = []

if __name__ == "__main__":
    #return movie_df, rating_df, user_df
    df_movies, df_user_ratings, df_user_data = get_data()
    sampled_users = df_user_data.sample(n=timesteps, random_state=1, replace=True)

    model = XGBRegressor()
    
    for t in range(0,timesteps):
        # Get contextual data regarding the current user
        current_user = sampled_users.iloc[t]
        current_user_ratings = df_user_ratings.loc[df_user_ratings['user_id']==current_user.user_id]
        current_user_rated_movies = df_movies[df_movies['movie_id'].isin(current_user_ratings.movie_id.values)]
        user_features = current_user.to_numpy()

        # Get contextual data regarding all movies rated by current_user
        action_features = current_user_rated_movies.to_numpy()
        # Gets all ratings which will serve as the ground truth when comparing the recommendations
        ratings = current_user_ratings.rating.values
        movies, features = action_features.shape

        # X used for the model.predict
        X = np.concatenate((np.tile(user_features,(movies,1)),action_features), axis=1)
        y = ratings.reshape(movies,1)


        if t < 10:
            random.seed(42)
            action = random.randint(0,movies-1)
        else:
            pass
        
        reward = ratings[action]
        max_reward = ratings[ratings.argmax(axis=0)]
        print("Recommended movie had rating {}, most liked movie had rating {}, regret for timestep {} is therefore {}".format(reward,max_reward,t,max_reward-reward))
        regret = get_regret(max_reward,reward)

        X_history.append(X[action])
        y_history.append(reward)
        print(X_history)
        print((np.array(X_history)).shape)
        #model.fit(np.array(X_history),y_history)




