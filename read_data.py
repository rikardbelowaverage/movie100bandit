import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from utils.setup_logger import logger

def get_movie_data():
    # Get movie data
    movie_features = ['movie_id',
                            'movie_title',
                            'release_date',
                            'video_release_date',
                            'imdb_url',
                            'unknown',
                            'action',
                            'adventure',
                            'animation',
                            'children',
                            'comedy',
                            'crime',
                            'documentary',
                            'drama',
                            'fantasy',
                            'film_noir',
                            'horror',
                            'musical',
                            'mystery'
                            'romance',
                            'sci_fi',
                            'thriller',
                            'war',
                            'western']
    
    movie_data = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin1', index_col=False,
                    names=movie_features)
    movie_data.movie_id -= 1 # make this column zero-indexed
    #print(movie_data.columns)
    columns_to_drop = ['video_release_date', 'imdb_url']
    movie_data=movie_data.drop(columns=columns_to_drop) # all instances had NaN
    movie_data['movie_title'] = movie_data['movie_title'].astype("category").cat.codes.astype(int)
    for col in movie_data.columns:
        movie_data[col] = movie_data[col].astype("category").cat.codes.astype(int)
    
    #print(movie_data.describe())
    # Get user ratings
    user_rating_df = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id','movie_id','rating','timestamp'])
    user_rating_df.user_id -= 1 # make this column zero-indexed
    user_rating_df.movie_id -= 1 # make this column zero-indexed
    user_rating_df['user_id'] = user_rating_df['user_id'].astype("category").cat.codes.astype(int)
    user_rating_df['movie_id'] = user_rating_df['movie_id'].astype("category").cat.codes.astype(int)
    movie_data=movie_data.assign(avg_rating=user_rating_df.groupby('movie_id')['rating'].transform('mean'))
    logger.info("Movie_data columns:"+str(movie_data.columns))
    movie_data = movie_data[['avg_rating','movie_id', 'movie_title', 'release_date', 'unknown', 'action',
       'adventure', 'animation', 'children', 'comedy', 'crime', 'documentary',
       'drama', 'fantasy', 'film_noir', 'horror', 'musical', 'mysteryromance',
       'sci_fi', 'thriller', 'war', 'western']]
    #user_rating_df['timestamp'] = user_rating_df['timestamp'].astype("category")
    logger.info("Movie_data columns:"+str(movie_data.columns))
    logger.info(user_rating_df.columns)
    # Get domestic data
    user_data = pd.read_csv('ml-100k/u.user', sep='|', header=None,names=['user_id','age','gender','work','zipcode','age_group'])
    user_data.user_id -= 1 # make this column zero-indexed
    user_data['user_id'] = user_data['user_id'].astype("category").cat.codes.astype(int)
    #user_data['age'] = user_data['age'].astype(int)
    user_data["age_group"] = pd.cut(x=user_data['age'], bins=[0,10,15,20,30,40,50,100], labels=["kid","young_teen","old_teen","twenties","thirties","golden_age","old_but_gold"])
    user_data["age_group"] = user_data["age_group"].astype("category").cat.codes.astype(int)
    user_data=user_data.drop(columns='age') 
    user_data['gender'] = user_data['gender'].astype("category").cat.codes.astype(int)
    user_data['work'] = user_data['work'].astype("category").cat.codes.astype(int)
    user_data['zipcode'] = user_data['zipcode'].astype(str).str[0]
    user_data['zipcode'] = user_data['zipcode'].astype("category").cat.codes.astype(int)
    allowed_zips = [i for i in range(10)]
    user_data.loc[~user_data["zipcode"].isin(allowed_zips),"zipcode"]="V"
    user_data['zipcode'] = user_data['zipcode'].astype("category").cat.codes.astype(int)
    user_data=user_data.assign(avg_given_rating=user_rating_df.groupby('user_id')['rating'].transform('mean'))
    #user_data=user_data[['user_id', 'gender', 'work', 'zipcode', 'age_group',
    #   'avg_given_rating']]
    logger.info("user_data_columns:"+str(user_data.columns))

    feature_types = ['q' for _ in range(2)]+['c' for _ in range(24)]
   # feature_types = ['c' for _ in range(20)]+['q']+['c' for _ in range(4)]
    logger.info(feature_types)
    return movie_data, user_rating_df, user_data, feature_types

def get_mushroom_data():
    pd.set_option('display.max_columns', None)
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    # data (as pandas dataframes) 
    X = mushroom.data.features
    X['poisonous'] = mushroom.data.targets
    #Xy = X.append(y, ignore_index=True)
    #Xy = X
    X.dropna(subset=['poisonous'])

    #logging.info('Xy0: ' + str(Xy))
    #Xy = Xy.astype("category").apply(lambda x: x.cat.codes).astype("int64")
    #logging.info('Xy1: ' + str(Xy))
    #Xy = Xy[Xy['poisonous'] >= 0]

    X['stalk-root'] = X['stalk-root'].fillna('m')  #'m' for missing

    #X.fillna('na', inplace=True)
    X = X.astype("category").apply(lambda x: x.cat.codes).astype(int)
    #X = X + 1
    #X = X.replace(-1, np.NaN)
    #X = X.replace(-1, 0)
    
    # metadata 
    #print(mushroom.metadata) 
    
    # variable information 
    #print(mushroom.variables) 
    null_mask = X.isnull().any(axis=1)
    null_rows = X[null_mask]

    logger.info('null_rows: ' + str(null_rows))

    minMax = X.agg([min, max])

    logger.info('minMax: ' + str(minMax))

    logger.info('X: ' + str(X))
    #logging.info('Xy2: ' + str(Xy))

    set_classes = X["poisonous"].unique()
    n_classes = len(set_classes)
    n_features = X.shape[1] - 1
    feature_types = ['c' for i in range(23)]

    return X, feature_types