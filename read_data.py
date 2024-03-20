import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data():
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
    print(movie_data.columns)
    columns_to_drop = ['video_release_date', 'imdb_url']
    movie_data=movie_data.drop(columns=columns_to_drop) # all instances had NaN
    movie_data['movie_title'] = movie_data['movie_title'].astype("category").cat.codes.astype(int)
    for col in movie_data.columns:
        movie_data[col] = movie_data[col].astype("category").cat.codes.astype(int)
    print(movie_data.describe())
    # Get user ratings
    user_rating_df = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id','movie_id','rating','timestamp'])
    user_rating_df.user_id -= 1 # make this column zero-indexed
    user_rating_df.movie_id -= 1 # make this column zero-indexed
    user_rating_df['user_id'] = user_rating_df['user_id'].astype("category").cat.codes.astype(int)
    user_rating_df['movie_id'] = user_rating_df['movie_id'].astype("category").cat.codes.astype(int)
    #user_rating_df['timestamp'] = user_rating_df['timestamp'].astype("category")

    # Get domestic data
    user_data = pd.read_csv('ml-100k/u.user', sep='|', header=None,names=['user_id','age','gender','work','zipcode'])
    user_data.user_id -= 1 # make this column zero-indexed
    user_data['user_id'] = user_data['user_id'].astype("category").cat.codes.astype(int)
    user_data['age'] = user_data['age'].astype(int)
    user_data['gender'] = user_data['gender'].astype("category").cat.codes.astype(int)
    user_data['work'] = user_data['work'].astype("category").cat.codes.astype(int)
    user_data['zipcode'] = user_data['zipcode'].astype(str).str[0]
    user_data['zipcode'] = user_data['zipcode'].astype("category").cat.codes.astype(int)
    allowed_zips = [str(i) for i in range(10)]
    user_data.loc[~user_data["zipcode"].isin(allowed_zips),"zipcode"]="V"
    user_data['zipcode'] = user_data['zipcode'].astype("category").cat.codes.astype(int)

    return movie_data, user_rating_df, user_data