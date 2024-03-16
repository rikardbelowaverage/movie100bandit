import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    # Get movie data
    movie_data = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin1', index_col=False,
                    names=['movie_id',
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
                            'western'])
    movie_data.movie_id -= 1 # make this column zero-indexed
    columns_to_drop = ['video_release_date', 'imdb_url']
    movie_data=movie_data.drop(columns=columns_to_drop) # all instances had NaN
    # Get user ratings
    user_rating_df = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=['user_id','movie_id','rating','timestamp'])
    user_rating_df.user_id -= 1 # make this column zero-indexed
    user_rating_df.movie_id -= 1 # make this column zero-indexed
    user_rating_df['user_id'] = user_rating_df['user_id'].astype("category")
    user_rating_df['movie_id'] = user_rating_df['movie_id'].astype("category")
    #user_rating_df['timestamp'] = user_rating_df['timestamp'].astype("category")

    # Get domestic data
    user_data = pd.read_csv('ml-100k/u.user', sep='|', header=None,names=['user_id','age','gender','work','zipcode'])
    user_data.user_id -= 1 # make this column zero-indexed
    user_data['user_id'] = user_data['user_id'].astype("category")
    user_data['age'] = user_data['age'].astype(int)
    user_data['gender'] = user_data['gender'].astype("category")
    user_data['work'] = user_data['work'].astype("category")
    user_data['zipcode'] = user_data['zipcode'].astype(str).str[0]
    user_data['zipcode'] = user_data['zipcode'].astype("category")
    allowed_zips = ["0","1","2","3","4","5","6","7","8","9"]
    user_data.loc[~user_data["zipcode"].isin(allowed_zips),"zipcode"]="V"
    user_data['zipcode'] = user_data['zipcode'].astype("category")

    return movie_data, user_rating_df, user_data