import pandas as pd

def load_data(dr):
    """
    :param dr: data directory
    :return: movies, ratings, tags dataframes
    """
    movies = pd.read_csv(dr + 'movies.csv', index_col='movieId')
    ratings = pd.read_csv(dr + 'ratings.csv', usecols=['userId', 'movieId', 'rating'])
    tags = pd.read_csv(dr + 'tags.csv', usecols=['userId', 'movieId', 'tag'])
    return movies, ratings, tags