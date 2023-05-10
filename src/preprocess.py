import pandas as pd
from nltk import WordPunctTokenizer
from collections import Counter

# Movies Preprocessing

def preprocess_movies(movies, ratings, tokenizer):
    movies = genre_to_columns(movies)
    movies = add_year(movies)
    movies = tokenize_column(movies, 'title', tokenizer)
    movies = add_average_rating(movies, ratings)
    return movies

def genre_to_columns(movies):
    # ohe for genres in movies
    genres = movies['genres'].str.get_dummies('|')
    genres.rename(columns={'(no genres listed)': 'Other'}, inplace=True)
    genres.drop('IMAX', axis=1, inplace=True)
    movies = pd.concat([movies, genres], axis=1)
    movies.drop('genres', axis=1, inplace=True)
    return movies

def add_year(movies):
    year = movies['title'].str.extract('.*\((.*)\).*', expand=True)
    # nan -> 0, 2006-2007 -> 2006, 2011- -> 2011
    year = year.fillna(0).replace('(\d{4}).*', '\\1', regex=True).astype('int')
    movies['year'] = year
    # remove year from title
    movies['title'] = movies['title'].str.replace(' \(.*\)', '', regex=True)
    return movies

def tokenize_column(df, column, tokenizer):
    df[column] = df[column].apply(lambda x: ' '.join(tokenizer.tokenize(str(x))))
    return df

def add_average_rating(movies, ratings):
    avg_ratings = ratings.groupby('movieId')['rating'].mean()
    movies['avg_rating'] = avg_ratings
    return movies

# Users Preprocessing

def generate_users(movies, ratings, tags, n, min_count, tokenizer):
    tags = tokenize_column(tags, 'tag', tokenizer)
    users = get_users_with_softmax_genres(ratings, movies)
    users = add_tags(users, ratings, tags, n, min_count)
    return users

def get_users_with_softmax_genres(ratings, movies):
    movies_by_user = ratings.groupby('userId')['movieId'].apply(list)
    genres = list(movies.columns[1:-2])
    genre_count_by_user = movies_by_user.apply(lambda x: movies.loc[x, genres].sum())
    genre_count_by_user /= genre_count_by_user.sum(axis=1).values.reshape(-1, 1)
    users = pd.DataFrame(genre_count_by_user.values, index=genre_count_by_user.index, columns=genres)
    return users

def extract_top_tags_with_rating_in_range(ratings, tags, low, high, n, title, min_count=1):
    movies_with_range_rating_by_user = ratings[(low <= ratings['rating']) & (ratings['rating'] < high)].groupby('userId')['movieId'].apply(list)
    most_popular_tags_by_user = []
    for mvs in movies_with_range_rating_by_user:
        cnt = Counter()
        tags_for_movies = tags[tags['movieId'].isin(mvs)]['tag']
        for tag in tags_for_movies:
            cnt.update(tag.split())
        most_popular_tags_by_user.append(cnt.most_common(n))
    for cnt in most_popular_tags_by_user:
        for i in range(len(cnt)):
            if cnt[i][1] < min_count:
                cnt[i] = ('', 0)
    most_popular_tags_by_user = pd.DataFrame(most_popular_tags_by_user, index=movies_with_range_rating_by_user.index)
    most_popular_tags_by_user.fillna('', inplace=True)
    assert most_popular_tags_by_user.isna().sum().sum() == 0
    most_popular_tags_by_user = most_popular_tags_by_user.applymap(lambda x: x[0] if x else x)
    most_popular_tags_by_user.rename(columns=lambda x: f'{title}_{x}', inplace=True)
    return most_popular_tags_by_user

def add_tags(users, ratings, tags, n, min_count):
    positive_tags = extract_top_tags_with_rating_in_range(ratings, tags, 8, 11, n, 'positive', min_count)
    neutral_tags = extract_top_tags_with_rating_in_range(ratings, tags, 6, 8, n, 'neutral', min_count)
    negative_tags = extract_top_tags_with_rating_in_range(ratings, tags, 0, 6, n, 'negative', min_count)
    users = pd.concat([users, positive_tags, neutral_tags, negative_tags], axis=1)
    return users