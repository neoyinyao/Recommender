import os

os.environ['DGLBACKEND'] = 'tensorflow'
import re
import pickle

import argparse
import pandas as pd
import tensorflow as tf
import dgl

from graph_builder import PandasGraphBuilder
from util import train_test_split_by_time, build_val_test_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path
    data_dir = '../data/ml-1m'
    output_path = '../data/train_dataset.pkl'

    users = []
    with open(os.path.join(data_dir, 'users.dat'), encoding='latin1') as f:
        for l in f:
            id_, gender, age, occupation, zip_ = l.strip().split('::')
            users.append({
                'user_id': int(id_),
                'gender': gender,
                'age': age,
                'occupation': occupation,
                'zip': zip_,
            })
    users = pd.DataFrame(users).astype('category')
    movies = []
    with open(os.path.join(data_dir, 'movies.dat'), encoding='latin1') as f:
        for l in f:
            id_, title, genres = l.strip().split('::')
            genres_set = set(genres.split('|'))

            # extract year
            assert re.match(r'.*$', title)
            year = title[-5:-1]
            title = title[:-6].strip()

            data = {'movie_id': int(id_), 'year': year}
            for g in genres_set:
                data[g] = True
            movies.append(data)
    movies = pd.DataFrame(movies).astype({'year': 'category'})
    ratings = []
    with open(os.path.join(data_dir, 'ratings.dat'), encoding='latin1') as f:
        for l in f:
            user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
            ratings.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'rating': rating,
                'timestamp': timestamp,
            })
    ratings = pd.DataFrame(ratings)
    distinct_users_in_ratings = ratings['user_id'].unique()
    distinct_movies_in_ratings = ratings['movie_id'].unique()
    users = users[users['user_id'].isin(distinct_users_in_ratings)]
    movies = movies[movies['movie_id'].isin(distinct_movies_in_ratings)]
    genre_columns = movies.columns.drop(['movie_id', 'year'])
    movies[genre_columns] = movies[genre_columns].fillna(False).astype('bool')
    movies_categorical = movies

    utype = 'user'
    itype = 'movie'
    u2i_etype = 'watched'
    i2u_etype = 'watched-by'
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'user_id', 'user')
    graph_builder.add_entities(movies_categorical, 'movie_id', 'movie')
    graph_builder.add_binary_relations(ratings, 'user_id', 'movie_id', 'watched')
    graph_builder.add_binary_relations(ratings, 'movie_id', 'user_id', 'watched-by')

    g = graph_builder.build()
    g.nodes['user'].data['gender'] = tf.constant(users['gender'].cat.codes.values)
    g.nodes['user'].data['age'] = tf.constant(users['age'].cat.codes.values)
    g.nodes['user'].data['occupation'] = tf.constant(users['occupation'].cat.codes.values)
    g.nodes['user'].data['zip'] = tf.constant(users['zip'].cat.codes.values)

    g.nodes['movie'].data['year'] = tf.constant(movies['year'].cat.codes.values)
    g.nodes['movie'].data['genre'] = tf.constant(movies[genre_columns].values)

    g.edges['watched'].data['rating'] = tf.constant(ratings['rating'].values)
    g.edges['watched'].data['timestamp'] = tf.constant(ratings['timestamp'].values)
    g.edges['watched-by'].data['rating'] = tf.constant(ratings['rating'].values)
    g.edges['watched-by'].data['timestamp'] = tf.constant(ratings['timestamp'].values)
    train_indices, val_indices, test_indices = train_test_split_by_time(ratings, 'timestamp', 'user_id')

    train_g = dgl.edge_subgraph(g, edges={u2i_etype: train_indices, i2u_etype: train_indices}, preserve_nodes=True,
                                store_ids=True)
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, utype, itype, u2i_etype)
    dataset = {
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'user-type': utype,
        'item-type': itype,
        'user-to-item-type': u2i_etype,
        'item-to-user-type': i2u_etype,
        'timestamp-edge-column': 'timestamp'}
    total_dataset = {
        'total-graph': g,
        'user-type': utype,
        'item-type': itype,
        'user-to-item-type': u2i_etype,
        'item-to-user-type': i2u_etype}
    with open('../data/total_dataset.pkl', 'wb') as f:
        pickle.dump(total_dataset, f)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
