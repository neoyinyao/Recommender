import numpy as np
from scipy import sparse as ssp


def train_test_split_by_time(df, timestamp, user):
    df['train_mask'] = np.ones((len(df),), dtype=np.bool)
    df['val_mask'] = np.zeros((len(df),), dtype=np.bool)
    df['test_mask'] = np.zeros((len(df),), dtype=np.bool)

    def train_test_split(df):
        df = df.sort_values([timestamp])
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    df = df.groupby(user, group_keys=False).apply(train_test_split).sort_index()
    print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))
    return df['train_mask'].to_numpy().nonzero()[0], \
           df['val_mask'].to_numpy().nonzero()[0], \
           df['test_mask'].to_numpy().nonzero()[0]


def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    n_users = g.number_of_nodes(utype)
    n_items = g.number_of_nodes(itype)
    val_src, val_dst = g.find_edges(val_indices, etype=etype)
    test_src, test_dst = g.find_edges(test_indices, etype=etype)
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()
    val_matrix = ssp.coo_matrix((np.ones_like(val_src), (val_src, val_dst)), (n_users, n_items))
    test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))

    return val_matrix, test_matrix
