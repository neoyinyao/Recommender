import numpy as np
import tensorflow as tf
import dgl


def get_item_reprs(model, pinsage_sampler, train_g, itype, batch_size):
    """
    :param model: PinSage Model
    :param pinsage_sampler: data_loader.PinSageSampler instance
    :param train_g: train dgl heterogeneous graph instance
    :param itype: item node type in train_g
    :param batch_size:batch_size
    :return: node embedding by pinsage model,shape (num_nodes,hidden_size)
    """
    test_dataset = tf.data.Dataset.range(train_g.num_nodes(itype)).batch(batch_size)
    item_reprs = None
    for seeds in test_dataset:
        blocks = pinsage_sampler.generate_blocks(seeds)
        batch_reprs = model.get_repr(blocks)  # None,hidden_size
        if item_reprs is not None:
            item_reprs = np.concatenate([item_reprs, batch_reprs.numpy()], axis=0)
        else:
            item_reprs = batch_reprs
    return item_reprs


def recommend(full_graph, top_k, item_reprs, user2item_etype, utype, timestamp, batch_size):
    """
    return (n_user, top_k) matrix of recommended items for each user
    """
    graph_slice = full_graph.edge_type_subgraph([user2item_etype])
    n_users = full_graph.number_of_nodes(utype)
    latest_interactions = dgl.sampling.select_topk(graph_slice, 1, timestamp, edge_dir='out')
    user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')
    # each user should have at least one "latest" interaction
    assert np.any(np.equal(user, np.arange(n_users)))
    recommends = None
    user_dataset = tf.data.Dataset.range(n_users).batch(batch_size)
    for user_batch in user_dataset:
        latest_items_batch = tf.nn.embedding_lookup(latest_items, user_batch)
        latest_item_repr = item_reprs[latest_items_batch, :]  # test_batch_size,hidden_size
        similarity = np.matmul(latest_item_repr, item_reprs.T)
        for i, u in enumerate(user_batch):  # exclude items that are already interacted
            interacted_items = full_graph.successors(u, etype=user2item_etype)
            similarity[i, interacted_items] = -np.inf
        recommend_batch = np.argpartition(similarity, kth=top_k)[:, -top_k:]
        if recommends is not None:
            recommends = np.concatenate((recommends, recommend_batch), axis=0)
        else:
            recommends = recommend_batch
    return recommends


def hit_rate_eval(recommendations, ground_truth):
    """
    recommendations: n_users,top_k
    ground_truth: sparse matrix, n_users,n_items with match val 1
    """
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    hit = relevance.any(axis=1).mean()
    return hit
