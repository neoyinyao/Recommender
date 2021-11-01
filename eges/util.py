import pickle
import random
import json
from tqdm import tqdm
import dgl
import tensorflow as tf


def load_metadata():
    item2cat = {}
    item2brand = {}
    with open('./data/meta_Electronics.json') as f:
        for l in tqdm(f):
            example = json.loads(l.strip())
            item_id = example['asin']
            item2cat[item_id] = example['main_cat']
            item2brand[item_id] = example['brand']
    with open('item2cat.json', 'w') as f:
        json.dump(item2cat, f)
    with open('item2brand.json', 'w') as f:
        json.dump(item2brand, f)
    # with open('data/item2cat.json', 'r') as f:
    #     item2cat = json.load(f)
    # with open('data/item2brand.json', 'r') as f:
    #     item2brand = json.load(f)
    item_pairs_count = {}
    with open('./data/meta_Electronics.json') as f:
        for l in tqdm(f):
            example = json.loads(l.strip())
            item = example['asin']
            co_occurrence_items = example['also_buy']
            if not co_occurrence_items:
                continue
            for co_occurrence_item in co_occurrence_items:
                if co_occurrence_item in item2cat:
                    item_pairs = [(item, co_occurrence_item), (co_occurrence_item, item)]
                    for item_pair in item_pairs:
                        if item_pair in item_pairs_count:
                            item_pairs_count[item_pair] += 1
                        else:
                            item_pairs_count[item_pair] = 1

    directed_item_pairs_count = {}
    for item_pair, count in item_pairs_count.items():
        reverse_item_pair = (item_pair[1], item_pair[0])
        if reverse_item_pair not in directed_item_pairs_count:
            max_count = max(count, item_pairs_count[(item_pair[1], item_pair[0])])
            directed_item_pairs_count[item_pair] = max_count
    with open('data/directed_item_pairs_count.pkl', 'wb') as f:
        pickle.dump(directed_item_pairs_count, f)
    return directed_item_pairs_count, item2cat, item2brand


def train_test_split(directed_item_pairs_count):
    directed_item_pairs_count_li = list(directed_item_pairs_count.keys())
    random.shuffle(directed_item_pairs_count_li)
    train_size = len(directed_item_pairs_count_li) * 2 // 3  # 训练数据2/3
    train_item_pairs = directed_item_pairs_count_li[:train_size]
    test_item_pairs = directed_item_pairs_count_li[train_size:]
    return train_item_pairs, test_item_pairs


def build_vocab(train_item_pairs, directed_item_pairs_count, item2cat, item2brand):
    # 根据in degree构建item2idx
    in_degree = {}
    for query_item, match_item in train_item_pairs:
        count = directed_item_pairs_count[(query_item, match_item)]
        if match_item not in in_degree:
            in_degree[match_item] = count
        else:
            in_degree[match_item] += count

        if query_item not in in_degree:
            in_degree[query_item] = count
        else:
            in_degree[query_item] += count
    in_degree_li = []
    for item_id, degree in in_degree.items():
        in_degree_li.append((item_id, degree))
    in_degree_li.sort(key=lambda x: x[1], reverse=True)
    item2idx = {'': 0}
    for idx, (item_id, _) in enumerate(in_degree_li, start=1):
        item2idx[item_id] = idx
    idx2item = {}
    for item, idx in item2idx.items():
        idx2item[idx] = item

    # 根据训练数据构建vocab
    train_item2cat = {}
    train_item2brand = {}
    for item in item2idx:
        if item in item2cat:
            train_item2cat[item] = item2cat[item]
        if item in item2brand:
            train_item2brand[item] = item2brand[item]

    cat_vocab = {}
    cat_set = set()
    for _, cat in train_item2cat.items():
        cat_set.add(cat)
    for idx, cat in enumerate(cat_set, start=1):
        cat_vocab[cat] = idx
    cat_vocab[''] = 0

    brand_vocab = {}
    brand_set = set()
    for _, brand in train_item2brand.items():
        brand_set.add(brand)
    for idx, brand in enumerate(brand_set, start=1):
        brand_vocab[brand] = idx
    brand_vocab[''] = 0

    return item2idx, idx2item, train_item2cat, train_item2brand, brand_vocab, cat_vocab


def build_train_graph(directed_item_pairs_count, train_item_pairs, item2idx):
    # 构造训练图
    src = []
    dst = []
    edata = []
    for query_item, match_item in train_item_pairs:
        src.append(item2idx[query_item])
        dst.append(item2idx[match_item])
        edata.append(directed_item_pairs_count[(query_item, match_item)])
        src.append(item2idx[match_item])
        dst.append(item2idx[query_item])
        edata.append(directed_item_pairs_count[(query_item, match_item)])
    num_nodes = len(item2idx)
    train_g = dgl.graph(data=(src, dst), num_nodes=num_nodes)
    edata = tf.constant(edata, dtype=tf.float32)
    train_g.edata['weight'] = edata
    return train_g


def build_train_util():
    with open('./data/item2cat.json', 'r') as f:
        item2cat = json.load(f)
    with open('./data/item2brand.json', 'r') as f:
        item2brand = json.load(f)
    with open('./data/directed_item_pairs_count.pkl', 'rb') as f:
        directed_item_pairs_count = pickle.load(f)

    train_item_pairs, test_item_pairs = train_test_split(directed_item_pairs_count)
    item2idx, idx2item, train_item2cat, train_item2brand, brand_vocab, cat_vocab = build_vocab(train_item_pairs,
                                                                                               directed_item_pairs_count,
                                                                                               item2cat, item2brand)
    train_g = build_train_graph(directed_item_pairs_count, train_item_pairs, item2idx)
    return train_g, test_item_pairs, item2idx, idx2item, train_item2cat, train_item2brand, brand_vocab, cat_vocab


if __name__ == '__main__':
    load_metadata()
