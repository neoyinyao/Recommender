import numpy as np
import pickle
import random

with open('data/mid_voc.pkl', 'rb') as f:
    item_id_voc = pickle.load(f)
with open('data/cat_voc.pkl', 'rb') as f:
    item_cat_voc = pickle.load(f)

item_id2cat_id = {'default_mid': 'default_cat'}
with open('data/local_train_splitByUser') as f:
    print('prepare item_id2cat_id')
    for line in f:
        line = line.strip().split("\t")
        _, _, item_id, cat_id, his_item_ids, his_cat_ids = line
        his_item_ids = his_item_ids.split("")
        his_cat_ids = his_cat_ids.split("")
        item_id2cat_id[item_id] = cat_id
        for item_id, cat_id in zip(his_item_ids, his_cat_ids):
            item_id2cat_id[item_id] = cat_id


def padding(sequence, history_max_length):
    if len(sequence) >= history_max_length:
        mask = [1] * history_max_length
        sequence = sequence[:history_max_length]
    elif len(sequence) < history_max_length:
        mask = [1] * len(sequence) + [0] * (history_max_length - len(sequence))
        sequence.extend([0] * (history_max_length - len(sequence)))
    return sequence, mask


def get_item_idx(item_id):
    return item_id_voc[item_id] if item_id in item_id_voc else 0


def get_cat_idx(cat_id):
    return item_cat_voc[cat_id] if cat_id in item_cat_voc else 0


def parse_line(line, history_max_length):
    line = line.strip().split("\t")
    label, user_id, item_id, cat_id, his_item_ids, his_cat_ids = line
    label = float(label)
    item_idx = [get_item_idx(item_id)]
    cat_idx = [get_cat_idx(cat_id)]

    his_item_ids = his_item_ids.split("")
    his_item_idxs = [get_item_idx(item_id) for item_id in his_item_ids]
    his_item_idxs, mask = padding(his_item_idxs, history_max_length)

    his_cat_ids = his_cat_ids.split("")
    his_cat_idxs = [get_cat_idx(cat_id) for cat_id in his_cat_ids]
    his_cat_idxs, mask = padding(his_cat_idxs, history_max_length)

    label = np.array(label, np.float32)
    item_idx = np.array(item_idx, np.int32)
    cat_idx = np.array(cat_idx, np.int32)
    his_item_idxs = np.array(his_item_idxs, np.int32)
    his_cat_idxs = np.array(his_cat_idxs, np.int32)
    mask = np.array(mask, np.float32)
    return label, item_idx, cat_idx, his_item_idxs, his_cat_idxs, mask


def din_example_generator(file, history_max_length):
    with open(file) as f:
        for line in f:
            yield parse_line(line, history_max_length)


def dien_example_generator(file, history_max_length):
    neg_pool = list(item_id_voc.keys())
    with open(file) as f:
        for line in f:
            label, item_idx, cat_idx, pos_his_item_idxs, pos_his_cat_idxs, mask = parse_line(line, history_max_length)
            neg_his_item_ids = random.choices(neg_pool, k=history_max_length)
            neg_his_item_idxs = [get_item_idx(item_id) for item_id in neg_his_item_ids]
            neg_his_cat_idxs = [get_cat_idx(item_id2cat_id[item_id]) for item_id in neg_his_item_ids]
            neg_his_item_idxs = np.array(neg_his_item_idxs, np.int32)
            neg_his_cat_idxs = np.array(neg_his_cat_idxs, np.int32)
            yield label, item_idx, cat_idx, pos_his_item_idxs, pos_his_cat_idxs, neg_his_item_idxs, neg_his_cat_idxs, mask
