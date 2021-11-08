import json

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('./data/item_vocab.json', 'r') as f:
    item_vocab = json.load(f)
reverse_vocab = {}
for item_id, idx in item_vocab.items():
    reverse_vocab[idx] = item_id
with open('./data/cat_vocab.json', 'r') as f:
    cat_vocab = json.load(f)
with open('./data/item_id2cat_id.json', 'r') as f:
    item_id2cat_id = json.load(f)


# def padding(sequence, history_max_length):
#     if len(sequence) >= history_max_length:
#         mask = [1] * history_max_length
#         sequence = sequence[:history_max_length]
#     elif len(sequence) < history_max_length:
#         mask = [1] * len(sequence) + [0] * (history_max_length - len(sequence))
#         sequence.extend([0] * (history_max_length - len(sequence)))
#     return sequence, mask


def index_item_id(item_id):
    return item_vocab[item_id] if item_id in item_vocab else item_vocab['unk']


def index_cat_id(cat_id):
    return cat_vocab[cat_id] if cat_id in cat_id else cat_vocab['unk']


def parse_line(line, history_max_length, sample_negative=False):
    line = line.strip().split("\t")
    label, user_id, target_item_id, target_cat_id, pos_his_item_ids, pos_his_cat_ids = line
    label = [float(label)]
    target_item_idx = [index_item_id(target_item_id)]
    target_cat_idx = [index_cat_id(target_cat_id)]

    pos_his_item_ids = pos_his_item_ids.split("")
    pos_his_item_idxes = [[index_item_id(item_id) for item_id in pos_his_item_ids]]
    pos_his_item_idxes = pad_sequences(pos_his_item_idxes, maxlen=history_max_length, padding='post', truncating='pre')

    pos_his_cat_ids = pos_his_cat_ids.split("")
    pos_his_cat_idxes = [[index_cat_id(cat_id) for cat_id in pos_his_cat_ids]]
    pos_his_cat_idxes = pad_sequences(pos_his_cat_idxes, maxlen=history_max_length, padding='post', truncating='pre')

    label = np.array(label, np.float32)
    target_item_idx = np.array(target_item_idx, np.int32)  # 1,
    target_cat_idx = np.array(target_cat_idx, np.int32)  # 1,
    pos_his_item_idxes = np.squeeze(pos_his_item_idxes)  # his_max_length,
    pos_his_cat_idxes = np.squeeze(pos_his_cat_idxes)  # his_max_length,
    feature = {'target_item': target_item_idx, 'target_cat': target_cat_idx, 'pos_his_item': pos_his_item_idxes,
               'pos_his_cat': pos_his_cat_idxes}
    if sample_negative:
        neg_his_item_idxes = np.random.randint(1, len(item_vocab), (history_max_length,))  # filter 0 for 0 is mask idx
        neg_his_cat_idxes = [index_cat_id(item_id2cat_id[reverse_vocab[item_idx]]) for item_idx in neg_his_item_idxes]
        neg_his_cat_idxes = np.array(neg_his_cat_idxes, np.int32)
        feature['neg_his_item'] = neg_his_item_idxes
        feature['neg_his_cat'] = neg_his_cat_idxes
    return feature, label


def example_generator(model_type, file, history_max_length):
    # print(type(model_type))
    model_type = model_type.decode("utf-8")  # tensorflow transform
    with open(file) as f:
        for line in f:
            if model_type == 'BASE' or model_type == 'DIN':
                feature, label = parse_line(line, history_max_length, sample_negative=False)
            elif model_type == 'DIEN':
                feature, label = parse_line(line, history_max_length, sample_negative=True)
            yield feature, label
