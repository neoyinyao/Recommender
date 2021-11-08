import json


def build_vocab():
    with open('data/local_train_splitByUser') as f:
        print('build_vocab')
        item_id2cat_id = {'unk': 'unk'}
        item_ids_set = set()
        cat_ids_set = set()
        for line in f:
            line = line.strip().split("\t")
            _, _, item_id, cat_id, his_item_ids, his_cat_ids = line
            his_item_ids = his_item_ids.split("")
            his_cat_ids = his_cat_ids.split("")
            item_ids_set.add(item_id)
            item_ids_set.update(his_item_ids)
            cat_ids_set.add(cat_id)
            cat_ids_set.update(his_cat_ids)
            item_id2cat_id[item_id] = cat_id
            for item_id, cat_id in zip(his_item_ids, his_cat_ids):
                item_id2cat_id[item_id] = cat_id
        item_vocab = {}
        cat_vocab = {}
        for idx, item in enumerate(item_ids_set, start=1):
            item_vocab[item] = idx
        for idx, cat in enumerate(cat_ids_set, start=1):
            cat_vocab[cat] = idx
        item_vocab['mask'] = 0
        item_vocab['unk'] = len(item_vocab)
        cat_vocab['mask'] = 0
        cat_vocab['unk'] = len(cat_vocab)
        with open('./data/item_vocab.json', 'w') as f:
            json.dump(item_vocab, f)
        with open('./data/cat_vocab.json', 'w') as f:
            json.dump(cat_vocab, f)
        with open('./data/item_id2cat_id.json', 'w') as f:
            json.dump(item_id2cat_id, f)
