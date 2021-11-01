import random

import dgl
import tensorflow as tf


def get_item_idx(item2idx, item_id):
    return item2idx[item_id] if item_id in item2idx else 0


def get_side_info(item2brand, brand_vocab, item2cat, cat_vocab, item_id):
    if item_id in item2cat:
        cat_id = item2cat[item_id]
        cat_idx = cat_vocab[cat_id] if cat_id in cat_vocab else 0
    else:
        cat_idx = 0
    if item_id in item2brand:
        brand_id = item2brand[item_id]
        brand_idx = brand_vocab[brand_id] if brand_id in brand_vocab else 0
    else:
        brand_idx = 0
    return [cat_idx], [brand_idx]


def build_dataset(model_type, mode, train_g, sample_k, random_walk_length, num_ns, cat_vocab, brand_vocab, item2cat,
                  item2brand, item2idx, idx2item, test_item_pairs):
    def train_example_generator():
        while True:
            src_nodes = list(range(1, len(idx2item)))  # exclude 0 because 0 is oov
            seeds = random.choices(src_nodes, k=sample_k)
            sequences, _ = dgl.sampling.random_walk(train_g, nodes=seeds, length=random_walk_length,
                                                    prob='weight', restart_prob=0)  # sample_k,random_walk_length
            for sequence in sequences:
                couples, labels = tf.keras.preprocessing.sequence.skipgrams(sequence, window_size=5,
                                                                            vocabulary_size=len(idx2item),
                                                                            negative_samples=0)
                for target_item, context_item in couples:
                    context_class = context_item[tf.newaxis, tf.newaxis]  # 1,1
                    negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                        true_classes=context_class,
                        num_true=1,
                        num_sampled=num_ns,
                        unique=True,
                        range_max=len(idx2item),
                        name="negative_sampling")

                    # Build context and label vectors (for one target item)
                    negative_sampling_candidates = tf.expand_dims(
                        negative_sampling_candidates, 1)  # num_ns,1

                    context = tf.concat([context_class, negative_sampling_candidates], 0)  # 1+num_ns,1
                    label = tf.constant([1] + [0] * num_ns, dtype="float32")  # 1+num_ns
                    context = tf.squeeze(context, axis=-1)  # 1+num_ns
                    target_item = target_item[tf.newaxis]  # 1,
                    if model_type == 'GES' or 'EGES':
                        target_item_idx = int(target_item.numpy())
                        target_item_id = idx2item[target_item_idx]
                        target_item_cat_idx, target_item_brand_idx = get_side_info(item2brand, brand_vocab, item2cat,
                                                                                   cat_vocab,
                                                                                   target_item_id)
                        yield target_item, target_item_cat_idx, target_item_brand_idx, context, label
                    else:
                        yield target_item, context, label

    def test_example_generator():
        neg_pool = list(item2idx.keys())  # exclude 0, because 0 is oov
        for query_item, match_item in test_item_pairs:
            query_item_idx = [get_item_idx(item2idx, query_item)]
            match_item_idx = [get_item_idx(item2idx, match_item)]
            neg_item = random.choice(neg_pool)
            neg_item_idx = [get_item_idx(item2idx, neg_item)]

            if model_type == 'BGE':
                yield query_item_idx, match_item_idx, neg_item_idx
            elif model_type == 'GES' or model_type == 'EGES':
                query_item_cat_idx, query_item_brand_idx = get_side_info(item2brand, brand_vocab, item2cat, cat_vocab,
                                                                         query_item)
                match_item_cat_idx, match_item_brand_idx = get_side_info(item2brand, brand_vocab, item2cat, cat_vocab,
                                                                         match_item)
                neg_item_cat_idx, neg_item_brand_idx = get_side_info(item2brand, brand_vocab, item2cat, cat_vocab,
                                                                     neg_item)
                yield query_item_idx, query_item_cat_idx, query_item_brand_idx, match_item_idx, match_item_cat_idx, match_item_brand_idx, neg_item_idx, neg_item_cat_idx, neg_item_brand_idx

    if model_type == 'BGE':
        if mode == 'train':
            output_types = (tf.int32, tf.int32, tf.float32)
            output_shapes = (tf.TensorShape((1,)), tf.TensorShape((1 + num_ns,)), tf.TensorShape((1 + num_ns,)))
            dataset = tf.data.Dataset.from_generator(train_example_generator, output_types=output_types,
                                                     output_shapes=output_shapes)
        elif mode == 'test':
            output_types = (tf.int32, tf.int32, tf.float32)
            output_shapes = (tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((1,)))
            dataset = tf.data.Dataset.from_generator(test_example_generator, output_types=output_types,
                                                     output_shapes=output_shapes)
    elif model_type == 'GES' or model_type == 'EGES':
        if mode == 'train':
            output_types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.float32)
            output_shapes = (
                tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((1 + num_ns,)),
                tf.TensorShape((1 + num_ns,)))
            dataset = tf.data.Dataset.from_generator(train_example_generator, output_types=output_types,
                                                     output_shapes=output_shapes)
        elif mode == 'test':
            output_types = (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
            output_shapes = (
                tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((1,)),
                tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((1,)),
                tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((1,)),
            )
            dataset = tf.data.Dataset.from_generator(test_example_generator, output_types=output_types,
                                                     output_shapes=output_shapes)
    return dataset
