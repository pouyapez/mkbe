import numpy as np
import tensorflow as tf

import input_pipeline.dataset_loader_yago as dl
import models.yago_convE_kb as model
import input_pipeline.negative_sampling_yago as ns


# subset can be ["id", "text_id", "num_id", "image_id", "image_num_id", "image_text_id", "text_num_id", "image_text_num_id"]
subset = "text_id"
experiment_name = "suboptimal"
#test_subset = "hasGender"

files_train = [
    "../code/YAGO/Multi-Model/YAGO-processed/train.npy",
    "../code/YAGO/Multi-Model/YAGO-processed/idencoder.npy",
    "../code/YAGO/Multi-Model/YAGO-processed/texts.npy"
]

files_test = [
    "../code/YAGO/Multi-Model/YAGO-processed/test.npy",
    "../code/YAGO/Multi-Model/YAGO-processed/idencoder.npy",
    "../code/YAGO/Multi-Model/YAGO-processed/texts.npy"
]

config = {
    "logdir": "/mnt/data/log"
}

#summary_writer = tf.summary.FileWriter(config["logdir"] + "/train", max_queue=2, flush_secs=5)
#test_writer = tf.summary.FileWriter(config["logdir"] + "/test", max_queue=2, flush_secs=5)

train_set = dl.Dataset(files_train)
test_set = dl.Dataset(files_test)

hyperparams = {
    "dtype": tf.float32,
    "id_dtype": tf.int32,
    "emb_dim": 256,
    "MLPLayers": 2,
    "GRULayers": 2,
    "CNNTextLayers": 2,
    "CNNTextDilation": 2,
    "CNNTextKernel": 4,
    "entity_size": train_set.idencoders["maxentityid"] + 1,
    "relation_size": len(train_set.idencoders["rel2id"]) + 1,
    "word_size": len(train_set.idencoders["word2id"]) + 1, 
    "normalize_e1": False,
    "normalize_relation": True,
    "normalize_e2": True,
    "test_normalize_e1": False,
    "test_normalize_relation": True,
    "test_normalize_e2": True,
    "regularization_coefficient": 0.000001,
    "learning_rate": 0.03,
    "learning_rate_reduced": 7e-5,
    "margin": 1.5,
    "label_smoothing": 0.1,
    "init_scale": 36,
    "cnn_scale": 6,
    "max_epoch": 10000,
    "batch_size": 1024,
    "bias": False,
    "debug": False,
    "emb_keepprob": 0.77,
    "fm_keepprob": 0.77,
    "mlp_keepprob": 0.6,
    "enc_keepprob": 0.9
}

model_nodes = model.define_graph(hyperparams)

test_nodes = model.test_graph(hyperparams, model_nodes, config=None)

debug_nodes = model.debug_graph(hyperparams, test_nodes, config=None)

highest_hits1_r, pos_scores_highest, neg_scores_highest = 0, None, None

# Configure Training environment
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    sess.run(tf.global_variables_initializer())
    print("initialized")

    for epoch in range(hyperparams["max_epoch"]):
        batch = ns.aggregate_sampled_batch(ns.negative_sampling_aligned(
            train_set.next_batch(hyperparams["batch_size"]), None, train_set.idencoders, train_set.texts))

        test_batch = ns.aggregate_sampled_batch(ns.negative_sampling_aligned(
            test_set.next_batch(hyperparams["batch_size"] // 32), None, test_set.idencoders, test_set.texts))

        # Feed dict for training
        feed = dl.build_feed(model_nodes, batch)

        test_feed = dl.build_feed(model_nodes, test_batch)
        _, _, loss = sess.run(
            [model_nodes["training_op"], model_nodes["rlr_train_op"], model_nodes["loss_to_show"]],
            feed_dict=feed)
        test_summary = sess.run(
            model_nodes["training_summary"], feed_dict=test_feed)

        # Write summaries
        # summary_writer.add_summary(summary, epoch)
        # test_writer.add_summary(test_summary, epoch)

        if epoch % 20 == 1:
            print("Training Loss:", loss)