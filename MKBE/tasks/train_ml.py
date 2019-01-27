from models import ml_convE_kb as model
from input_pipeline import negative_sampling as ns
from input_pipeline import dataset_loader as dl

import Evaluation

import tensorflow as tf
import numpy as np


# subset can be ["movie_title_poster_user_rating", "movie_title_user_rating", "movie_title_rating", "movie_rating",
#                   "user_rating", "rating"]
fold = 1
subset = "movie_title_poster_user_rating"
experiment_name = "best_{:}".format(fold)

files_train = [
    "../assets/ml100k-processed/u{:}-{:}-train.npy".format(fold, subset),
    "../assets/ml100k-processed/u{:}-{:}-idencoder.npy".format(fold, subset),
    "../assets/ml100k-processed/u{:}-{:}-titles.npy".format(fold, subset),
    "../code/movielens/ml-100k/feature_maps.npy",
    "../assets/ml100k-processed/u{:}-{:}-title-dict.npy".format(fold, subset)
]

files_test = [
    "../assets/ml100k-processed/u{:}-{:}-test.npy".format(fold, subset),
    "../assets/ml100k-processed/u{:}-{:}-idencoder.npy".format(fold, subset),
    "../assets/ml100k-processed/u{:}-{:}-titles.npy".format(fold, subset),
    "../code/movielens/ml-100k/feature_maps.npy",
    "../assets/ml100k-processed/u{:}-{:}-title-dict.npy".format(fold, subset)
]

config = {
    "logdir": "/mnt/data/log"
}

summary_writer = tf.summary.FileWriter(config["logdir"] + "/train", max_queue=2, flush_secs=5)
test_writer = tf.summary.FileWriter(config["logdir"] + "/test", max_queue=2, flush_secs=5)

train_set = dl.Dataset(files_train)
test_set = dl.Dataset(files_test)

hyperparams = {
    "subset": subset,
    "dtype": tf.float16,
    "id_dtype": tf.int32,
    "emb_dim": 128,
    "MLPLayers": 3,
    "GRULayers": 3,
    "ksize": 5,
    "depth": 136,
    "drate": 2,
    "user_size": train_set.idencoders["maxuserid"] + 1,
    "movie_size": train_set.idencoders["maxmovieid"] + 1,
    "relation_size": len(train_set.idencoders["rel2id"]) + 1,
    "gender_size": len(train_set.idencoders["gender2id"]) + 1,
    "job_size": len(train_set.idencoders["job2id"]) + 1,
    "zip_size": len(train_set.idencoders["zip2id"]) + 1,
    "char_size": len(train_set.idencoders["char2id"]) + 1,
    "normalize_e1": False,
    "normalize_relation": True,
    "normalize_e2": True,
    "test_normalize_e1": False,
    "test_normalize_relation": True,
    "test_normalize_e2": True,
    "regularization_coefficient": 5e-6,
    "learning_rate": 0.7,
    "learning_rate_reduced": 0.0001,
    "margin": 1.5,
    "init_scale": 24,
    "down_scale": 120,
    "cnn_scale": 24,
    "max_epoch": 1800,
    "batch_size": 512,
    "bias": True,
    "debug": False,
    "activation": None
}

# hyperparams.update(np.load("scores/{:}/best_{:}_hyperparams.npy".format(subset, fold)).item())
# hyperparams["max_epoch"] = 1800
tf.set_random_seed(2048)

model_nodes = model.define_graph(hyperparams, {})

test_nodes = model.test_graph(hyperparams, model_nodes, config=None)

debug_nodes = model.debug_graph(hyperparams, test_nodes, config=None)

saver = tf.train.Saver()

highest_hits1_r, pos_scores_highest, neg_scores_highest = 0, None, None

# Configure Training environment
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
sess_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_2
with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    print("initialized, on subset {}!".format(subset))

    for epoch in range(hyperparams["max_epoch"]):
        batch = ns.aggregate_sampled_batch(ns.negative_sampling_aligned(
            train_set.next_batch(hyperparams["batch_size"]), None, train_set.idencoders,
            train_set.titles, train_set.poster_arr))

        # Feed dict for training
        feed = dl.build_feed(model_nodes, batch)

        [loss, summary, _, _] = sess.run(
            [model_nodes["loss_to_show"], model_nodes["training_summary"],
             model_nodes["training_op"], model_nodes["rlr_train_op"]
             ],
            feed_dict=feed)

        # Write summaries
        summary_writer.add_summary(summary, epoch)

        if epoch % 480 == 1:
            print("loss", loss)
            # Due to the limited accuracy of float32, HITS@5_r isn't exactly 100%
            metrics = ["MRR_movie", "HITS@10_movie", "HITS@3_movie", "HITS@1_movie", "MRR_r", "HITS@5_r", "HITS@3_r",
                       "HITS@2_r", "HITS@1_r"]
            metrics = ["MRR_r", "HITS@5_r", "HITS@3_r", "HITS@2_r", "HITS@1_r"]
            debug_names = ["pos_conv5_4_shape", "neg_conv5_4_shape", "pos_pool5_shape", "neg_pool5_shape",
                           "pos_poster_vec_shape", "neg_poster_vec_shape"] if hyperparams["debug"] else []
            points = []
            for offset in range(0, test_set.set_size, 256):
                batch = test_set.next_batch_inorder(256, offset)
                results = sess.run(
                    [test_nodes[m] for m in metrics] + [test_nodes["test_summary"]],
                    feed_dict=dl.build_feed_test(test_nodes, hyperparams, test_set.idencoders, batch)
                )
                debug_results = sess.run([model_nodes[n] for n in debug_names], feed_dict=feed)

                summary_writer.add_summary(results[-1], epoch)

                points.append(results[: -1])

            mean_m = np.mean(points, axis=0)

            print("Test metrics: ", ", ".join("{:}: {:.4f}".format(m, v) for m, v in zip(metrics, mean_m)))

            if hyperparams["debug"]:
                print("Debug nodes:", ", ".join("{:}: {:}".format(k, v) for k, v in zip(debug_names, debug_results)))

        if epoch % (train_set.set_size * 1.8 // hyperparams["batch_size"]) == 2:
            print("loss", loss)

            metrics = ["MRR_movie", "HITS@10_movie", "HITS@3_movie", "HITS@1_movie", "MRR_r", "HITS@5_r", "HITS@3_r",
                       "HITS@2_r", "HITS@1_r"]
            metrics = ["MRR_r", "HITS@5_r", "HITS@3_r", "HITS@2_r", "HITS@1_r"]

            positive_scores = []
            negative_scores = []

            for offset in range(0, test_set.set_size, 256):
                batch = test_set.next_batch_inorder(256, offset)
                neg_score, pos_score = sess.run(
                    [test_nodes["neg_scoring_rating"], test_nodes["pos_scoring"]],
                    feed_dict=dl.build_feed_test(test_nodes, hyperparams, test_set.idencoders, batch))

                positive_scores.append(pos_score)
                negative_scores.append(neg_score)

            sample_score_arr = np.concatenate(positive_scores)
            negative_score_arr = np.concatenate(negative_scores)

            hits1_r, eval_num = Evaluation.hits(negative_score_arr, sample_score_arr, False)
            print(epoch, " ".join("{:}: {:.4f}".format(name, num)
                                  for name, num in zip(["MRR", "hits1", "hits2", "hits3"], eval_num)))

            if hits1_r > highest_hits1_r:
                pos_scores_highest = sample_score_arr
                neg_scores_highest = negative_score_arr

                highest_hits1_r = hits1_r

    save_path = saver.save(sess, "../assets/weights/u{:}-convE.ckpt".format(fold))