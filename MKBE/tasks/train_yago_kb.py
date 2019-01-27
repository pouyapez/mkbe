from input_pipeline.yago_input_pipeline import train_dataflow, test_dataflow, profiling_dataflow, profiling_test_df
from models.yago_convE_kb_model import YAGOConveMultimodel
from train.yago_training import SingleGPUTrainer
from test.test_runner import TestRunner
from tensorpack import *
import numpy as np
import tensorflow as tf


files = {
    "train_S": "../assets/yago-processed/train_s.mdb",
    "train_N": "../assets/yago-processed/train_n.mdb",
    "train_I": "../assets/yago-processed/train_i.mdb",
    "train_D": "../assets/yago-processed/train_d.mdb",
    "test": "../assets/yago-processed/test.mdb",
    "meta": "../assets/yago-processed/meta.npy"
}


meta = np.load(files["meta"]).item(0)

hyperparams = {
    "dtype": tf.float32,
    "id_dtype": tf.int32,
    "emb_dim": 200,
    "MLPLayers": 2,
    "GRULayers": 2,
    "CNNTextLayers": 2,
    "CNNTextDilation": 2,
    "CNNTextKernel": 4,
    "entity_size": meta["maxentityid"] + 1,
    "relation_size": len(meta["rel2id"]) + 1,
    "word_size": len(meta["word2id"]) + 1,
    "normalize_e1": False,
    "normalize_relation": False,
    "normalize_e2": False,
    "test_normalize_e1": False,
    "test_normalize_relation": False,
    "test_normalize_e2": False,
    "regularization_coefficient": 0.0,
    "learning_rate": 0.003,
    "lr_decay": 0.995,
    "label_smoothing": 0.1,
    "batch_size": 256,
    "bias": False,
    "debug": False,
    "emb_keepprob": 0.8,
    "fm_keepprob": 0.8,
    "mlp_keepprob": 0.7,
    "enc_keepprob": 0.9
}

utils.logger.set_logger_dir("./logs", action="d")

cbs = [
    PeriodicCallback(TensorPrinter(["loss", "lr"]), every_k_steps=1000),
    TestRunner(
        test_dataflow(files["test"], files["meta"], 32),
        [ScalarStats("mrr"), ScalarStats("hits_1"), ScalarStats("hits_3"), ScalarStats("hits_10"),
         ScalarStats("label_smoothing"), ScalarStats("inv_e2")])
]

monitors = [
    callbacks.ScalarPrinter(),
    callbacks.JSONWriter(),
    TFEventWriter(logdir="/mnt/data/log", max_queue=5, flush_secs=2)
]

cfg = TrainConfig(
    model=YAGOConveMultimodel(hyperparams),
    data=train_dataflow(files["train_S"], files["meta"], hyperparams["batch_size"], 300),
    max_epoch=200,
    steps_per_epoch=meta["train_size"] // hyperparams["batch_size"],
    monitors=monitors,
    callbacks=cbs
)

trainer = SingleGPUTrainer(hyperparams)

launch_train_with_config(cfg, trainer)
