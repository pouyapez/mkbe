from tensorpack import *
from tensorpack.dataflow import *
from input_pipeline.yago_lmdb_loader import LoaderS, TestLoaderDataflow


def sparse_to_dense(datapoint):
    e1, r, e2_train, e2_test = datapoint
    return e1, r, e2_train.toarray(), e2_test


def train_dataflow(s_file, idenc_file, batch_size, epoch, gpu_list=None):
    s_loader = LoaderS(s_file, idenc_file)
    df_in = DataFromGenerator(s_loader.gen_batch(batch_size, epoch))
    df_multi = MultiProcessPrefetchData(df_in, 96, 10)
    df_map = MapData(df_multi, sparse_to_dense)
    ins = input_source.QueueInput(df_map)
    gpu_ins = input_source.StagingInput(ins, [0] if gpu_list is None else gpu_list)
    return gpu_ins


def test_dataflow(s_file, idenc_file, batch_size, gpu_list=None):
    df_in = TestLoaderDataflow(s_file, idenc_file)
    #df_in = dataflow.DataFromGenerator(s_loader.gen_sample_inorder())
    df_batched = dataflow.BatchData(df_in, batch_size, remainder=True)
    ins = FeedInput(df_batched, infinite=False)
    return ins


def profiling_dataflow(s_file, idenc_file, batch_size, epoch, gpu_list=None):
    s_loader = LoaderS(s_file, idenc_file)
    df_in = DataFromGenerator(s_loader.gen_batch(batch_size, epoch))
    df_multi = MultiProcessPrefetchData(df_in, 96, 10)
    df_map = MapData(df_multi, sparse_to_dense)
    df_test = TestDataSpeed(df_map)

    return df_test


def profiling_test_df(s_file, idenc_file, batch_size, gpu_list=None):
    df_in = TestLoaderDataflow(s_file, idenc_file)
    #df_in = dataflow.DataFromGenerator(s_loader.gen_sample_inorder())
    df_batched = dataflow.BatchData(df_in, batch_size, remainder=True)
    df_test = TestDataSpeed(df_batched)
    return df_test