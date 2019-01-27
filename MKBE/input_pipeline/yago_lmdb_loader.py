import numpy as np
import msgpack, msgpack_numpy, lmdb, os
from scipy import sparse
from tensorpack import *

msgpack_numpy.patch()


def decode_key(byte_k):
    return int(str(byte_k, encoding="utf-8"))


def encode_key(int_k):
    return u"{0:0>10}".format(int_k).encode("UTF-8")


class LoaderS:
    def __init__(self, path, meta_path):
        isdir = os.path.isdir(path)
        self.lmdb_env = lmdb.open(
            path, subdir=isdir, readonly=True, lock=False, readahead=True, map_size=1099511627776 * 2, max_readers=100)
        self.txn = self.lmdb_env.begin()

        self.meta = np.load(meta_path).item(0)
        self.id2entity = dict((v, k) for k, v in self.meta["entity2id"].items())
        self.id2rel = dict((v, k) for k, v in self.meta["rel2id"].items())
        self.max_entity = max(self.id2entity.keys())

    def index_str_by_str(self, k):
        byte_k = encode_key(self.meta["entity2id"][k])
        with self.txn.cursor() as cur:
            record = msgpack.loads(cur.get(byte_k))

        return dict((self.id2rel[r], self.id2entity[e2]) for r, e2 in record.items())

    def index_int_by_int(self, k):
        byte_k = encode_key(k)
        with self.txn.cursor() as cur:
            record_bytes = cur.get(byte_k)
            record = msgpack.loads(record_bytes) if record_bytes is not None else None

        return record

    def gen_batch(self, batch_size, epoch=3):
        for _ in range(self.meta["train_size"] * epoch // batch_size):
            triplets = []
            ks = np.random.randint(1, self.max_entity + 1, batch_size * 2)
            for e1 in ks:
                record = self.index_int_by_int(e1)
                if record is not None:
                    for r, e2_l in record.items():
                        # e2_onehot = np.zeros((1, self.meta["maxentityid"] + 1), dtype=np.float16)
                        rows = np.array([0] * len(e2_l))
                        col = np.array(e2_l)
                        data = np.array([1] * len(e2_l), dtype=np.int8)
                        e2_onehot = sparse.csr_matrix((data, (rows, col)), shape=(1, self.meta["maxentityid"] + 1))
                        triplets.append((e1, r, e2_onehot))

            batch_idx = np.random.choice(len(triplets), batch_size)
            e1 = list(triplets[idx][0] for idx in batch_idx)
            r = list(triplets[idx][1] for idx in batch_idx)
            e2 = list(triplets[idx][2] for idx in batch_idx)
            e2_test = np.zeros((batch_size), dtype=np.int32)

            yield np.array(e1, dtype=np.int32), np.array(r, dtype=np.int32), sparse.vstack(e2), e2_test

    def gen_sample_inorder(self):
        with self.txn.cursor() as cur:
            for k, record_byte in cur:
                record = msgpack.loads(record_byte)
                e1 = decode_key(k)
                for r, e2_l in record.items():
                    for e2 in e2_l:
                        e2_onehot = np.zeros((self.meta["maxentityid"] + 1,), dtype=np.int8)
                        e2_onehot[e2] = 1
                        yield e1, r, e2_onehot, e2


class TestLoaderDataflow(DataFlow, LoaderS):
    def __init__(self, path, meta_path):
        LoaderS.__init__(self, path, meta_path)
        DataFlow.__init__(self)

    def reset_state(self):
        self.gen = iter(self.gen_sample_inorder())

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def get_data(self):
        self.reset_state()
        return self

    def size(self):
        return self.meta["test_size"]