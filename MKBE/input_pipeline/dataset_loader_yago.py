# Relations used: 0-37, numerical, bio, image
import numpy as np


class Dataset:
    def __init__(self, files, setname="train"):
        setfile, encfile, texts = files

        setarr = np.load(setfile, encoding="latin1")
        self.idencoders = np.load(encfile, encoding="latin1").reshape((1))[0]
        self.texts = np.load(texts, encoding="latin1")
        self.entity = self.idencoders["maxentityid"]

        self.e1 = setarr[:, 0]
        self.r = setarr[:, 1]
        self.e2 = setarr[:, 2]
        self.set_size = self.e1.shape[0]
        print(self.set_size)

    def next_batch(self, batch_size, next_img_set=False):
        idx = np.random.randint(self.set_size, size=batch_size)
        return self.e1[idx], self.r[idx], self.e2[idx]

    def next_batch_inorder(self, batch_size, offset):
        end = offset + batch_size
        return self.e1[offset:end], self.r[offset:end], self.e2[offset:end]


class AddImg:
    def __init__(self, idenc, bsize):
        self.fm_files = ["YAGO-processed/feature_maps_{}.npy".format(n) for n in range(4)]
        self.fm_dict = {}
        self.fm_counter = 0
        self.load_new_fm()
        self.bsize = bsize // 2
        self.names = []
        self.set_size = 0

    def load_new_fm(self):
        self.fm_counter = (self.fm_counter + 1) % len(self.fm_files)
        self.fm_dict = np.load(self.fm_files[self.fm_counter]).item()


def build_feed(nodes, batch):
    params = {
        "emb_keepprob:0": 0.77,
        "fm_keepprob:0": 0.77,
        "mlp_keepprob:0": 0.6,
        "enc_keepprob:0": 0.9
    }
    feeds = dict((nodes[k], batch[k]) for k in batch.keys())
    feeds.update(params)
    return feeds


def build_feed_test(nodes, hyperparams, idenc, batch):
    return {
        #  nodes["rating_relations"]:
        #      np.array([v for k, v in idenc["rel2id"].items() if k <= 37],
        #               dtype=np.int32),
        nodes["pos_e1"]: batch[0].astype(np.int32),
        nodes["pos_r"]: batch[1].astype(np.int32),
        nodes["pos_e2"]: batch[2].astype(np.int32)
    }
