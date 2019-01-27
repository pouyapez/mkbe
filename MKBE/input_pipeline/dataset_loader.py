# Relations used: age, gender, occupation, zip, title, release date, genre, rating(1-5)
import numpy as np


class Dataset:
    def __init__(self, files, setname="train"):
        setfile, encfile, titles, posters, title_dict = files

        setarr = np.load(setfile)
        self.idencoders = np.load(encfile).reshape((1))[0]
        self.titles = np.load(titles)
        self.title_dict = np.load(title_dict).item()
        self.posters = np.load(posters).item()
        self.poster_arr = np.concatenate(list(self.posters.values()), axis=0)
        self.users = self.idencoders["maxuserid"]
        self.movies = self.idencoders["maxmovieid"]

        self.title_keys = list(self.title_dict.keys())

        self.e1 = setarr[:, 0]
        self.r = setarr[:, 1]
        self.e2 = setarr[:, 2]
        self.set_size = self.e1.shape[0]

    def next_batch(self, batch_size):
        idx = np.random.randint(self.set_size, size=batch_size)
        return self.e1[idx], self.r[idx], self.e2[idx]

    def next_batch_inorder(self, batch_size, offset):
        end = offset + batch_size
        return self.e1[offset:end], self.r[offset:end], self.e2[offset:end]

    def title_triplets(self, batch_size):
        e1 = np.random.choice(self.title_keys, batch_size)
        r = np.array([self.idencoders["rel2id"]["is of_title"]] * batch_size, dtype=np.int)
        e2 = np.array([self.title_dict[n] for n in e1], dtype=list)



def build_feed(nodes, batch):
    params = {
        "emb_keepprob:0": 0.77,
        "fm_keepprob:0": 0.77,
        "mlp_keepprob:0": 0.6,
        "enc_keepprob:0": 0.9,
        "is_training:0": True
    }
    feeds = dict((nodes[k], batch[k]) for k in batch.keys())
    feeds.update(params)
    return feeds


def build_feed_test(nodes, hyperparams, idenc, batch):
    return {
        nodes["rating_relations"]:
            np.array([v for k, v in idenc["rel2id"].items() if "rate" in k],
                     dtype=np.int32),
        nodes["pos_user"]: batch[0].astype(np.int32),
        nodes["pos_r"]: batch[1].astype(np.int32),
        nodes["pos_movie"]: batch[2].astype(np.int32)
    }
