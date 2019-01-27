from collections import defaultdict

import numpy as np
import msgpack, msgpack_numpy, os, lmdb

msgpack_numpy.patch()

in_files = {
    "train": "../code/YAGO/data/YAGO3-10/train.txt",
    "test": "../code/YAGO/data/YAGO3-10/test.txt",
    "valid": "../code/YAGO/data/YAGO3-10/valid.txt",
    "numerical": "../code/YAGO/Multi-Model/data/num.txt",
    "text": "../code/YAGO/Multi-Model/data/text.txt"
}

out_files = {
    "train_S": "../assets/yago-processed/train_s.mdb",
    "train_N": "../assets/yago-processed/train_n.mdb",
    "train_I": "../assets/yago-processed/train_i.mdb",
    "train_D": "../assets/yago-processed/train_d.mdb",
    "test": "../assets/yago-processed/test.mdb",
    "meta": "../assets/yago-processed/meta.npy"
}


class LMDB_Writer:
    def __init__(self, path, write_frequency=1024):
        isdir = os.path.isdir(path)
        self.lmdb_env = lmdb.open(
            path, subdir=isdir, map_size=1099511627776 * 2, readonly=False, meminit=False, map_async=True)

        self.txn = self.lmdb_env.begin(write=True)
        self.write_f = write_frequency
        self.counter = 0

    def write_kv(self, k_bytes, v_bytes):
        self.txn.put(k_bytes, v_bytes)
        self.counter += 1

        if self.counter % self.write_f == 0:
            self.txn.commit()
            self.txn = self.lmdb_env.begin(write=True)

    def close(self):
        self.txn.commit()
        self.lmdb_env.sync()
        self.lmdb_env.close()

    def __del__(self):
        try:
            self.txn.commit()
            self.lmdb_env.sync()
            self.lmdb_env.close()
        except:
            pass


def parseline(line):
    return [s.strip() for s in line.split("\t")]


def read_filter_dict(infiles):
    entity2id = {}
    relation2id = {}
    entity_counter = 0
    relation_counter = 0

    unreadable_lines = []
    # build dictionary for relational triple
    for sub in ["train", "valid", "test"]:
        with open(infiles[sub], encoding="latin") as file:
            for linenum, line in enumerate(file):
                triplet = parseline(line)
                if len(triplet) != 3:
                    unreadable_lines.append((linenum, line))
                else:
                    e1, r, e2 = triplet
                    r_rev = r + "_reverse"

                    if e1 not in entity2id:
                        entity_counter += 1
                        entity2id[e1] = entity_counter

                    if e2 not in entity2id:
                        entity_counter += 1
                        entity2id[e2] = entity_counter

                    if r not in relation2id:
                        relation_counter += 1
                        relation2id[r] = relation_counter

                    if r_rev not in relation2id:
                        relation_counter += 1
                        relation2id[r_rev] = relation_counter

    print("Unreadable lines:", len(unreadable_lines))

    # Normalize numerical values
    num_raw = []
    with open(infiles["numerical"], encoding="utf-8") as file:
        for line in file:
            e1, r, e2 = parseline(line)
            num_raw.append(int(e2))
            if r not in relation2id:
                relation_counter += 1
                relation2id[r] = relation_counter

    scale_params = {
        "mean": np.mean(num_raw),
        "std": np.std(num_raw)
    }
    e2_num = np.array(num_raw)
    e2_num = np.subtract(e2_num, scale_params["mean"])
    e2_num /= scale_params["std"]

    # Encode text
    words = set()
    with open(infiles["text"], encoding="utf-8") as file:
        for line in file:
            e1, e2 = parseline(line)
            words.update(e2.split())

    words.update(["<go>", "<eos>"])
    word2id = dict(zip(words, range(len(words))))

    relation_counter += 1
    relation2id['bio'] = relation_counter

    meta = {
        "entity2id": entity2id,
        "rel2id": relation2id,
        "word2id": word2id,
        "maxentityid": max(entity2id.values()),
        "scale_params": scale_params
    }

    return meta


def encode_int_kv_tobyte(int_k, record):
    byte_k = u"{0:0>10}".format(int_k).encode("utf-8")
    byte_v = msgpack.dumps(record)
    return byte_k, byte_v


def encode_store_S(infiles, outfiles, meta):
    triplets = defaultdict(dict)
    writer = LMDB_Writer(outfiles["train_S"])
    e2id = meta["entity2id"]
    r2id = meta["rel2id"]

    with open(infiles["train"], encoding="latin") as file:
        counter = 0
        for line in file:
            splits = parseline(line)
            if len(splits) == 3:
                counter += 2
                raw_e1, raw_r, raw_e2 = parseline(line)
                e1, r, e2, r_rev = e2id[raw_e1], r2id[raw_r], e2id[raw_e2], r2id[raw_r + "_reverse"]

                if r not in triplets[e1]:
                    triplets[e1][r] = [e2]
                else:
                    triplets[e1][r].append(e2)

                if r_rev not in triplets[e2]:
                    triplets[e2][r_rev] = [e1]
                else:
                    triplets[e2][r_rev].append(e1)

        meta["train_size"] = counter

    for e1, record in triplets.items():
        byte_k, byte_v = encode_int_kv_tobyte(e1, record)
        writer.write_kv(byte_k, byte_v)
    writer.close()

    return meta


def encode_store_test_S(infiles, outfiles, meta):
    triplets = defaultdict(dict)
    writer = LMDB_Writer(outfiles["test"])
    e2id = meta["entity2id"]
    r2id = meta["rel2id"]

    with open(infiles["test"], encoding="latin") as file:
        counter = 0
        for line in file:
            counter += 1
            raw_e1, raw_r, raw_e2 = parseline(line)
            e1, r, e2 = e2id[raw_e1], r2id[raw_r], e2id[raw_e2]
            if r not in triplets[e1]:
                triplets[e1][r] = [e2]
            else:
                triplets[e1][r].append(e2)
        meta["test_size"] = counter

    for e1, record in triplets.items():
        byte_k, byte_v = encode_int_kv_tobyte(e1, record)
        writer.write_kv(byte_k, byte_v)
    writer.close()

    return meta


meta = read_filter_dict(in_files)
meta = encode_store_S(in_files, out_files, meta)
meta = encode_store_test_S(in_files, out_files, meta)
np.save(out_files["meta"], meta)
print(meta["train_size"], meta["test_size"])
