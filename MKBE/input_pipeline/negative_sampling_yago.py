import numpy as np


def negative_sampling_aligned(batch, hyperparams, idenc, texts):
    # Negative sampling: randomly choose an entity in the dictionary for categorical data,
    # or sample from a normal distribution for real numbers
    e1, r, e2 = batch
    rel2id = idenc["rel2id"]

    # Extract num strips
    idx_num = np.where(
        (r == rel2id["happenedOnDate"]) | (r == rel2id["wasBornOnDate"]) | (r == rel2id["diedOnDate"]) | (
        r == rel2id["wasCreatedOnDate"]) | (r == rel2id["wasDestroyedOnDate"]))[0]
    pos_num = e1[idx_num].astype(np.int32), r[idx_num].astype(np.int32), e2[idx_num].astype(np.float32)
    neg_num = e1[idx_num].astype(np.int32), r[idx_num].astype(np.int32), \
              np.random.normal(size=len(idx_num)).astype(np.float32)

    # Extract relational triplet
    idx_triplet = np.where(r <= 37)[0]
    corrupted_e2 = np.random.choice(len(idenc["entity2id"]), size=len(idx_triplet))
    pos_triplet = e1[idx_triplet].astype(np.int32), r[idx_triplet].astype(np.int32), \
                  e2[idx_triplet].astype(np.int32)
    neg_triplet = e1[idx_triplet].astype(np.int32), r[idx_triplet].astype(np.int32), \
                  corrupted_e2.astype(np.int32)

    # Extract text
    idx_text = np.where(r == rel2id["bio"])[0]

    if len(idx_text) > 0:
        corrupted_e2 = np.random.choice(texts, size=len(idx_text))
        pos_len = np.array([len(line) for line in e2[idx_text]], dtype=np.int32)
        neg_len = np.array([len(line) for line in corrupted_e2], dtype=np.int32)
        max_pos_len = max(pos_len)
        max_neg_len = max(neg_len)

        pos_e2 = np.array([line + [0] * (max_pos_len - len(line)) for line in e2[idx_text]], dtype=np.int32)
        neg_e2 = np.array([line + [0] * (max_neg_len - len(line)) for line in corrupted_e2], dtype=np.int32)

        pos_text = e1[idx_text], r[idx_text], pos_e2, pos_len
        neg_text = e1[idx_text], r[idx_text], neg_e2, neg_len
    else:
        ept_a = np.zeros((0, 20), dtype=np.int32)
        ept_b = np.array([], dtype=np.int32)
        pos_text = ept_b, ept_b, ept_a, np.zeros((0,), dtype=np.int32)
        neg_text = ept_b, ept_b, ept_a, np.zeros((0,), dtype=np.int32)

    ###### Extract image

    return pos_num, neg_num, pos_triplet, neg_triplet, pos_text, neg_text


def aggregate_sampled_batch(batch):
    pos_num, neg_num, pos_triplet, neg_triplet, pos_text, neg_text = batch

    pos_e1 = np.concatenate([batch[idx][0] for idx in range(0, 5, 2)], axis=0).astype(np.int32)
    pos_r = np.concatenate([batch[idx][1] for idx in range(0, 5, 2)], axis=0).astype(np.int32)
    pos_e2 = pos_triplet[2]

    neg_e1 = np.concatenate([batch[idx][0] for idx in range(1, 6, 2)], axis=0).astype(np.int32)
    neg_r = np.concatenate([batch[idx][1] for idx in range(1, 6, 2)], axis=0).astype(np.int32)
    neg_e2 = neg_triplet[2]

    pos_num = pos_num[2]
    neg_num = neg_num[2]

    return {
        "pos_e1": pos_e1,
        "pos_r": pos_r,
        "pos_e2": pos_e2,
        "neg_e1": neg_e1,
        "neg_r": neg_r,
        "neg_e2": neg_e2,
        "pos_num": pos_num,
        "neg_num": neg_num,
        "pos_text": pos_text[2],
        "neg_text": neg_text[2],
        "pos_text_len": pos_text[3],
        "neg_text_len": neg_text[3]

    }
