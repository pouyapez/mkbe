import numpy as np


def negative_sampling_aligned(batch, hyperparams, idenc, titles, poster_arr):
    # Negative sampling: randomly choose an entity in the dictionary for categorical data,
    # or sample from a normal distribution for real numbers
    af = "is of_"
    e1, r, e2 = batch
    rel2id = idenc["rel2id"]

    # Extract age strips
    idx_age = np.where(r == rel2id[af + "age"])[0]
    pos_age = e1[idx_age].astype(np.int32), r[idx_age].astype(np.int32), e2[idx_age].astype(np.float32)
    neg_age = e1[idx_age].astype(np.int32), r[idx_age].astype(np.int32), \
              np.random.normal(size=len(idx_age)).astype(np.float32)

    # Extract gender strips
    idx_gender = np.where(r == rel2id[af + "gender"])[0]
    pos_gender = e1[idx_gender].astype(np.int32), r[idx_gender].astype(np.int32), e2[idx_gender].astype(np.int32)
    neg_gender = e1[idx_gender].astype(np.int32), r[idx_gender].astype(np.int32), 1 - e2[idx_gender].astype(np.int32)

    # Extract occupation
    idx_occupation = np.where(r == rel2id[af + "occupation"])[0]
    corrupted_e2 = np.random.choice(len(idenc["job2id"]), size=len(idx_occupation))
    pos_occupation = e1[idx_occupation].astype(np.int32), r[idx_occupation].astype(np.int32), \
                     e2[idx_occupation].astype(np.int32)
    neg_occupation = e1[idx_occupation].astype(np.int32), r[idx_occupation].astype(np.int32), \
                     corrupted_e2.astype(np.int32)

    # Extract zip
    idx_zip = np.where(r == rel2id[af + "zip"])[0]
    corrupted_e2 = np.random.choice(len(idenc["zip2id"]), size=len(idx_zip))
    pos_zip = e1[idx_zip].astype(np.int32), r[idx_zip].astype(np.int32), e2[idx_zip].astype(np.int32)
    neg_zip = e1[idx_zip].astype(np.int32), r[idx_zip].astype(np.int32), corrupted_e2.astype(np.int32)

    # Extract title
    idx_title = np.where(r == rel2id[af + "title"])[0]

    if len(idx_title) > 0:
        corrupted_e2 = np.random.choice(titles, size=len(idx_title))
        pos_len = np.array([len(line) for line in e2[idx_title]], dtype=np.int32)
        neg_len = np.array([len(line) for line in corrupted_e2], dtype=np.int32)
        max_pos_len = max(pos_len)
        max_neg_len = max(neg_len)

        pos_e2 = np.array([line + [0] * (max_pos_len - len(line)) for line in e2[idx_title]], dtype=np.int32)
        neg_e2 = np.array([line + [0] * (max_neg_len - len(line)) for line in corrupted_e2], dtype=np.int32)

        pos_title = e1[idx_title], r[idx_title], pos_e2, pos_len
        neg_title = e1[idx_title], r[idx_title], neg_e2, neg_len
    else:
        ept_a = np.zeros((0, 20), dtype=np.int32)
        ept_b = np.array([], dtype=np.int32)
        pos_title = ept_b, ept_b, ept_a, np.zeros((0,), dtype=np.int32)
        neg_title = ept_b, ept_b, ept_a, np.zeros((0,), dtype=np.int32)

    # Extract Poster
    idx_poster = np.where(r == rel2id[af + "poster"])[0]

    if len(idx_poster) > 0:
        corrupted_e2_idx = np.random.choice(len(poster_arr), size=len(idx_poster))
        corrupted_e2 = poster_arr[corrupted_e2_idx, :, :, :]
        pos_poster = e1[idx_poster].astype(np.int32), r[idx_poster].astype(np.int32), np.concatenate(e2[idx_poster], axis=0)
        neg_poster = e1[idx_poster].astype(np.int32), r[idx_poster].astype(np.int32), corrupted_e2
    else:
        ept_b = np.array([], dtype=np.int32)
        pos_poster = ept_b, ept_b, np.array([[[[]]]], dtype=np.float32).reshape((-1, 16, 16, 512))
        neg_poster = ept_b, ept_b, np.array([[[[]]]], dtype=np.float32).reshape((-1, 16, 16, 512))

    # Extract release date
    idx_date = np.where(r == rel2id[af + "release date"])[0]
    pos_date = e1[idx_date].astype(np.int32), r[idx_date].astype(np.int32), e2[idx_date].astype(np.float32)
    neg_date = e1[idx_date].astype(np.int32), r[idx_date].astype(np.int32), \
               np.random.normal(size=len(idx_date)).astype(np.float32)

    # Extract genre
    idx_genre = np.where(r == rel2id[af + "genre"])[0]
    if len(idx_genre) > 0:
        pos_e2 = np.concatenate([np.expand_dims(e2[idx], axis=0) for idx in idx_genre], axis=0).astype(np.float32)
        pos_genre = e1[idx_genre].astype(np.int32), r[idx_genre].astype(np.int32), pos_e2
        neg_genre = e1[idx_genre].astype(np.int32), r[idx_genre].astype(np.int32), 1 - pos_e2
    else:
        ept_b = np.array([], dtype=np.int32)
        pos_genre = ept_b, ept_b, np.array([[]], dtype=np.float32)
        neg_genre = ept_b, ept_b, np.array([[]], dtype=np.float32)

    # Extract ratings
    pos_rating_list = []
    neg_rating_list = []

    # Negative sampling for ratings
    for rating in range(1, 6):
        idx_rating = np.where(r == rel2id["rate_{:}".format(rating)])[0]
        corrupted_r = np.array([rel2id["rate_{:}".format(r)] for r in range(1, 6) if r != rating] * len(idx_rating),
                               dtype=np.int32)
        pos_rating = np.tile(e1[idx_rating], 4), np.tile(r[idx_rating], 4), np.tile(e2[idx_rating], 4)
        neg_rating = np.tile(e1[idx_rating], 4), corrupted_r, np.tile(e2[idx_rating], 4)
        pos_rating_list.append(pos_rating)
        neg_rating_list.append(neg_rating)

    # Negative sampling for movies
    idx_rating = np.where(np.logical_or.reduce([r == rel2id["rate_{:}".format(rating)] for rating in range(1, 6)]))[0]
    corrupted_e2 = np.random.choice(idenc["maxmovieid"], size=len(idx_rating))
    pos_rating = e1[idx_rating], r[idx_rating], e2[idx_rating]
    neg_rating = e1[idx_rating], r[idx_rating], corrupted_e2
    pos_rating_list.append(pos_rating)
    neg_rating_list.append(neg_rating)

    pos_rating = np.concatenate([line[0] for line in pos_rating_list], axis=0).astype(np.int32), \
                 np.concatenate([line[1] for line in pos_rating_list], axis=0).astype(np.int32), \
                 np.concatenate([line[2] for line in pos_rating_list], axis=0).astype(np.int32)
    neg_rating = np.concatenate([line[0] for line in neg_rating_list], axis=0).astype(np.int32), \
                 np.concatenate([line[1] for line in neg_rating_list], axis=0).astype(np.int32), \
                 np.concatenate([line[2] for line in neg_rating_list], axis=0).astype(np.int32)

    return pos_age, neg_age, pos_gender, neg_gender, pos_occupation, neg_occupation, pos_zip, neg_zip, pos_title, \
           neg_title, pos_date, neg_date, pos_genre, neg_genre, pos_rating, neg_rating, pos_poster, neg_poster


def aggregate_sampled_batch(batch):
    pos_age, neg_age, pos_gender, neg_gender, pos_occupation, neg_occupation, pos_zip, neg_zip, pos_title, \
    neg_title, pos_date, neg_date, pos_genre, neg_genre, pos_movierating, neg_movierating, pos_poster, \
    neg_poster = batch

    pos_user_e1 = np.concatenate([batch[idx][0] for idx in range(0, 7, 2)], axis=0).astype(np.int32)
    pos_user_r = np.concatenate([batch[idx][1] for idx in range(0, 7, 2)], axis=0).astype(np.int32)

    neg_user_e1 = np.concatenate([batch[idx][0] for idx in range(1, 8, 2)], axis=0).astype(np.int32)
    neg_user_r = np.concatenate([batch[idx][1] for idx in range(1, 8, 2)], axis=0).astype(np.int32)

    pos_movie_e1 = np.concatenate([batch[idx][0] for idx in range(8, 13, 2)], axis=0).astype(np.int32)
    pos_movie_r = np.concatenate([batch[idx][1] for idx in range(8, 13, 2)], axis=0).astype(np.int32)

    neg_movie_e1 = np.concatenate([batch[idx][0] for idx in range(9, 14, 2)], axis=0).astype(np.int32)
    neg_movie_r = np.concatenate([batch[idx][1] for idx in range(9, 14, 2)], axis=0).astype(np.int32)

    pos_userrating = pos_movierating[0]
    pos_relrating = pos_movierating[1]
    pos_ratedmovie = pos_movierating[2]

    neg_userrating = neg_movierating[0]
    neg_relrating = neg_movierating[1]
    neg_ratedmovie = neg_movierating[2]

    pos_poster_movie, pos_poster_rel, pos_poster_fm = pos_poster
    neg_poster_movie, neg_poster_rel, neg_poster_fm = neg_poster


    return {
        "pos_user_e1": pos_user_e1,
        "pos_user_r": pos_user_r,
        "neg_user_e1": neg_user_e1,
        "neg_user_r": neg_user_r,
        "pos_movie_e1": pos_movie_e1,
        "pos_movie_r": pos_movie_r,
        "neg_movie_e1": neg_movie_e1,
        "neg_movie_r": neg_movie_r,
        "pos_age": pos_age[2],
        "neg_age": neg_age[2],
        "pos_gender": pos_gender[2],
        "neg_gender": neg_gender[2],
        "pos_occupation": pos_occupation[2],
        "neg_occupation": neg_occupation[2],
        "pos_zip": pos_zip[2],
        "neg_zip": neg_zip[2],
        "pos_title": pos_title[2],
        "neg_title": neg_title[2],
        "pos_title_len": pos_title[3],
        "neg_title_len": neg_title[3],
        "pos_date": pos_date[2],
        "neg_date": neg_date[2],
        "pos_genre": pos_genre[2],
        "neg_genre": neg_genre[2],
        "pos_userrating": pos_userrating,
        "neg_userrating": neg_userrating,
        "pos_relrating": pos_relrating,
        "neg_relrating": neg_relrating,
        "pos_movierating": pos_ratedmovie,
        "neg_movierating": neg_ratedmovie,
        "pos_poster_movie": pos_poster_movie,
        "pos_poster_rel": pos_poster_rel,
        "pos_poster_fm": pos_poster_fm,
        "neg_poster_movie": neg_poster_movie,
        "neg_poster_rel": neg_poster_rel,
        "neg_poster_fm": neg_poster_fm
    }


def build_gan_feed(batch, hyperparams, is_training=True):
    e1, r, e2 = batch
    pos_len = np.array([len(line) for line in e2], dtype=np.int32)
    max_pos_len = max(pos_len)

    pos_e2 = np.array([line + [0] * (max_pos_len - len(line)) for line in e2], dtype=np.int32)

    pos_title = e1, r, pos_e2, pos_len

    return {
        "pos_movie_e1": e1,
        "pos_movie_r": r,
        "pos_title": pos_e2,
        "pos_title_len": pos_len,
        "emb_keepprob": hyperparams["emb_keepprob"],
        "fm_keepprob": hyperparams["fm_keepprob"],
        "mlp_keepprob": hyperparams["mlp_keepprob"],
        "training_flag": is_training
    }

