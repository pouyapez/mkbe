import itertools

import numpy as np
import pandas as pd

# subset can be ["movie_user_rating", "movie_title_rating", "movie_rating", "user_rating", "rating"]
fold = 1
subset = "movie_title_poster_user_rating"
#subset = "movie_title_user_rating"

in_files = {
    "user-train": "../code/movielens/ml-100k/u.user",
    "movie-train": "../code/movielens/ml-100k/u.item",
    "rating-train": "../code/movielens/ml-100k/u{:}.base".format(fold),
    "rating-test": "../code/movielens/ml-100k/u{:}.test".format(fold),
    "cached-posters": "../code/movielens/ml-100k/feature_maps.npy"
}

out_files = {
    "scale": "../assets/ml100k-processed/u{:}-{:}-scale.npy".format(fold, subset),
    "train": "../assets/ml100k-processed/u{:}-{:}-train.npy".format(fold, subset),
    "test": "../assets/ml100k-processed/u{:}-{:}-test.npy".format(fold, subset),
    "idencoders": "../assets/ml100k-processed/u{:}-{:}-idencoder.npy".format(fold, subset),
    "titles": "../assets/ml100k-processed/u{:}-{:}-titles.npy".format(fold, subset),
    "title_dict": "../assets/ml100k-processed/u{:}-{:}-title-dict.npy".format(fold, subset)
}

user_headers = ["userid", "age", "gender", "occupation", "zip"]
user_r = ["is of_" + h for h in ["age", "gender", "occupation", "zip"]]
movie_headers = ["movieid", "title", "release date", "video release date", "IMDb URL", "unknown", "action", "adventure",
                 "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror",
                 "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
movie_r = ["is of_" + h for h in ["title", "release date", "genre", "poster"]]
rating_headers = ["userid", "movieid", "rating", "timestamp"]


def read_and_filter():
    userdf = pd.read_csv(in_files["user-train"], engine="c", names=user_headers, sep="|")
    moviedf = pd.read_csv(in_files["movie-train"], engine="c", names=movie_headers, sep="|", encoding="latin1")
    rating_train = pd.read_csv(in_files["rating-train"], engine="c", names=rating_headers, sep="\t")
    rating_test = pd.read_csv(in_files["rating-test"], engine="c", names=rating_headers, sep="\t")

    # Normalize user ages
    age_scale_params = {
        "mean": userdf.mean()["age"],
        "std": userdf.std()["age"]
    }
    userdf["age"] -= age_scale_params["mean"]
    userdf["age"] /= age_scale_params["std"]

    # Slice first 2 digits of zip codes
    userdf["zip"] = userdf["zip"].str.slice(0, 2)

    # Normalize movie release dates
    moviedf["release date"] = pd.to_datetime(moviedf["release date"]).astype("int64")
    date_scale_params = {
        "mean": moviedf.mean()["release date"],
        "std": moviedf.std()["release date"]
    }
    moviedf["release date"] -= date_scale_params["mean"]
    moviedf["release date"] /= date_scale_params["std"]

    # Remove year from movie titles
    moviedf["title"] = moviedf["title"].str.replace(r" \([0-9]+\)$", "")

    scale_params = {
        "age": age_scale_params,
        "date": date_scale_params
    }
    np.save(out_files["scale"], np.array(scale_params))

    return userdf, moviedf, rating_train, rating_test, scale_params


def build_dict(userdf, moviedf, rating_train, rating_test):
    genders = set(userdf["gender"])
    gender2id = dict(zip(genders, range(len(genders))))

    occupations = set(userdf["occupation"])
    job2id = dict(zip(occupations, range(len(occupations))))

    zipcodes = set(userdf["zip"])
    zip2id = dict(zip(zipcodes, range(len(zipcodes))))

    chars = set(itertools.chain.from_iterable(moviedf["title"].values))
    chars.update(["<go>", "<eos>"])
    char2id = dict(zip(chars, range(len(chars))))

    relations = set("rate_{:}".format(rating) for rating in set(rating_train["rating"]))
    relations.update(user_r)
    relations.update(movie_r)
    rel2id = dict(zip(relations, range(len(relations))))

    idenc = {
        "gender2id": gender2id,
        "job2id": job2id,
        "zip2id": zip2id,
        "char2id": char2id,
        "rel2id": rel2id,
        "maxuserid": max(userdf["userid"]),
        "maxmovieid": max(moviedf["movieid"])
    }

    np.save(out_files["idencoders"], np.array(idenc))

    return gender2id, job2id, zip2id, char2id, rel2id


def encode(userdf, moviedf, rating_train, rating_test, gender2id, job2id, zip2id, char2id, rel2id):
    train_triplets = []
    test_triplets = []
    title_symlist = []
    title_idlist = []
    attr2enc = {
        "gender": gender2id,
        "occupation": job2id,
        "zip": zip2id
    }

    af = "is of_"
    # Encode user attributes
    if "user" in subset:
        for attribute in ["age", "gender", "occupation", "zip"]:
            userids = userdf["userid"]
            attrs = userdf[attribute]
            for e1, e2 in zip(userids, attrs):
                encoded_e2 = attr2enc[attribute][e2] if attribute in attr2enc else e2
                train_triplets.append((e1, rel2id[af + attribute], encoded_e2))

    if "movie" in subset:
        movieids = moviedf["movieid"]
        if "title" in subset:
            # Encode movie titles
            titles = moviedf["title"]
            for e1, e2 in zip(movieids, titles):
                encoded_e2 = [char2id["<go>"]] + [char2id[c] for c in e2] + [char2id["<eos>"]]
                train_triplets.append((e1, rel2id[af + "title"], encoded_e2))
                title_symlist.append(encoded_e2)
                title_idlist.append(e1)

        # Encode movie release dates
        release_date = moviedf["release date"]
        for e1, e2 in zip(movieids, release_date):
            train_triplets.append((e1, rel2id[af + "release date"], e2))

        # Encode movie genres
        genre = moviedf[["unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary",
                         "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller",
                         "war", "western"]]
        for e1, e2 in zip(movieids, genre.values):
            train_triplets.append((e1, rel2id[af + "genre"], e2))

    if "poster" in subset:
        poster_dict = np.load(in_files["cached-posters"]).item()
        for e1, e2 in poster_dict.items():
            train_triplets.append((e1, rel2id[af + "poster"], e2))

    # Encode training ratings
    for e1, e2, r, _ in rating_train.values:
        encoded_r = rel2id["rate_{:}".format(r)]
        train_triplets.append((e1, encoded_r, e2))

    # Encode test ratings
    for e1, e2, r, _ in rating_test.values:
        encoded_r = rel2id["rate_{:}".format(r)]
        test_triplets.append((e1, encoded_r, e2))

    training_set = np.array(train_triplets, dtype=tuple)
    test_set = np.array(test_triplets, dtype=tuple)
    title_set = np.array(title_symlist, dtype=list)
    title_dict = dict(zip(title_idlist, title_symlist))
    print(len(title_dict))

    np.random.shuffle(training_set)
    np.random.shuffle(test_set)

    np.save(out_files["test"], test_set)
    np.save(out_files["train"], training_set)
    np.save(out_files["titles"], title_set)
    np.save(out_files["title_dict"], title_dict)


if __name__ == "__main__":
    userdf, moviedf, rating_train, rating_test, scale_params = read_and_filter()
    gender2id, job2id, zip2id, char2id, rel2id = build_dict(userdf, moviedf, rating_train, rating_test)
    encode(userdf, moviedf, rating_train, rating_test, gender2id, job2id, zip2id, char2id, rel2id)
