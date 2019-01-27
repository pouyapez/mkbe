# Relations used: age, gender, occupation, zip, title, release date, genre, rating(1-5)
import tensorflow as tf

import metrics


def activation(x):
    """
    with tf.name_scope("selu") as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    """

    return tf.nn.relu(x)
        # return tf.tanh(x)


def define_graph(hyperparams, nodes, config=None):
    dtype = tf.float32 if config is None or "dtype" not in config else config["dtype"]
    id_dtype = tf.int32 if config is None or "id_dtype" not in config else config["id_dtype"]

    pos_user_e1 = tf.placeholder(tf.int32)
    pos_user_r = tf.placeholder(tf.int32)
    neg_user_e1 = tf.placeholder(tf.int32)
    neg_user_r = tf.placeholder(tf.int32)
    pos_movie_e1 = tf.placeholder(tf.int32)
    pos_movie_r = tf.placeholder(tf.int32)
    neg_movie_e1 = tf.placeholder(tf.int32)
    neg_movie_r = tf.placeholder(tf.int32)
    pos_age = tf.placeholder(tf.float32)
    neg_age = tf.placeholder(tf.float32)
    pos_gender = tf.placeholder(tf.int32)
    neg_gender = tf.placeholder(tf.int32)
    pos_occupation = tf.placeholder(tf.int32)
    neg_occupation = tf.placeholder(tf.int32)
    pos_zip = tf.placeholder(tf.int32)
    neg_zip = tf.placeholder(tf.int32)
    pos_title = tf.placeholder(tf.int32, shape=(None, None))
    neg_title = tf.placeholder(tf.int32, shape=(None, None))
    pos_title_len = tf.placeholder(tf.int32)
    neg_title_len = tf.placeholder(tf.int32)
    pos_date = tf.placeholder(tf.float32)
    neg_date = tf.placeholder(tf.float32)
    pos_genre = tf.placeholder(tf.float32)
    neg_genre = tf.placeholder(tf.float32)
    pos_userrating = tf.placeholder(tf.int32)
    neg_userrating = tf.placeholder(tf.int32)
    pos_relrating = tf.placeholder(tf.int32)
    neg_relrating = tf.placeholder(tf.int32)
    pos_movierating = tf.placeholder(tf.int32)
    neg_movierating = tf.placeholder(tf.int32)
    pos_poster_movie = tf.placeholder(tf.int32)
    pos_poster_rel = tf.placeholder(tf.int32)
    pos_poster_fm = tf.placeholder(tf.float32, shape=(None, None, None, 512))
    neg_poster_movie = tf.placeholder(tf.int32)
    neg_poster_rel = tf.placeholder(tf.int32)
    neg_poster_fm = tf.placeholder(tf.float32, shape=(None, None, None, 512))

    mlp_keepprob = tf.placeholder(tf.float32, name="mlp_keepprob")
    enc_keepprob = tf.placeholder(tf.float32, name="enc_keepprob")
    emb_keepprob = tf.placeholder(tf.float32, name="emb_keepprob")
    fm_keepprob = tf.placeholder(tf.float32, name="fm_keepprob")
    is_training = tf.placeholder(tf.bool, name="is_training")

    # Weights for embeddings
    if hyperparams["emb_dim"] > 3:
        user_weights = tf.get_variable(
            "user_weights", shape=[hyperparams["user_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"],
                                                                       mode="FAN_OUT", dtype=dtype))
        rel_weights = tf.get_variable(
            "relation_weights", shape=[hyperparams["relation_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"],
                                                                       mode="FAN_OUT", dtype=dtype))
        movie_weights = tf.get_variable(
            "movie_weights", shape=[hyperparams["movie_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"],
                                                                       mode="FAN_OUT", dtype=dtype))
        gender_weights = tf.get_variable(
            "gender_weights", shape=[hyperparams["gender_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"],
                                                                       mode="FAN_OUT", dtype=dtype))
        job_weights = tf.get_variable(
            "job_weights", shape=[hyperparams["job_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"],
                                                                       mode="FAN_OUT", dtype=dtype))
        zip_weights = tf.get_variable(
            "zip_weights", shape=[hyperparams["zip_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"],
                                                                       mode="FAN_OUT", dtype=dtype))
        char_weights = tf.get_variable(
            "char_weights", shape=[hyperparams["char_size"], hyperparams["emb_dim"] // 2],
            dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"],
                                                                       mode="FAN_OUT", dtype=dtype))

    else:
        user_weights = tf.get_variable(
            "user_weights", shape=[hyperparams["user_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        rel_weights = tf.get_variable(
            "relation_weights", shape=[hyperparams["relation_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        movie_weights = tf.get_variable(
            "movie_weights", shape=[hyperparams["movie_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        gender_weights = tf.get_variable(
            "gender_weights", shape=[hyperparams["gender_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        job_weights = tf.get_variable(
            "job_weights", shape=[hyperparams["job_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        zip_weights = tf.get_variable(
            "zip_weights", shape=[hyperparams["zip_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        char_weights = tf.get_variable(
            "char_weights", shape=[hyperparams["char_size"], hyperparams["emb_dim"] // 2],
            dtype=dtype, initializer=tf.truncated_normal_initializer(dtype=dtype)
        )

    # Biases embeddings
    user_bias = tf.get_variable("user_bias", shape=[hyperparams["user_size"], 1], dtype=dtype,
                                initializer=tf.truncated_normal_initializer(dtype=dtype))
    rel_bias = tf.get_variable("rel_bias", shape=[hyperparams["relation_size"], 1], dtype=dtype,
                               initializer=tf.truncated_normal_initializer(dtype=dtype))
    movie_bias = tf.get_variable("movie_bias", shape=[hyperparams["movie_size"], 1], dtype=dtype,
                                 initializer=tf.truncated_normal_initializer(dtype=dtype))

    # Embedding lookup
    pos_user_e1_emb = tf.nn.embedding_lookup(user_weights, pos_user_e1)
    neg_user_e1_emb = tf.nn.embedding_lookup(user_weights, neg_user_e1)
    pos_user_r_emb = tf.nn.embedding_lookup(rel_weights, pos_user_r)
    neg_user_r_emb = tf.nn.embedding_lookup(rel_weights, neg_user_r)
    pos_movie_e1_emb = tf.nn.embedding_lookup(movie_weights, pos_movie_e1)
    neg_movie_e1_emb = tf.nn.embedding_lookup(movie_weights, neg_movie_e1)
    pos_movie_r_emb = tf.nn.embedding_lookup(rel_weights, pos_movie_r)
    neg_movie_r_emb = tf.nn.embedding_lookup(rel_weights, neg_movie_r)
    pos_gender_emb = tf.nn.embedding_lookup(gender_weights, pos_gender)
    neg_gender_emb = tf.nn.embedding_lookup(gender_weights, neg_gender)
    pos_occupation_emb = tf.nn.embedding_lookup(job_weights, pos_occupation)
    neg_occupation_emb = tf.nn.embedding_lookup(job_weights, neg_occupation)
    pos_zip_emb = tf.nn.embedding_lookup(zip_weights, pos_zip)
    neg_zip_emb = tf.nn.embedding_lookup(zip_weights, neg_zip)
    pos_title_emb = tf.nn.embedding_lookup(char_weights, pos_title)
    neg_title_emb = tf.nn.embedding_lookup(char_weights, neg_title)
    pos_userrating_emb = tf.nn.embedding_lookup(user_weights, pos_userrating)
    neg_userrating_emb = tf.nn.embedding_lookup(user_weights, neg_userrating)
    pos_relrating_emb = tf.nn.embedding_lookup(rel_weights, pos_relrating)
    neg_relrating_emb = tf.nn.embedding_lookup(rel_weights, neg_relrating)
    pos_ratedmovie_emb = tf.nn.embedding_lookup(movie_weights, pos_movierating)
    neg_ratedmovie_emb = tf.nn.embedding_lookup(movie_weights, neg_movierating)
    pos_poster_movie_emb = tf.nn.embedding_lookup(movie_weights, pos_poster_movie)
    neg_poster_movie_emb = tf.nn.embedding_lookup(movie_weights, neg_poster_movie)
    pos_poster_rel_emb = tf.nn.embedding_lookup(rel_weights, pos_poster_rel)
    neg_poster_rel_emb = tf.nn.embedding_lookup(rel_weights, neg_poster_rel)

    pos_userrating_bias_emb = tf.nn.embedding_lookup(user_bias, pos_userrating)
    neg_userrating_bias_emb = tf.nn.embedding_lookup(user_bias, neg_userrating)
    pos_relrating_bias_emb = tf.nn.embedding_lookup(rel_bias, pos_relrating)
    neg_relrating_bias_emb = tf.nn.embedding_lookup(rel_bias, neg_relrating)
    pos_ratedmovie_bias_emb = tf.nn.embedding_lookup(movie_bias, pos_movierating)
    neg_ratedmovie_bias_emb = tf.nn.embedding_lookup(movie_bias, neg_movierating)

    # Collect Regularization variables
    regularized_variables = []

    # MLP Encoding
    # For ages
    age_weights = [tf.get_variable(
        "age_weights_{:}".format(layer),
        shape=[1 if layer == 0 else hyperparams["emb_dim"], hyperparams["emb_dim"]],
        dtype=dtype, initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["down_scale"],
                                                                                mode="FAN_AVG", dtype=dtype)
    ) for layer in range(hyperparams["MLPLayers"])]

    age_bias = [tf.get_variable(
        "age_bias_{:}".format(layer), shape=[hyperparams["emb_dim"]], dtype=dtype, initializer=tf.zeros_initializer
    ) for layer in range(hyperparams["MLPLayers"])]

    regularized_variables += age_weights
    regularized_variables += age_bias

    # Broadcasting for scalar-vector multiplication in the first layer and vector-matrix multiplication for other layers
    pos_age_node = [tf.reshape(pos_age, (-1, 1))]
    neg_age_node = [tf.reshape(neg_age, (-1, 1))]

    for w, b in zip(age_weights, age_bias):
        pos_age_node.append(tf.nn.dropout(activation(
            tf.add(b, tf.multiply(pos_age_node[-1], w) if len(pos_age_node) == 1 else tf.matmul(pos_age_node[-1], w))),
                                          enc_keepprob))

        neg_age_node.append(tf.nn.dropout(activation(
            tf.add(b, tf.multiply(neg_age_node[-1], w) if len(neg_age_node) == 1 else tf.matmul(neg_age_node[-1], w))),
                                          enc_keepprob))

    # For dates
    date_weights = [tf.get_variable(
        "date_weights_{:}".format(layer),
        shape=[1 if layer == 0 else hyperparams["emb_dim"], hyperparams["emb_dim"]],
        dtype=dtype,
        initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["down_scale"], mode="FAN_AVG",
                                                                   dtype=dtype)
    ) for layer in range(hyperparams["MLPLayers"])]

    date_bias = [tf.get_variable(
        "date_bias_{:}".format(layer), shape=[hyperparams["emb_dim"]], dtype=dtype, initializer=tf.zeros_initializer
    ) for layer in range(hyperparams["MLPLayers"])]

    regularized_variables += date_weights
    regularized_variables += date_bias

    # Broadcasting for scalar-vector multiplication in the first layer and vector-matrix multiplication for other layers
    pos_date_node = [tf.reshape(pos_date, (-1, 1))]
    neg_date_node = [tf.reshape(neg_date, (-1, 1))]

    for w, b in zip(date_weights, date_bias):
        pos_date_node.append(tf.nn.dropout(activation(tf.add(b, tf.multiply(pos_date_node[-1], w) if len(
            pos_date_node) == 1 else tf.matmul(pos_date_node[-1], w))), enc_keepprob))

        neg_date_node.append(tf.nn.dropout(activation(tf.add(b, tf.multiply(neg_date_node[-1], w) if len(
            neg_date_node) == 1 else tf.matmul(neg_date_node[-1], w))), enc_keepprob))

    # For genres
    genre_weights = [tf.get_variable(
        "genre_weights_{:}".format(layer),
        shape=[19 if layer == 0 else hyperparams["emb_dim"], hyperparams["emb_dim"]],
        dtype=dtype,
        initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["down_scale"], mode="FAN_AVG",
                                                                   dtype=dtype)
    ) for layer in range(hyperparams["MLPLayers"])]

    genre_bias = [tf.get_variable(
        "genre_bias_{:}".format(layer), shape=[hyperparams["emb_dim"]], dtype=dtype, initializer=tf.zeros_initializer
    ) for layer in range(hyperparams["MLPLayers"])]

    regularized_variables += genre_weights
    regularized_variables += genre_bias

    pos_genre_node = [tf.reshape(pos_genre, (-1, 19))]
    neg_genre_node = [tf.reshape(neg_genre, (-1, 19))]

    for w, b in zip(genre_weights, genre_bias):
        pos_genre_node.append(tf.nn.dropout(activation(tf.add(b, tf.matmul(pos_genre_node[-1], w))), enc_keepprob))
        neg_genre_node.append(tf.nn.dropout(activation(tf.add(b, tf.matmul(neg_genre_node[-1], w))), enc_keepprob))

    # GRU Encoding
    GRU_base_units = hyperparams["emb_dim"] // 2

    # With bidirectional GRU and concatenation of outputs, the dimension of input vectors times 2 after each layer
    GRUCells = [tf.nn.rnn_cell.GRUCell(GRU_base_units if layer < 2 else hyperparams["emb_dim"])
                for layer in range(hyperparams["GRULayers"] * 2)]

    pos_title_nodes = [pos_title_emb]
    neg_title_nodes = [neg_title_emb]
    for layer in range(hyperparams["GRULayers"]):
        with tf.variable_scope("GRUEncoder_{:}".format(layer)):
            if layer > 0:
                out_fw, out_bw = tf.nn.bidirectional_dynamic_rnn(
                    GRUCells[layer * 2], GRUCells[layer * 2 + 1], dtype=dtype,
                    inputs=pos_title_nodes[-1], sequence_length=pos_title_len, swap_memory=True)[0]
                pos_title_nodes.append(tf.nn.dropout((out_fw + out_bw) / 2.0, enc_keepprob))
            else:
                pos_title_nodes.append(tf.concat(tf.nn.bidirectional_dynamic_rnn(
                    GRUCells[layer * 2], GRUCells[layer * 2 + 1], dtype=dtype,
                    inputs=pos_title_nodes[-1], sequence_length=pos_title_len, swap_memory=True)[0], 2))

        # Share weights between encoders for positive samples and negative samples
        with tf.variable_scope("GRUEncoder_{:}".format(layer), reuse=True):
            if layer > 0:
                out_fw, out_bw = tf.nn.bidirectional_dynamic_rnn(
                    GRUCells[layer * 2], GRUCells[layer * 2 + 1], dtype=dtype,
                    inputs=neg_title_nodes[-1], sequence_length=neg_title_len, swap_memory=True)[0]
                neg_title_nodes.append(tf.nn.dropout((out_fw + out_bw) / 2.0, enc_keepprob))
            else:
                neg_title_nodes.append(tf.concat(tf.nn.bidirectional_dynamic_rnn(
                    GRUCells[layer * 2], GRUCells[layer * 2 + 1], dtype=dtype,
                    inputs=neg_title_nodes[-1], sequence_length=neg_title_len, swap_memory=True)[0], 2))

        regularized_variables += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="GRUEncoder_{:}".format(layer))

        gru_weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="GRUEncoder_{:}".format(layer))

    # CNN Encoding with bilinear pooling
    """
    proj_matrix1 = tf.get_variable("proj_matrix1", shape=[1, 1, 512, 1024], dtype=dtype,
                                   initializer=tf.contrib.layers.variance_scaling_initializer(
                                       factor=hyperparams["down_scale"], mode="FAN_AVG", dtype=dtype))
    proj_matrix2 = tf.get_variable("proj_matrix2", shape=[1, 1, 512, 1024], dtype=dtype,
                                   initializer=tf.contrib.layers.variance_scaling_initializer(
                                       factor=hyperparams["down_scale"], mode="FAN_AVG", dtype=dtype))

    pos_branch1 = tf.nn.conv2d(pos_poster_fm, proj_matrix1, strides=[1, 1, 1, 1], padding='SAME')
    pos_branch2 = tf.nn.conv2d(pos_poster_fm, proj_matrix2, strides=[1, 1, 1, 1], padding='SAME')
    pos_poster_vec = compact_bilinear_pooling_layer(pos_branch1, pos_branch2, hyperparams["emb_dim"])

    neg_branch1 = tf.nn.conv2d(neg_poster_fm, proj_matrix1, strides=[1, 1, 1, 1], padding='SAME')
    neg_branch2 = tf.nn.conv2d(neg_poster_fm, proj_matrix2, strides=[1, 1, 1, 1], padding='SAME')
    neg_poster_vec = compact_bilinear_pooling_layer(neg_branch1, neg_branch2, hyperparams["emb_dim"])
    """

    with tf.variable_scope("cnn_encoder"):
        ksize = hyperparams["ksize"]
        depth = hyperparams["depth"]
        drate = hyperparams["drate"]

        cnn_weights = tf.get_variable("cnn_weights", shape=[ksize, ksize, 512, depth], dtype=dtype,
                                      initializer=tf.contrib.layers.variance_scaling_initializer(
                                          factor=hyperparams["cnn_scale"], mode="FAN_AVG", dtype=dtype))
        cnn_bias = tf.get_variable("cnn_bias", shape=[depth], dtype=dtype, initializer=tf.zeros_initializer)

        pos_conv5_4 = tf.nn.dropout(tf.nn.relu(
            tf.nn.bias_add(tf.nn.convolution(pos_poster_fm, cnn_weights, "VALID", dilation_rate=[drate, drate]),
                           cnn_bias)), enc_keepprob)
        neg_conv5_4 = tf.nn.dropout(tf.nn.relu(
            tf.nn.bias_add(tf.nn.convolution(neg_poster_fm, cnn_weights, "VALID", dilation_rate=[drate, drate]),
                           cnn_bias)), enc_keepprob)

        # print(pos_conv5_4.shape, neg_conv5_4.shape)
        # pos_conv5_4_shape = tf.shape(pos_conv5_4)
        # neg_conv5_4_shape = tf.shape(neg_conv5_4)

        pos_pool5 = tf.nn.dropout(tf.reduce_mean(pos_conv5_4, axis=[1, 2]), enc_keepprob)
        neg_pool5 = tf.nn.dropout(tf.reduce_mean(neg_conv5_4, axis=[1, 2]), enc_keepprob)

        # print(pos_pool5.shape, neg_pool5.shape)
        # pos_pool5_shape = tf.shape(pos_pool5)
        # neg_pool5_shape = tf.shape(neg_pool5)

        fc_weights = tf.get_variable("fc_weights", shape=[depth, hyperparams["emb_dim"]], dtype=dtype,
                                     initializer=tf.contrib.layers.variance_scaling_initializer(
                                         factor=hyperparams["cnn_scale"], mode="FAN_AVG", dtype=dtype))
        fc_bias = tf.get_variable("fc_bias", shape=[hyperparams["emb_dim"]], dtype=dtype,
                                  initializer=tf.zeros_initializer)

        pos_poster_vec = tf.nn.dropout(activation(tf.add(tf.matmul(pos_pool5, fc_weights), fc_bias)), enc_keepprob)
        neg_poster_vec = tf.nn.dropout(activation(tf.add(tf.matmul(pos_pool5, fc_weights), fc_bias)), enc_keepprob)

        # print(pos_poster_vec.shape, neg_poster_vec.shape)
        # pos_poster_vec_shape = tf.shape(pos_poster_vec)
        # neg_poster_vec_shape = tf.shape(neg_poster_vec)

        regularized_variables += [cnn_weights, cnn_bias, fc_weights, fc_bias]

    # Aggregate and normalize e1
    pos_e1_list = [pos_user_e1_emb, pos_movie_e1_emb, pos_userrating_emb, pos_poster_movie_emb]
    neg_e1_list = [neg_user_e1_emb, neg_movie_e1_emb, neg_userrating_emb, neg_poster_movie_emb]

    if hyperparams["normalize_e1"]:
        pos_e1 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e1_list], axis=0),
            dim=1)

        neg_e1 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e1_list], axis=0),
            dim=1)

    else:
        pos_e1 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e1_list], axis=0)
        neg_e1 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e1_list], axis=0)
        regularized_variables += [user_weights, movie_weights]

    # Aggregate r
    pos_r_list = [pos_user_r_emb, pos_movie_r_emb, pos_relrating_emb, pos_poster_rel_emb]
    neg_r_list = [neg_user_r_emb, neg_movie_r_emb, neg_relrating_emb, neg_poster_rel_emb]

    if hyperparams["normalize_relation"]:
        pos_r = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_r_list], axis=0),
            dim=1)

        neg_r = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_r_list], axis=0),
            dim=1)

    else:
        pos_r = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_r_list], axis=0)
        neg_r = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_r_list], axis=0)
        regularized_variables += [rel_weights]

    # Aggregate and normalize e2
    pos_e2_list = [pos_age_node[-1], pos_gender_emb, pos_occupation_emb, pos_zip_emb, pos_title_nodes[-1][:, 0, :],
                   pos_date_node[-1], pos_genre_node[-1], pos_ratedmovie_emb, pos_poster_vec]
    neg_e2_list = [neg_age_node[-1], neg_gender_emb, neg_occupation_emb, neg_zip_emb, neg_title_nodes[-1][:, 0, :],
                   neg_date_node[-1], neg_genre_node[-1], neg_ratedmovie_emb, neg_poster_vec]

    if hyperparams["normalize_e2"]:
        pos_e2 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e2_list], axis=0),
            dim=1)
        neg_e2 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e2_list], axis=0),
            dim=1)

    else:
        pos_e2 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e2_list], axis=0)
        neg_e2 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e2_list], axis=0)
        regularized_variables += [movie_weights]

    if not hyperparams["bias"]:
        pos_bias = pos_relrating_bias_emb
        neg_bias = neg_relrating_bias_emb
    else:
        pos_bias = 0
        neg_bias = 0

    pose1_shape = tf.shape(pos_e1, name="pose1_shape")

    # ConvE link prediction
    with tf.variable_scope("convE"):
        emb_dim = hyperparams["emb_dim"]
        pose1_img = tf.reshape(pos_e1, (-1, emb_dim // 16, 16, 1))
        nege1_img = tf.reshape(neg_e1, (-1, emb_dim // 16, 16, 1))
        posr_img = tf.reshape(pos_r, (-1, emb_dim // 16, 16, 1))
        negr_img = tf.reshape(neg_r, (-1, emb_dim // 16, 16, 1))

        pos_stack = tf.layers.batch_normalization(tf.concat([pose1_img, posr_img], 2), training=is_training)
        neg_stack = tf.layers.batch_normalization(tf.concat([nege1_img, negr_img], 2), training=is_training)

        pos_indrop = tf.nn.dropout(pos_stack, emb_keepprob)
        neg_indrop = tf.nn.dropout(neg_stack, emb_keepprob)

        convE_ker = tf.get_variable("convE_ker", shape=[3, 3, 1, 32], dtype=dtype,
                                    initializer=tf.contrib.layers.variance_scaling_initializer(
                                        factor=hyperparams["cnn_scale"], mode="FAN_AVG", dtype=dtype))
        convE_bias = tf.get_variable("convE_bias", shape=[32], dtype=dtype, initializer=tf.zeros_initializer)

        pos_convE_conv = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.bias_add(tf.nn.convolution(pos_indrop, convE_ker, "SAME"), convE_bias), training=is_training))
        neg_convE_conv = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.bias_add(tf.nn.convolution(neg_indrop, convE_ker, "SAME"), convE_bias), training=is_training))

        pos_flat = tf.reshape(tf.nn.dropout(pos_convE_conv, fm_keepprob), (-1, emb_dim * 32 * 2))
        neg_flat = tf.reshape(tf.nn.dropout(neg_convE_conv, fm_keepprob), (-1, emb_dim * 32 * 2))

        pos_flat_shape = tf.shape(pos_flat, name="pos_flat_shape")

        convE_fc_w = tf.get_variable("convE_fc_w", shape=[hyperparams["emb_dim"] * 32 * 2, hyperparams["emb_dim"]],
                                     dtype=dtype, initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=hyperparams["init_scale"], mode="FAN_AVG", dtype=dtype))

        pos_fc = tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(tf.matmul(pos_flat, convE_fc_w), mlp_keepprob),
                                                          training=is_training))
        neg_fc = tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(tf.matmul(neg_flat, convE_fc_w), mlp_keepprob),
                                                          training=is_training))

        pos = tf.reduce_sum(tf.multiply(pos_fc, pos_e2), 1, keep_dims=True) + pos_bias
        neg = tf.reduce_sum(tf.multiply(neg_fc, neg_e2), 1, keep_dims=True) + neg_bias
        regularized_variables += [convE_ker, convE_bias, convE_fc_w]

    # Regularization term
    regularizer = tf.contrib.layers.l2_regularizer(hyperparams["regularization_coefficient"])
    regularization_term = tf.contrib.layers.apply_regularization(regularizer, regularized_variables)

    # Collect variables to be trained
    lr1_vars = [user_weights, movie_weights, user_bias, movie_bias]
    lr2_vars = [rel_weights, rel_bias, convE_ker, convE_bias, convE_fc_w]

    lr1_vars += age_weights + age_bias + [gender_weights, job_weights, zip_weights]

    lr2_vars += gru_weights

    lr1_vars += date_weights + date_bias + genre_weights + genre_bias

    lr1_vars += [cnn_weights, cnn_bias, fc_weights, fc_bias]
    # lr1_vars += [proj_matrix1, proj_matrix2]

    # Minimize Hinge Loss
    loss = tf.reduce_sum((tf.maximum(neg - pos + hyperparams["margin"], 0))) + regularization_term
    loss_to_show = tf.reduce_mean((tf.maximum(neg - pos + hyperparams["margin"], 0))) + regularization_term
    training_op = tf.train.AdagradOptimizer(learning_rate=hyperparams["learning_rate"]).minimize(
        loss, var_list=lr1_vars)

    rlr_train_op = tf.train.AdagradOptimizer(learning_rate=hyperparams["learning_rate_reduced"]).minimize(
        loss, var_list=lr2_vars)

    summary_nodes = [tf.summary.scalar("loss", loss_to_show),
                     tf.summary.scalar("regularization_term", regularization_term),
                     tf.summary.histogram("pos", pos),
                     tf.summary.histogram("neg", neg),
                     tf.summary.histogram("user_emb", user_weights),
                     tf.summary.histogram("relation_emb", rel_weights),
                     tf.summary.histogram("movie_emb", movie_weights)]

    training_summary = tf.summary.merge_all()
    return locals()


def scoring_and_counting(hyperparams, nodes, config=None):
    # Input placeholders
    rating_relations = tf.placeholder(tf.int32, shape=[5])
    pos_user = tf.placeholder(tf.int32)
    pos_r = tf.placeholder(tf.int32)
    pos_movie = tf.placeholder(tf.int32)
    emb_dim = hyperparams["emb_dim"]
    bsize = tf.shape(pos_user)[0]

    # Weights to use
    user_weights = nodes["user_weights"]
    movie_weights = nodes["movie_weights"]
    relation_weights = nodes["rel_weights"]

    relation_bias = nodes["rel_bias"]

    is_training = nodes["is_training"]
    mlp_keepprob = nodes["mlp_keepprob"]
    emb_keepprob = nodes["emb_keepprob"]
    fm_keepprob = nodes["fm_keepprob"]

    # Normalize e2 weights
    if hyperparams["test_normalize_e2"]:
        normalized_movie_weights = tf.nn.l2_normalize(movie_weights, dim=1)
    else:
        normalized_movie_weights = movie_weights

    # Normalize r weights
    if hyperparams["test_normalize_relation"]:
        normalized_relation_weights = tf.nn.l2_normalize(relation_weights, dim=1)
    else:
        normalized_relation_weights = relation_weights

    # Normalize e1 weights
    if hyperparams["test_normalize_e1"]:
        normalized_user_weights = tf.nn.l2_normalize(user_weights, dim=1)
    else:
        normalized_user_weights = user_weights

    # Embedding positive and negative samples
    pos_user_emb = tf.nn.embedding_lookup(normalized_user_weights, pos_user)
    pos_r_emb = tf.nn.embedding_lookup(normalized_relation_weights, pos_r)
    pos_movie_emb = tf.nn.embedding_lookup(normalized_movie_weights, pos_movie)
    rating_relation_weights = tf.nn.embedding_lookup(normalized_relation_weights, rating_relations)

    if hyperparams["bias"]:
        pos_score_bias = tf.reshape(tf.nn.embedding_lookup(relation_bias, pos_r), (-1, 1))
        neg_score_bias = tf.reshape(tf.nn.embedding_lookup(relation_bias, rating_relations), (1, 5, 1))
    else:
        pos_score_bias = 0
        neg_score_bias = 0

    # ConvE link prediction
    with tf.variable_scope("convE", reuse=tf.AUTO_REUSE):
        pose1_img = tf.reshape(pos_user_emb, (-1, emb_dim // 16, 16, 1))
        posr_img = tf.reshape(pos_r_emb, (-1, emb_dim // 16, 16, 1))
        negr_img = tf.reshape(rating_relation_weights, (-1, 5, emb_dim // 16, 16, 1))

        pos_stack = tf.layers.batch_normalization(tf.concat([pose1_img, posr_img], 2), training=False)
        neg_stack = tf.layers.batch_normalization(
            tf.concat([tf.tile(tf.expand_dims(pose1_img, axis=1), [1, 5, 1, 1, 1]),
                       tf.tile(negr_img, [bsize, 1, 1, 1, 1])], 3),
            training=False)

        pos_indrop = tf.nn.dropout(pos_stack, 1.0)
        neg_indrop = tf.nn.dropout(neg_stack, 1.0)

        convE_ker = tf.get_variable("convE_ker")
        convE_bias = tf.get_variable("convE_bias")

        pos_convE_conv = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.bias_add(tf.nn.convolution(pos_indrop, convE_ker, "SAME"), convE_bias), training=False))
        neg_convE_conv = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.bias_add(tf.nn.convolution(
                neg_indrop, tf.expand_dims(convE_ker, axis=0), "SAME"), convE_bias),
            training=False))

        pos_flat = tf.reshape(tf.nn.dropout(pos_convE_conv, 1), (-1, emb_dim * 32 * 2))
        neg_flat = tf.reshape(tf.nn.dropout(neg_convE_conv, 1), (-1, emb_dim * 32 * 2))

        convE_fc_w = tf.get_variable("convE_fc_w")

        pos_fc = tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(tf.matmul(pos_flat, convE_fc_w), 1),
                                                          training=False))
        neg_fc = tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(tf.matmul(neg_flat, convE_fc_w), 1),
                                                          training=False))

        neg_e1_r = tf.reshape(neg_fc, (bsize, 5, emb_dim))
        pos_movie_tile = tf.tile(tf.expand_dims(pos_movie_emb, 1), [1, 5, 1])

        pos = tf.reduce_sum(tf.multiply(pos_fc, pos_movie_emb), 1, keep_dims=True) + pos_score_bias
        neg = tf.reduce_sum(tf.multiply(neg_e1_r, pos_movie_tile), 2, keep_dims=True) + neg_score_bias

    pos_scoring = tf.reshape(pos, (-1, 1, 1))
    neg_scoring_rating = tf.reshape(neg, (-1, 5, 1))

    # movie_higher_values = tf.reduce_sum(tf.cast(neg_scoring_movie > pos_scoring, tf.float32), axis=1)
    rating_higher_values = tf.reduce_sum(tf.cast(neg_scoring_rating > pos_scoring, tf.float32), axis=1)
    return locals()


def test_graph(hyperparams, nodes, config=None):
    nodes = scoring_and_counting(hyperparams, nodes, config=config)
    metric_values = {
        # "MRR_movie": metrics.mrr(nodes["movie_higher_values"]),
        # "HITS@10_movie": metrics.hits_n(nodes["movie_higher_values"], 10),
        # "HITS@3_movie": metrics.hits_n(nodes["movie_higher_values"], 3),
        # "HITS@1_movie": metrics.hits_n(nodes["movie_higher_values"], 1),
        "MRR_r": metrics.mrr(nodes["rating_higher_values"]),
        "HITS@5_r": metrics.hits_n(nodes["rating_higher_values"], 5),
        "HITS@3_r": metrics.hits_n(nodes["rating_higher_values"], 3),
        "HITS@2_r": metrics.hits_n(nodes["rating_higher_values"], 2),
        "HITS@1_r": metrics.hits_n(nodes["rating_higher_values"], 1)
    }
    nodes.update(metric_values)

    summaries = [tf.summary.scalar(k, v) for k, v in metric_values.items()] + [
        tf.summary.histogram("rating score rankings", nodes["rating_higher_values"]),
        # tf.summary.histogram("movie score rankings", nodes["movie_higher_values"])
    ]

    nodes["test_summary"] = tf.summary.merge(summaries)

    return nodes


def debug_graph(hyperparams, nodes, config=None):
    """
    rating_rankings_min = tf.reduce_max(nodes["rating_higher_values"])
    rating_rankings_max = tf.reduce_max(nodes["rating_higher_values"])

    neg_score_rating_shape = tf.shape(nodes["neg_scoring_rating"])
    neg_r_emb_shape = tf.shape(nodes["neg_r_emb"])
    pos_u_m_mul_shape = tf.shape(nodes["pos_user_movie_mul"])
    pos_scoring_shape = tf.shape(nodes["pos_scoring"])

    """
    return locals()
