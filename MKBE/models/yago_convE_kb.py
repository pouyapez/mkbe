import tensorflow as tf


def activation(x):
    """
    with tf.name_scope("selu") as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))
    """
    return tf.tanh(x)


def define_graph(hyperparams, config=None):
    dtype = tf.float32 if config is None or "dtype" not in config else config["dtype"]
    id_dtype = tf.int32 if config is None or "id_dtype" not in config else config["id_dtype"]

    pos_e1 = tf.placeholder(tf.int32, name="pos_e1")
    pos_r = tf.placeholder(tf.int32, name="pos_r")
    pos_e2 = tf.placeholder(tf.int32, name="pos_e2")
    neg_e1 = tf.placeholder(tf.int32, name="neg_e1")
    neg_r = tf.placeholder(tf.int32, name="neg_r")
    neg_e2 = tf.placeholder(tf.int32, name="neg_e2")
    pos_num = tf.placeholder(tf.float32, name="pos_num")
    neg_num = tf.placeholder(tf.float32, name="neg_num")
    pos_text = tf.placeholder(tf.int32, shape=(None, None), name="pos_text")
    neg_text = tf.placeholder(tf.int32, shape=(None, None), name="neg_text")
    pos_text_len = tf.placeholder(tf.int32, name="pos_text_len")
    neg_text_len = tf.placeholder(tf.int32, name="neg_text_len")

    mlp_keepprob = tf.placeholder(tf.float32, name="mlp_keepprob")
    enc_keepprob = tf.placeholder(tf.float32, name="enc_keepprob")
    emb_keepprob = tf.placeholder(tf.float32, name="emb_keepprob")
    fm_keepprob = tf.placeholder(tf.float32, name="fm_keepprob")

    # Weights for embeddings
    if hyperparams["emb_dim"] > 3:
        entity_weights = tf.get_variable(
            "user_weights", shape=[hyperparams["entity_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"], mode="FAN_OUT",
                                                                       dtype=dtype))
        rel_weights = tf.get_variable(
            "relation_weights", shape=[hyperparams["relation_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"], mode="FAN_OUT",
                                                                       dtype=dtype))

        word_weights = tf.get_variable(
            "word_weights", shape=[hyperparams["word_size"], hyperparams["emb_dim"] // 2],
            dtype=dtype,
            initializer=tf.contrib.layers.variance_scaling_initializer(factor=hyperparams["init_scale"], mode="FAN_OUT",
                                                                       dtype=dtype))

    else:
        entity_weights = tf.get_variable(
            "entity_weights", shape=[hyperparams["entity_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        rel_weights = tf.get_variable(
            "relation_weights", shape=[hyperparams["relation_size"], hyperparams["emb_dim"]], dtype=dtype,
            initializer=tf.truncated_normal_initializer(dtype=dtype))
        word_weights = tf.get_variable(
            "word_weights", shape=[hyperparams["word_size"], hyperparams["emb_dim"] // 2],
            dtype=dtype, initializer=tf.truncated_normal_initializer(dtype=dtype)
        )

    # Biases embeddings
    entity_bias = tf.get_variable("entity_bias", shape=[hyperparams["entity_size"], 1], dtype=dtype,
                                  initializer=tf.truncated_normal_initializer(dtype=dtype))
    rel_bias = tf.get_variable("rel_bias", shape=[hyperparams["relation_size"], 1], dtype=dtype,
                               initializer=tf.truncated_normal_initializer(dtype=dtype))

    # Embedding lookup
    pos_e1_emb = tf.nn.embedding_lookup(entity_weights, pos_e1)
    neg_e1_emb = tf.nn.embedding_lookup(entity_weights, neg_e1)
    pos_r_emb = tf.nn.embedding_lookup(rel_weights, pos_r)
    neg_r_emb = tf.nn.embedding_lookup(rel_weights, neg_r)
    pos_e2_emb = tf.nn.embedding_lookup(entity_weights, pos_e2)
    neg_e2_emb = tf.nn.embedding_lookup(entity_weights, neg_e2)
    pos_text_emb = tf.nn.embedding_lookup(word_weights, pos_text)
    neg_text_emb = tf.nn.embedding_lookup(word_weights, neg_text)

    pos_e1_bias_emb = tf.nn.embedding_lookup(entity_bias, pos_e1)
    neg_e1_bias_emb = tf.nn.embedding_lookup(entity_bias, neg_e1)
    pos_rel_bias_emb = tf.nn.embedding_lookup(rel_bias, pos_r)
    neg_rel_bias_emb = tf.nn.embedding_lookup(rel_bias, neg_r)
    pos_e2_bias_emb = tf.nn.embedding_lookup(entity_bias, pos_e2)
    neg_e2_bias_emb = tf.nn.embedding_lookup(entity_bias, neg_e2)

    # Collect Regularization variables
    regularized_variables = []

    # MLP Encoding
    # For num
    num_weights = [tf.get_variable(
        "num_weights_{:}".format(layer),
        shape=[1 if layer == 0 else hyperparams["emb_dim"], hyperparams["emb_dim"]],
        dtype=dtype, initializer=tf.contrib.layers.variance_scaling_initializer(factor=6.0, mode="FAN_OUT", dtype=dtype)
    ) for layer in range(hyperparams["MLPLayers"])]

    num_bias = [tf.get_variable(
        "num_bias_{:}".format(layer), shape=[hyperparams["emb_dim"]], dtype=dtype, initializer=tf.zeros_initializer
    ) for layer in range(hyperparams["MLPLayers"])]

    regularized_variables += num_weights
    regularized_variables += num_bias

    # Broadcasting for scalar-vector multiplication in the first layer and vector-matrix multiplication for other layers
    pos_num_node = [tf.reshape(pos_num, (-1, 1))]
    neg_num_node = [tf.reshape(neg_num, (-1, 1))]

    for w, b in zip(num_weights, num_bias):
        pos_num_node.append(activation(tf.add(
            b, tf.multiply(pos_num_node[-1], w) if len(pos_num_node) == 1 else tf.matmul(pos_num_node[-1], w))))

        neg_num_node.append(activation(tf.add(
            b, tf.multiply(neg_num_node[-1], w) if len(neg_num_node) == 1 else tf.matmul(neg_num_node[-1], w))))

    """
    # GRU Encoding
    GRU_base_units = hyperparams["emb_dim"] // (2 ** hyperparams["GRULayers"])

    # With bidirectional GRU and concatenation of outputs, the dimension of input vectors times 2 after each layer
    GRUCells = [tf.nn.rnn_cell.GRUCell(GRU_base_units * 2 ** (layer // 2))
                for layer in range(hyperparams["GRULayers"] * 2)]

    pos_text_nodes = [pos_text_emb]
    neg_text_nodes = [neg_text_emb]
    for layer in range(hyperparams["GRULayers"]):
        with tf.variable_scope("GRUEncoder_{:}".format(layer)):
            pos_text_nodes.append(tf.concat(tf.nn.bidirectional_dynamic_rnn(
                GRUCells[layer * 2], GRUCells[layer * 2 + 1], dtype=dtype,
                inputs=pos_text_nodes[-1], sequence_length=pos_text_len, swap_memory=True)[0], 2))

        # Share weights between encoders for positive samples and negative samples
        with tf.variable_scope("GRUEncoder_{:}".format(layer), reuse=True):
            neg_text_nodes.append(tf.concat(tf.nn.bidirectional_dynamic_rnn(
                GRUCells[layer * 2], GRUCells[layer * 2 + 1], dtype=dtype,
                inputs=neg_text_nodes[-1], sequence_length=neg_text_len, swap_memory=True)[0], 2))

        regularized_variables += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="GRUEncoder_{:}".format(layer))
    """
    """
    # CNN Encoding for text
    text_weights = [tf.get_variable(
        "text_weights_{:}".format(layer),
        shape=[hyperparams["CNNTextKernel"], 1,
               hyperparams["emb_dim"] // 2 if layer == 0 else hyperparams["emb_dim"], hyperparams["emb_dim"]],
        dtype=dtype, initializer=tf.contrib.layers.variance_scaling_initializer(factor=6.0, mode="FAN_OUT", dtype=dtype)
    ) for layer in range(hyperparams["CNNTextLayers"])]
    text_bias = [tf.get_variable(
        "text_bias_{:}".format(layer), shape=[hyperparams["emb_dim"]], dtype=dtype, initializer=tf.zeros_initializer
    ) for layer in range(hyperparams["CNNTextLayers"])]

    regularized_variables += text_weights
    regularized_variables += text_bias

    pos_text_nodes = [tf.expand_dims(pos_text_emb, 2)]
    neg_text_nodes = [tf.expand_dims(neg_text_emb, 2)]

    for w, b in zip(text_weights, text_bias):
        pos_text_nodes.append(tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(pos_text_nodes[-1], w, [1, 1, 1, 1], 'SAME'), b)))

        neg_text_nodes.append(tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(neg_text_nodes[-1], w, [1, 1, 1, 1], 'SAME'), b)))

    pos_text_vec = tf.reduce_mean(pos_text_nodes[-1][:, :, 0, :], axis=1)
    neg_text_vec = tf.reduce_mean(neg_text_nodes[-1][:, :, 0, :], axis=1)
    """

    # Aggregate and normalize e1
    pos_e1_list = [pos_e1_emb]
    neg_e1_list = [neg_e1_emb]

    if hyperparams["normalize_e1"]:
        Pos_e1 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e1_list], axis=0),
            dim=1)

        Neg_e1 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e1_list], axis=0),
            dim=1)

    else:
        Pos_e1 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e1_list], axis=0)
        Neg_e1 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e1_list], axis=0)
        regularized_variables += [entity_weights]

    # Aggregate r
    pos_r_list = [pos_r_emb]
    neg_r_list = [neg_r_emb]

    if hyperparams["normalize_relation"]:
        Pos_r = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_r_list], axis=0),
            dim=1)

        Neg_r = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_r_list], axis=0),
            dim=1)

    else:
        Pos_r = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_r_list], axis=0)
        Neg_r = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_r_list], axis=0)
        regularized_variables += [rel_weights]

    # Aggregate and normalize e2
    pos_e2_list = [pos_e2_emb, pos_num_node[-1]]
    neg_e2_list = [neg_e2_emb, neg_num_node[-1]]

    if hyperparams["normalize_e2"]:
        Pos_e2 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e2_list], axis=0),
            dim=1)
        Neg_e2 = tf.nn.l2_normalize(
            tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e2_list], axis=0),
            dim=1)

    else:
        Pos_e2 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in pos_e2_list], axis=0)
        Neg_e2 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in neg_e2_list], axis=0)
        regularized_variables += [entity_weights]

    if not hyperparams["bias"]:
        pos_bias = pos_rel_bias_emb
        neg_bias = neg_rel_bias_emb
    else:
        pos_bias = 0
        neg_bias = 0

    # ConvE link prediction
    with tf.variable_scope("convE"):
        emb_dim = hyperparams["emb_dim"]
        pose1_img = tf.reshape(Pos_e1, (-1, emb_dim // 16, 16, 1))
        nege1_img = tf.reshape(Neg_e1, (-1, emb_dim // 16, 16, 1))
        posr_img = tf.reshape(Pos_r, (-1, emb_dim // 16, 16, 1))
        negr_img = tf.reshape(Neg_r, (-1, emb_dim // 16, 16, 1))

        pos_stack = tf.layers.batch_normalization(tf.concat([pose1_img, posr_img], 2), training=True)
        neg_stack = tf.layers.batch_normalization(tf.concat([nege1_img, negr_img], 2), training=True)

        pos_indrop = tf.nn.dropout(pos_stack, emb_keepprob)
        neg_indrop = tf.nn.dropout(neg_stack, emb_keepprob)

        convE_ker = tf.get_variable("convE_ker", shape=[3, 3, 1, 32], dtype=dtype,
                                    initializer=tf.contrib.layers.variance_scaling_initializer(
                                        factor=hyperparams["cnn_scale"], mode="FAN_AVG", dtype=dtype))
        convE_bias = tf.get_variable("convE_bias", shape=[32], dtype=dtype, initializer=tf.zeros_initializer)

        pos_convE_conv = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.bias_add(tf.nn.convolution(pos_indrop, convE_ker, "SAME"), convE_bias), training=True))
        neg_convE_conv = tf.nn.relu(tf.layers.batch_normalization(
            tf.nn.bias_add(tf.nn.convolution(neg_indrop, convE_ker, "SAME"), convE_bias), training=True))

        pos_flat = tf.reshape(tf.nn.dropout(pos_convE_conv, fm_keepprob), (-1, emb_dim * 32 * 2))
        neg_flat = tf.reshape(tf.nn.dropout(neg_convE_conv, fm_keepprob), (-1, emb_dim * 32 * 2))

        pos_flat_shape = tf.shape(pos_flat, name="pos_flat_shape")

        convE_fc_w = tf.get_variable("convE_fc_w", shape=[hyperparams["emb_dim"] * 32 * 2, hyperparams["emb_dim"]],
                                     dtype=dtype, initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=hyperparams["init_scale"], mode="FAN_AVG", dtype=dtype))

        pos_fc = tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(tf.matmul(pos_flat, convE_fc_w), mlp_keepprob),
                                                          training=True))
        neg_fc = tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(tf.matmul(neg_flat, convE_fc_w), mlp_keepprob),
                                                          training=True))

        pos = tf.reduce_sum(tf.multiply(pos_fc, Pos_e2), 1, keep_dims=True) + pos_bias
        neg = tf.reduce_sum(tf.multiply(neg_fc, Neg_e2), 1, keep_dims=True) + neg_bias
        regularized_variables += [convE_ker, convE_bias, convE_fc_w]

    # Regularization term
    regularizer = tf.contrib.layers.l2_regularizer(hyperparams["regularization_coefficient"])
    regularization_term = tf.contrib.layers.apply_regularization(regularizer, regularized_variables)

    # Minimize Hinge Loss
    loss = tf.reduce_sum((tf.maximum(neg - pos + hyperparams["margin"], 0))) + regularization_term
    loss_to_show = tf.reduce_mean((tf.maximum(neg - pos + hyperparams["margin"], 0))) + regularization_term

    training_op = tf.train.AdagradOptimizer(learning_rate=hyperparams["learning_rate"]).minimize(
        loss, var_list=[entity_weights, entity_bias] + num_weights + num_bias + [convE_ker, convE_bias, convE_fc_w])

    rlr_train_op = tf.train.AdagradOptimizer(learning_rate=hyperparams["learning_rate_reduced"]).minimize(
        loss, var_list=[rel_weights, rel_bias]
    )

    summary_nodes = [tf.summary.scalar("loss", loss_to_show),
                     tf.summary.scalar("regularization_term", regularization_term),
                     tf.summary.histogram("pos", pos),
                     tf.summary.histogram("neg", neg),
                     tf.summary.histogram("entity_emb", entity_weights),
                     tf.summary.histogram("relation_emb", rel_weights)]

    training_summary = tf.summary.merge_all()

    return locals()


def scoring_and_counting(hyperparams, nodes, config=None):
    """
    # Input placeholders
    pos_e1 = tf.placeholder(tf.int32)
    pos_r = tf.placeholder(tf.int32)
    pos_e2 = tf.placeholder(tf.int32)

    # Weights to use
    entity_weights = nodes["entity_weights"]
    relation_weights = nodes["rel_weights"]

    relation_bias = nodes["rel_bias"]

    # Normalize e2 weights
    if hyperparams["test_normalize_e2"]:
        normalized_entity_weights = tf.nn.l2_normalize(entity_weights, dim=1)
    else:
        normalized_entity_weights = entity_weights

    # Normalize r weights
    if hyperparams["test_normalize_relation"]:
        normalized_relation_weights = tf.nn.l2_normalize(relation_weights, dim=1)
    else:
        normalized_relation_weights = relation_weights

    # Normalize e1 weights
    if hyperparams["test_normalize_e1"]:
        normalized_entity_weights = tf.nn.l2_normalize(entity_weights, dim=1)
    else:
        normalized_entity_weights = entity_weights

    # Embedding positive and negative samples
    pos_e1_emb = tf.nn.embedding_lookup(normalized_entity_weights, pos_e1)
    pos_r_emb = tf.nn.embedding_lookup(normalized_relation_weights, pos_r)
    pos_e2_emb = tf.nn.embedding_lookup(normalized_entity_weights, pos_e2)

    if hyperparams["bias"]:
        pos_score_bias = tf.reshape(tf.nn.embedding_lookup(relation_bias, pos_r), (-1, 1))
    else:
        pos_score_bias = 0
        neg_score_bias = 0

    # Reshape and transpose the movie weights and rating weights to a (1, dim, depth) tensor
    neg_e2_emb = tf.transpose(tf.reshape(
        normalized_entity_weights, (-1, hyperparams["emb_dim"], 1)),
        (2, 1, 0))
    neg_r_emb = tf.transpose(tf.reshape(
        relation_weights, (-1, hyperparams["emb_dim"], 1)),
        (2, 1, 0))

    # Scoring positive samples
    pos_scoring = tf.reduce_sum(
        tf.multiply(tf.multiply(pos_e1_emb, pos_r_emb), pos_e2_emb), axis=1, keep_dims=True) + pos_score_bias

    # Scoring negative samples with broadcasting
    pos_e1_r_mul = tf.multiply(pos_e1_emb, pos_r_emb)

    neg_scoring = tf.squeeze(tf.reduce_sum(
        tf.multiply(tf.reshape(pos_e1_r_mul, (-1, hyperparams["emb_dim"], 1)), neg_e2_emb),
        axis=1
    )) + pos_score_bias

    higher_values = tf.reduce_sum(tf.cast(neg_scoring > pos_scoring, tf.float32), axis=1)
    """

    return locals()


def test_graph(hyperparams, nodes, config=None):
    """
    nodes = scoring_and_counting(hyperparams, nodes, config=config)
    metric_values = {
        "MRR": metrics.mrr(nodes["higher_values"]),
        "HITS@10": metrics.hits_n(nodes["higher_values"], 10),
        "HITS@3": metrics.hits_n(nodes["higher_values"], 3),
        "HITS@1": metrics.hits_n(nodes["higher_values"], 1),
    }
    nodes.update(metric_values)

    summaries = [tf.summary.scalar(k, v) for k, v in metric_values.items()] + [
        tf.summary.histogram("score rankings", nodes["higher_values"])
    ]

    nodes["test_summary"] = tf.summary.merge(summaries)

    """
    return nodes


def debug_graph(hyperparams, nodes, config=None):
    """

    neg_r_emb_shape = tf.shape(nodes["neg_r_emb"])
    pos_scoring_shape = tf.shape(nodes["pos_scoring"])
    """

    return locals()
