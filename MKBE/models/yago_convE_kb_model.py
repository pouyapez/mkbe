import tensorflow as tf
from tensorpack import *
from tensorflow.contrib.keras import backend as K


class YAGOConveMultimodel(ModelDesc):
    def __init__(self, hyperparams):
        super(YAGOConveMultimodel, self).__init__()
        self.hyperparams = hyperparams

    def _get_inputs(self):
        return [InputDesc(tf.int32, (None,), "e1"),
                InputDesc(tf.int32, (None,), "r"),
                InputDesc(tf.int8, (None, self.hyperparams["entity_size"]), "e2_multihot"),
                InputDesc(tf.int32, (None,), "e2_ind")]

    def generate_onehot(self, indices):
        entity_size = self.hyperparams["entity_size"]
        return tf.one_hot(
            indices, entity_size, dtype=tf.float32, on_value=1.0 - self.hyperparams["label_smoothing"],
            off_value=self.hyperparams["label_smoothing"] / (entity_size - 1.0))

    def label_smoothing(self, onehots, lambd):
        e2 = tf.cast(onehots, tf.float32)
        e2_multi = (1.0 - lambd) * e2 + (10 * lambd / self.hyperparams["entity_size"])
        return e2_multi

    def _build_graph(self, inputs):
        hyperparams = self.hyperparams
        dtype = tf.float32
        id_dtype = tf.int32
        e1, r, e2_multihot, e2_ind = inputs

        label_smooth = tf.placeholder(tf.float32, name="label_smoothing", shape=())
        mlp_keepprob = tf.placeholder(tf.float32, name="mlp_keepprob")
        enc_keepprob = tf.placeholder(tf.float32, name="enc_keepprob")
        emb_keepprob = tf.placeholder(tf.float32, name="emb_keepprob")
        fm_keepprob = tf.placeholder(tf.float32, name="fm_keepprob")
        is_training = tf.placeholder(tf.bool, name="is_training")

        # Weights for embeddings
        if hyperparams["emb_dim"] > 3:
            self.entity_weights = tf.get_variable(
                "entity_weights", shape=[hyperparams["entity_size"], hyperparams["emb_dim"]], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=dtype))

            self.rel_weights = tf.get_variable(
                "relation_weights", shape=[hyperparams["relation_size"], hyperparams["emb_dim"]], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=dtype))

            self.word_weights = tf.get_variable(
                "word_weights", shape=[hyperparams["word_size"], hyperparams["emb_dim"] // 2],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=dtype))

        else:
            self.entity_weights = tf.get_variable(
                "entity_weights", shape=[hyperparams["entity_size"], hyperparams["emb_dim"]], dtype=dtype,
                initializer=tf.truncated_normal_initializer(dtype=dtype))
            self.rel_weights = tf.get_variable(
                "relation_weights", shape=[hyperparams["relation_size"], hyperparams["emb_dim"]], dtype=dtype,
                initializer=tf.truncated_normal_initializer(dtype=dtype))
            self.word_weights = tf.get_variable(
                "word_weights", shape=[hyperparams["word_size"], hyperparams["emb_dim"] // 2],
                dtype=dtype, initializer=tf.truncated_normal_initializer(dtype=dtype)
            )

        # Encode e1 and r
        e1_emb = tf.nn.embedding_lookup(self.entity_weights, e1)
        r_emb = tf.nn.embedding_lookup(self.rel_weights, r)

        # Collect Regularization variables
        regularized_variables = []

        # Aggregate and normalize e1
        e1_list = [e1_emb]

        if hyperparams["normalize_e1"]:
            Pos_e1 = tf.nn.l2_normalize(
                tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in e1_list], axis=0),
                dim=1)

        else:
            Pos_e1 = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in e1_list], axis=0)
            regularized_variables += [self.entity_weights]

        # Aggregate r
        r_list = [r_emb]

        if hyperparams["normalize_relation"]:
            Pos_r = tf.nn.l2_normalize(
                tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in r_list], axis=0),
                dim=1)

        else:
            Pos_r = tf.concat([tf.reshape(emb, (-1, hyperparams["emb_dim"])) for emb in r_list], axis=0)
            regularized_variables += [self.rel_weights]

        # ConvE link prediction
        with tf.variable_scope("convE"):
            emb_dim = hyperparams["emb_dim"]
            pose1_img = tf.reshape(Pos_e1, (-1, emb_dim // 10, 10, 1))
            posr_img = tf.reshape(Pos_r, (-1, emb_dim // 10, 10, 1))

            pos_stack = tf.layers.batch_normalization(tf.concat(
                [pose1_img, posr_img], 2), training=is_training, epsilon=1e-5, momentum=0.1)

            pos_indrop = tf.nn.dropout(pos_stack, emb_keepprob)

            convE_ker = tf.get_variable("convE_ker", shape=[3, 3, 1, 32], dtype=dtype,
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=dtype))
            convE_bias = tf.get_variable("convE_bias", shape=[32], dtype=dtype, initializer=tf.zeros_initializer)

            pos_convE_conv = tf.nn.relu(tf.layers.batch_normalization(tf.nn.bias_add(tf.nn.convolution(
                pos_indrop, convE_ker, "VALID"), convE_bias), training=is_training, epsilon=1e-5, momentum=0.1))

            fm_dropout = tf.contrib.keras.layers.SpatialDropout2D(1.0 - fm_keepprob)

            pos_flat = tf.reshape(fm_dropout(pos_convE_conv, training=is_training), (-1, 10368))

            self.convE_fc_w = tf.get_variable(
                "convE_fc_w", shape=[10368, hyperparams["emb_dim"]], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=dtype))
            self.convE_fc_b = tf.get_variable(
                "convE_fc_b", shape=[hyperparams["emb_dim"]], dtype=dtype,
                initializer=tf.constant_initializer(value=0.0))

            pos_fc = tf.nn.relu(tf.layers.batch_normalization(tf.nn.dropout(tf.nn.bias_add(tf.matmul(
                pos_flat, self.convE_fc_w), self.convE_fc_b), mlp_keepprob), training=is_training, epsilon=1e-5,
                momentum=0.1))

            self.pred = tf.matmul(pos_fc, self.entity_weights, transpose_b=True)

            regularized_variables += [convE_ker, convE_bias, self.convE_fc_w, self.convE_fc_b]

        # Generate e2 labels
        e2_label = self.label_smoothing(e2_multihot, label_smooth)

        # Sigmoid BCE loss/ sigmoid cross entropy
        #self.ll_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=e2_label, logits=self.pred))
        self.ll_loss = tf.reduce_mean(
            tf.losses.sigmoid_cross_entropy(e2_label, self.pred, reduction=tf.losses.Reduction.NONE))

        # Regularization term
        regularizer = tf.contrib.layers.l2_regularizer(hyperparams["regularization_coefficient"])
        regularization_term = tf.contrib.layers.apply_regularization(regularizer, regularized_variables)

        # Aggregate loss
        self.loss = tf.add(self.ll_loss, regularization_term, name="loss")
        self.cost = self.loss

        # Learning rate decay
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.maximum(tf.train.exponential_decay(
            hyperparams["learning_rate"], global_step / 15000, 1, hyperparams["lr_decay"]), 1e-7, name="lr")

        # Training op
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

        # Testing Graph
        self.test_graph(e2_ind)

        # Summaries
        self.summaries()

        return self.cost

    def test_graph(self, pos_e2):
        self.likelihood = tf.nn.sigmoid(self.pred)

        pos_score = tf.diag_part(tf.nn.embedding_lookup(tf.transpose(self.likelihood), pos_e2))
        cmp = tf.expand_dims(pos_score, axis=1) > self.likelihood
        self.rank = tf.reduce_sum(tf.cast(cmp, tf.int32), axis=1) + 1

        mrr = tf.reduce_mean(1.0 / tf.cast(self.rank, tf.float32), name="mrr")
        hits_10 = tf.reduce_mean(tf.cast(self.rank <= 10, tf.float32), name="hits_10")
        hits_3 = tf.reduce_mean(tf.cast(self.rank <= 3, tf.float32), name="hits_3")
        hits_1 = tf.reduce_mean(tf.cast(self.rank <= 1, tf.float32), name="hits_1")
        invalid_e2 = tf.reduce_mean(tf.cast(pos_e2 == 0, tf.float32), name="inv_e2")

        return mrr, hits_1, hits_3, hits_10

    def summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("bce_loss", self.ll_loss)
        tf.summary.histogram("logits", self.pred)
        tf.summary.histogram("rank", self.rank)
        tf.summary.histogram("probability", self.likelihood)
        tf.summary.histogram("entity weights", self.entity_weights)
        tf.summary.histogram("relation weights", self.rel_weights)
        tf.summary.histogram("dense weights", self.convE_fc_w)

    def _get_optimizer(self):
        return self.train_op, self.loss, self.ll_loss

