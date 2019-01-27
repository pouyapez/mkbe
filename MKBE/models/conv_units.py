"""CNN building blocks derived from Inception-ResNet-v2
"""

import tensorflow as tf


def print_variable_info():
    """
    auxiliary function to print trainable variable information
    """
    var_list = tf.trainable_variables()
    total = 0
    layer_name = ""
    layer_total = 0

    for var in var_list:
        num = 1
        for dim in var.shape:
            num *= int(dim)

        var_name_list = str(var.name).split('/')
        if var_name_list[0] != layer_name:
            if layer_total != 0:
                print("Layer {} total parameters: {}".format(layer_name, layer_total))
            print("---layer {} parameters---".format(var_name_list[0]))
            layer_total = 0
            layer_name = var_name_list[0]
        print("{}: {}, {}".format(var.name, str(var.shape), num))
        total += num
        layer_total += num
    print("Total parameters: {}".format(total))


def conv_bn_act(op, shape, stride, name, init, training, bn_momentum, act, padding='SAME'):
    """
    Build a convolution layer with batch normalization before activation
    :param op: input node
    :param shape: kernel shape
    :param stride: convolution stride
    :param name: node name
    :param init: initializer
    :param training: batch normalization training flag
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :param padding: padding requirement
    :return: post activation node
    """
    kernel = tf.get_variable("kernel_weights" + name, shape, initializer=init)
    conv = tf.nn.convolution(op, kernel, padding, strides=stride, name="conv" + name)
    bn = tf.layers.batch_normalization(conv, momentum=bn_momentum, training=training)
    post = act(bn)
    return post


def conv_bn(op, shape, stride, name, init, training, bn_momentum, padding='SAME'):
    """
    Build a convolution layer with batch normalization WITHOUT activation (i.e. affine linear layer)
    :param op: input node
    :param shape: kernel shape
    :param stride: convolution stride
    :param name: node name
    :param init: initializer
    :param training: batch normalization training flag
    :param bn_momentum: batch normalization momentum
    :param padding: padding requirement
    :return: post activation node
    """
    kernel = tf.get_variable("kernel_weights" + name, shape, initializer=init)
    conv = tf.nn.convolution(op, kernel, padding, strides=stride, name="conv" + name)
    bn = tf.layers.batch_normalization(conv, momentum=bn_momentum, training=training)
    return bn


def branch4_avgpool_5x5(
        ops,
        scope='branch4_avgpool_5x5',
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu,
        out_channel=32
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param init: initializer for convolution kernels
    :param training: training flag for batch_norm layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :param out_channel: output channels for each branch, advised to be smaller than or equal to input channels
    :return: output node
    """
    # get input channel number
    in_channel = ops.shape[-1]

    # convolution branches
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 1x1 branch
        branch0 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 4], [1, 1], "_0_a_1x1", init, training,
                              bn_momentum, act)

        # 1x1, 3x3 branch
        branch1 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 8], [1, 1], "_1_a_1x1", init, training,
                              bn_momentum, act)
        branch1 = conv_bn_act(branch1, [3, 3, out_channel // 8, out_channel // 4], [1, 1], "_1_b_3x3", init, training,
                              bn_momentum, act)

        # 1x1, 3x3, 3x3 branch
        branch2 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 8], [1, 1], "_2_a_1x1", init, training,
                              bn_momentum, act)
        branch2 = conv_bn_act(branch2, [3, 3, out_channel // 8, out_channel // 4], [1, 1], "_2_b_3x3", init, training,
                              bn_momentum, act)
        branch2 = conv_bn_act(branch2, [3, 3, out_channel // 4, out_channel // 4], [1, 1], "_2_c_3x3", init, training,
                              bn_momentum, act)

        # 3x3 avg_pool, 1x1 branch
        branch3 = tf.nn.avg_pool(ops, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
        branch3 = conv_bn_act(branch3, [1, 1, in_channel, out_channel // 4], [1, 1], "_3_a_1x1", init, training,
                              bn_momentum, act)

        # channel concatenation
        return tf.concat(axis=3, values=[branch0, branch1, branch2, branch3])


def branch4_res_5x5(
        ops,
        scope='branch4_res_5x5',
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu,
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param init: initializer for convolution kernels
    :param training: training flag for batch_norm layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :return: output node
    """
    # get input channel number
    in_channel = ops.shape[-1]

    # convolution tower
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 3 tower branches
        # 1x1 branch
        branch0 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 8], [1, 1], "_0_a_1x1", init, training, bn_momentum,
                              act)

        # 1x1, 3x3 branch
        branch1 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 8], [1, 1], "_1_a_1x1", init, training, bn_momentum,
                              act)
        branch1 = conv_bn_act(branch1, [3, 3, in_channel // 8, in_channel // 8], [1, 1], "_1_b_3x3", init, training,
                              bn_momentum, act)

        # 1x1, 3x3, 3x3 branch
        branch2 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 8], [1, 1], "_2_a_1x1", init, training, bn_momentum,
                              act)
        branch2 = conv_bn_act(branch2, [3, 3, in_channel // 8, in_channel * 3 // 16], [1, 1], "_2_b_3x3", init,
                              training, bn_momentum, act)
        branch2 = conv_bn_act(branch2, [3, 3, in_channel * 3 // 16, in_channel // 4], [1, 1], "_2_c_3x3", init,
                              training, bn_momentum, act)

        # tower top convolution
        concat = tf.concat(axis=3, values=[branch0, branch1, branch2])
        tower = conv_bn(concat, [1, 1, in_channel // 2, in_channel], [1, 1], "_tower_a_1x1", init, training,
                        bn_momentum)

        # residual summation
        return ops + tower


def branch3_maxpool_downsample_5x5(
        ops,
        scope='branch3_maxpool_downsample_5x5',
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param init: initializer for convolution kernels
    :param training: training flag for batch_norm layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :return: output node
    """
    # get input channel number
    in_channel = ops.shape[-1]

    # convolution branches
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 3x3 branch
        branch0 = conv_bn_act(ops, [3, 3, in_channel, in_channel // 2], [2, 2], "_0_a_3x3", init, training, bn_momentum,
                              act, 'VALID')

        # 1x1, 3x3, 3x3 branch
        branch1 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 4], [1, 1], "_1_a_1x1", init, training, bn_momentum,
                              act)
        branch1 = conv_bn_act(branch1, [3, 3, in_channel // 4, in_channel * 3 // 8], [1, 1], "_1_b_3x3", init, training,
                              bn_momentum, act)
        branch1 = conv_bn_act(branch1, [3, 3, in_channel * 3 // 8, in_channel // 2], [2, 2], "_1_c_3x3", init, training,
                              bn_momentum, act, 'VALID')

        # max pooling branch
        branch2 = tf.nn.max_pool(ops, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        # channel concatenation
        return tf.concat(axis=3, values=[branch0, branch1, branch2])


def branch4_avgpool_13x13(
        ops,
        scope='branch3_avgpool_13x13',
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu,
        out_channel=64
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param init: initializer for convolution kernels
    :param training: training flag for batch_norm layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :param out_channel: output channel number
    :return: output node
    """
    # get input channel number
    in_channel = ops.shape[-1]

    # convolution branches
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 1x1 branch
        branch0 = conv_bn_act(ops, [1, 1, in_channel, out_channel * 3 // 8], [1, 1], "_0_a_1x1", init, training,
                              bn_momentum, act)

        # 1x1, 1x7, 7x1 branch
        branch1 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 8], [1, 1], "_1_a_1x1", init, training,
                              bn_momentum, act)
        branch1 = conv_bn_act(branch1, [1, 7, out_channel // 8, out_channel * 3 // 16], [1, 1], "_1_b_1x7", init,
                              training, bn_momentum, act)
        branch1 = conv_bn_act(branch1, [7, 1, out_channel * 3 // 16, out_channel // 4], [1, 1], "_1_c_7x1", init,
                              training, bn_momentum, act)

        # 1x1, 7x1, 1x7, 7x1, 1x7 branch
        branch2 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 8], [1, 1], "_2_a_1x1", init, training,
                              bn_momentum, act)
        branch2 = conv_bn_act(branch2, [7, 1, out_channel // 8, out_channel * 3 // 16], [1, 1], "_2_b_7x1", init,
                              training, bn_momentum, act)
        branch2 = conv_bn_act(branch2, [1, 7, out_channel * 3 // 16, out_channel // 4], [1, 1], "_2_c_1x7", init,
                              training, bn_momentum, act)
        branch2 = conv_bn_act(branch2, [7, 1, out_channel // 4, out_channel // 4], [1, 1], "_2_d_7x1", init,
                              training, bn_momentum, act)
        branch2 = conv_bn_act(branch2, [1, 7, out_channel // 4, out_channel // 4], [1, 1], "_2_e_1x7", init,
                              training, bn_momentum, act)

        # avgpool, 1x1 branch
        branch3 = tf.nn.avg_pool(ops, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
        branch3 = conv_bn_act(branch3, [1, 1, in_channel, out_channel // 8], [1, 1], "_3_a_1x1", init, training,
                              bn_momentum, act)

        # channel concatenation
        return tf.concat(axis=3, values=[branch0, branch1, branch2, branch3])


def branch3_res_7x7(
        ops,
        scope='branch3_res_7x7',
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu,
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param init: initializer for convolution kernels
    :param training: training flag for batch_norm layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :return: output node
    """
    # get input channel number
    in_channel = ops.shape[-1]

    # convolution tower
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 3 tower branches
        # 1x1 branch
        branch0 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 4], [1, 1], "_0_a_1x1", init, training, bn_momentum,
                              act)

        # 1x1, 1x7, 7x1 branch
        branch1 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 8], [1, 1], "_1_a_1x1", init, training, bn_momentum,
                              act)
        branch1 = conv_bn_act(branch1, [1, 7, in_channel // 8, in_channel * 3 // 16], [1, 1], "_1_b_1x7", init,
                              training, bn_momentum, act)
        branch1 = conv_bn_act(branch1, [7, 1, in_channel * 3 // 16, in_channel // 4], [1, 1], "_1_c_7x1", init,
                              training, bn_momentum, act)

        # 1x1 tower branch
        concat = tf.concat(axis=3, values=[branch0, branch1])
        tower = conv_bn(concat, [1, 1, in_channel // 2, in_channel], [1, 1], "_tower_a_1x1", init, training,
                        bn_momentum)

        # residual summation
        return ops + tower


def branch3_maxpool_downsample_9x9(
        ops,
        scope='branch3_maxpool_downsample_9x9',
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param init: initializer for convolution kernels
    :param training: training flag for batch_norm layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :return: output node
    """
    # get input channel number
    in_channel = ops.shape[-1]

    # convolution branches
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 1x1, 3x3 branch
        branch0 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 8], [1, 1], "_0_a_1x1", init, training, bn_momentum,
                              act)
        branch0 = conv_bn_act(branch0, [3, 3, in_channel // 8, in_channel // 8], [2, 2], "_0_b_3x3", init, training,
                              bn_momentum, act, 'VALID')

        # 1x1, 1x7, 7x1, 3x3 branch
        branch1 = conv_bn_act(ops, [1, 1, in_channel, in_channel // 4], [1, 1], "_1_a_1x1", init, training, bn_momentum,
                              act)
        branch1 = conv_bn_act(branch1, [1, 7, in_channel // 4, in_channel // 4], [1, 1], "_1_b_1x7", init, training,
                              bn_momentum, act)
        branch1 = conv_bn_act(branch1, [7, 1, in_channel // 4, in_channel * 3 // 8], [1, 1], "_1_c_7x1", init, training,
                              bn_momentum, act)
        branch1 = conv_bn_act(branch1, [3, 3, in_channel * 3 // 8, in_channel * 3 // 8], [2, 2], "_1_d_3x3", init,
                              training, bn_momentum, act, 'VALID')

        # max pooling branch
        branch2 = tf.nn.max_pool(ops, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID')

        # channel concatenation
        return tf.concat(axis=3, values=[branch0, branch1, branch2])


def branch6_avgpool_5x5_downchannel(
        ops,
        scope='branch6_avgpool_5x5_downchannel',
        init=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu,
        out_channel=24
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param init: initializer for convolution kernels
    :param training: training flag for batch_norm layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :param out_channel: output channel number
    :return: output node
    """
    # get input channel number
    in_channel = ops.shape[-1]

    # convolution tower
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 1x1 branch
        branch0 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 6], [1, 1], "_0_a_1x1", init, training,
                              bn_momentum, act)

        # 1x1, (1x3, 3x1) branches
        branch1_2 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 4], [1, 1], "_1_2_a_1x1", init, training,
                                bn_momentum, act)
        branch1 = conv_bn_act(branch1_2, [1, 3, out_channel // 4, out_channel // 6], [1, 1], "_1_b_1x3", init, training,
                              bn_momentum, act)
        branch2 = conv_bn_act(branch1_2, [3, 1, out_channel // 4, out_channel // 6], [1, 1], "_2_b_3x1", init, training,
                              bn_momentum, act)

        # 1x1, 3x1, 1x3, (1x3, 3x1) branches
        branch3_4 = conv_bn_act(ops, [1, 1, in_channel, out_channel // 2], [1, 1], "_3_4_a_1x1", init, training,
                                bn_momentum, act)
        branch3_4 = conv_bn_act(branch3_4, [3, 1, out_channel // 2, out_channel // 3], [1, 1], "_3_4_b_3x1", init,
                                training, bn_momentum, act)
        branch3_4 = conv_bn_act(branch3_4, [1, 3, out_channel // 3, out_channel // 4], [1, 1], "_3_4_c_1x3", init,
                                training, bn_momentum, act)
        branch3 = conv_bn_act(branch3_4, [1, 3, out_channel // 4, out_channel // 6], [1, 1], "_3_d_1x3", init, training,
                              bn_momentum, act)
        branch4 = conv_bn_act(branch3_4, [3, 1, out_channel // 4, out_channel // 6], [1, 1], "_4_d_3x1", init, training,
                              bn_momentum, act)

        # avgpool, 1x1 branch
        branch5 = tf.nn.avg_pool(ops, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
        branch5 = conv_bn_act(branch5, [1, 1, in_channel, out_channel // 6], [1, 1], "_5_a_1x1", init, training,
                              bn_momentum, act)

        # channel concatenation
        return tf.concat(axis=3, values=[branch0, branch1, branch2, branch3, branch4, branch5])


def fc_dropout(
        ops,
        out_nodes,
        scope='fully_connected',
        init=tf.contrib.layers.xavier_initializer(uniform=False),
        training=True,
        bn_momentum=0.9,
        act=tf.nn.relu,
        keep_prob=0.9
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param out_nodes: output node number
    :param init: initializer for weights
    :param training: training flag for batch normalization layer
    :param bn_momentum: batch normalization momentum
    :param act: activation function
    :param keep_prob: keeping probability for dropout layer
    :return: output node
    """
    in_nodes = ops.shape[-1]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        weights = tf.get_variable("weights", [in_nodes, out_nodes], initializer=init)
        fc = tf.matmul(ops, weights)
        bn = tf.layers.batch_normalization(fc, momentum=bn_momentum, training=training)
        activations = act(bn)
        return tf.nn.dropout(activations, keep_prob)


def global_avg_dropout(
        ops,
        scope='global_avg_dropout',
        keep_prob=0.9
):
    """
    :param ops: input node
    :param scope: namescope of this layer
    :param keep_prob: keeping probability for dropout layer
    :return: output node
    """
    _, width, height, _ = ops.shape

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        avg = tf.nn.avg_pool(ops, [1, width, height, 1], [1, width, height, 1], 'VALID')
        return tf.nn.dropout(avg, keep_prob)


if __name__ == '__main__':
    input_node = tf.placeholder('float32', [16, 50, 50, 16])
    output_node = branch4_avgpool_5x5(input_node, out_channel=32)
    output_node = branch4_res_5x5(output_node)
    output_node = branch3_maxpool_downsample_5x5(output_node)
    output_node = branch4_avgpool_13x13(output_node)
    output_node = branch3_res_7x7(output_node)
    output_node = branch3_maxpool_downsample_9x9(output_node)
    output_node = branch6_avgpool_5x5_downchannel(output_node)
    print_variable_info()
    print(output_node.shape)
