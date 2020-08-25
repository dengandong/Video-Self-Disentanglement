import tensorflow as tf


def gap_frames(frames):
    avg_frame = tf.reduce_mean(frames, axis=1)
    return avg_frame


def reduce_dim(input_, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', [1, 1, 1, input_.get_shape()[-1], 1],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [input_.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv3d(input_, weight, strides=[1, 1, 1, 1, 1], padding="SAME") + bias
        # out = tf.layers.batch_normalization(out, axis=-1)  # axis = ?
        # out = tf.nn.relu(out)
        return out


def conv2d(input_, output_channels, kernel_size, name="conv2d"):
    input_channels = input_.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', [kernel_size, kernel_size, input_channels, output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels], initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv2d(input_, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
        out = tf.layers.batch_normalization(out, axis=-1)  # axis = ?
        out = tf.nn.relu(out)
        return out
    pass


def deconv2d(input_, output_channels, kernel_size, batch_size, ratio=2, name='deconv3d'):
    _, height, width, input_channels = input_.get_shape()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', [kernel_size, kernel_size, output_channels, input_channels])
        bias = tf.get_variable('bias', [output_channels], initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv2d_transpose(input_, weight,
                                     output_shape=tf.TensorShape([batch_size, ratio * height, ratio * width, output_channels]),
                                     strides=[1, 2, 2, 1], padding="SAME") + bias
        out = tf.layers.batch_normalization(out, axis=-1)
        out = tf.nn.relu(out)
    return out


def conv3d(input_, output_channels, kernel_size, name="conv3d"):
    input_channels = input_.get_shape()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', [1, kernel_size, kernel_size, input_channels, output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels], initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv3d(input_, weight, strides=[1, 1, 1, 1, 1], padding="SAME") + bias
        out = tf.layers.batch_normalization(out, axis=-1)  # axis = ?
        out = tf.nn.relu(out)
        return out


def deconv3d(input_, output_channels, kernel_size, batch_size, ratio=2, name='deconv3d'):
    batch_size, time_steps, height, width, input_channels = input_.get_shape()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weight = tf.get_variable('weight', [1, kernel_size, kernel_size, output_channels, input_channels])
        bias = tf.get_variable('bias', [output_channels], initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv3d_transpose(input_, weight,
                                     output_shape=tf.TensorShape([batch_size, time_steps, ratio*height, ratio*width, output_channels]),
                                     strides=[1, 1, 2, 2, 1], padding="SAME") + bias
        out = tf.layers.batch_normalization(out, axis=-1)
        out = tf.nn.relu(out)
    return out


def pool2d(input_, name):
    output = tf.nn.avg_pool2d(input_, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name=name)
    return output


def pool3d(input_, name):
    output = tf.nn.avg_pool3d(input_, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='VALID', name=name)
    return output


def last_dim_expand_conv(x):
    channels = x.get_shape()[-1]
    weight = tf.get_variable('weight', [1, 1, channels, 2 * channels])
    output = tf.nn.conv2d(x, weight, padding='SAME')  # (128, 224, 224, 6)
    return output


def last_dim_expand_linear(x):
    channels = x.get_shape()[-1]
    output = tf.layers.dense(x, 2 * channels)
    return output


def dim_expand_3d(x, k=16):
    # 3d tensor to 5d tensor
    x = tf.expand_dims(x, axis=2)
    x = tf.tile(x, [1, 1, k, 1])
    x = tf.expand_dims(x, axis=2)
    x = tf.tile(x, [1, 1, k, 1, 1])
    return x


def dim_expand_2d(x, k=16):
    # 2d tensor to 4d tensor
    x = tf.expand_dims(x, axis=1)
    x = tf.tile(x, [1, k, 1])
    x = tf.expand_dims(x, axis=1)
    x = tf.tile(x, [1, k, 1, 1])
    return x


def rnn_inference(frames, lstm_cell, name, reverse=False, mode='conv'):
    batch_size, time_steps, channels = frames.get_shape()[0], frames.get_shape()[1], frames.get_shape()[-1]
    # lstm_cell = cell.ConvLSTMCell(shape=[height, width], filters=channels, kernel=[3, 3], normalize=True)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    step_state = initial_state
    outputs = []
    # states = []

    if reverse:
        start, end, stride, index = time_steps - 1, -1, -1, 0
    else:
        start, end, stride, index = 0, time_steps, 1, time_steps - 1

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for t in range(start, end, stride):
            step_output, step_state = lstm_cell(frames[:, t], step_state)  # state-->LSTMStateTuple(c, h)
            # 1x1 convolution to expand the last dimension for disentanglement
            if mode == 'conv':
                step_output = last_dim_expand_conv(step_output)  # (128, 224, 224, 6)
            else:
                step_output = last_dim_expand_linear(step_output)
            if t == index:
                states = step_state
            outputs.append(step_output)
            # states.append(state[1])
        outputs = tf.stack(outputs, axis=1)
        # states = tf.stack(states, axis=1)

    return outputs, states


def cnn_2d_inference(frames, name='cnn_2d_inference'):
    """
    Compress the input frame to temporal vector.
    :param frames: [batch, time_steps, height, width, channel]
    :param name: str
    :return: [batch, time_steps, dims]
    """
    # batch_size, time_steps, height, width, channels = frames.get_shape()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1_1 = conv2d(frames, 64, 3, name='conv1_1')
        conv1_2 = conv2d(conv1_1, 64, 3, name='conv1_2')
        conv1_3 = conv2d(conv1_2, 64, 3, name='conv1_3')
        conv1 = pool2d(conv1_3, name='pool1')  # [batch, time_step, 112, 112, 64]

        conv2_1 = conv2d(conv1, 128, 3, name='conv2_1')
        conv2_2 = conv2d(conv2_1, 128, 3, name='conv2_2')
        conv2_3 = conv2d(conv2_2, 128, 3, name='conv2_3')
        conv2 = pool2d(conv2_3, name='pool2')  # [batch, time_step, 56, 56, 128]

        conv3_1 = conv2d(conv2, 256, 3, name='conv3_1')
        conv3_2 = conv2d(conv3_1, 256, 3, name='conv3_2')
        conv3_3 = conv2d(conv3_2, 512, 3, name='conv3_3')
        conv3 = pool2d(conv3_3, name='pool3')  # [batch, time_step, 28, 28, 512]

        # flatten = tf.reshape(conv3, [batch_size, time_steps, -1])
        # out = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
        out = tf.reduce_mean(conv3, axis=-2)
        out = tf.reduce_mean(out, axis=-2)  # [batch, time_step, 512]
        return out


def cnn_2d_decode_inference(code, batch_size, name='deconv2d_inference'):  # (128, 32, 600)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        code = tf.layers.dense(code, 64, activation=tf.nn.relu)  # (128, 512)
        code = dim_expand_2d(code)  # (128, 16, 16, 512)

        deconv3_3 = deconv2d(code, 64, 3, batch_size, name='deconv3_3')  # 32
        deconv3_2 = conv2d(deconv3_3, 64, 3, name='deconv3_2')
        deconv3_1 = conv2d(deconv3_2, 64, 3, name='deconv3_1')

        deconv2_3 = deconv2d(deconv3_1, 64, 3, batch_size, name='deconv2_3')  # 64
        deconv2_2 = conv2d(deconv2_3, 64, 3, name='deconv2_2')
        deconv2_1 = conv2d(deconv2_2, 64, 3, name='deconv2_1')

        deconv1_3 = deconv2d(deconv2_1, 64, 3, batch_size, name='deconv1_3')
        deconv1_2 = conv2d(deconv1_3, 64, 3, name='deconv1_2')
        deconv1_1 = conv2d(deconv1_2, 64, 3, name='deconv1_1')

        output = conv2d(deconv1_1, 3, 3, name='rgb')
    return output


def cnn_inference(frames, name='conv3d_inference'):
    """
    Compress the input frame to temporal vector.
    :param frames: [batch, time_steps, height, width, channel]
    :param name: str
    :return: [batch, time_steps, dims]
    """
    # batch_size, time_steps, height, width, channels = frames.get_shape()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv1_1 = conv3d(frames, 64, 3, name='conv1_1')
        conv1_2 = conv3d(conv1_1, 64, 3, name='conv1_2')
        conv1_3 = conv3d(conv1_2, 64, 3, name='conv1_3')
        conv1 = pool3d(conv1_3, name='pool1')  # [batch, time_step, 112, 112, 64]

        conv2_1 = conv3d(conv1, 64, 3, name='conv2_1')
        conv2_2 = conv3d(conv2_1, 64, 3, name='conv2_2')
        conv2_3 = conv3d(conv2_2, 64, 3, name='conv2_3')
        conv2 = pool3d(conv2_3, name='pool2')  # [batch, time_step, 56, 56, 128]

        conv3_1 = conv3d(conv2, 64, 3, name='conv3_1')
        conv3_2 = conv3d(conv3_1, 64, 3, name='conv3_2')
        conv3_3 = conv3d(conv3_2, 64, 3, name='conv3_3')
        conv3 = pool3d(conv3_3, name='pool3')  # [batch, time_step, 28, 28, 512]

        # flatten = tf.reshape(conv3, [batch_size, time_steps, -1])
        # out = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
        out = tf.reduce_mean(conv3, axis=-2)
        out = tf.reduce_mean(out, axis=-2)  # [batch, time_step, 512]
        return out


def cnn_decode_inference(code, batch_size, name='deconv3d_inference'):  # (128, 32, 600)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        code = tf.layers.dense(code, 64, activation=tf.nn.relu)
        code = dim_expand_3d(code)  # (128, 32, 28, 28, 512)

        deconv3_3 = deconv3d(code, 64, 3, batch_size, name='deconv3_3')  # 56
        deconv3_2 = conv3d(deconv3_3, 64, 3, name='deconv3_2')
        deconv3_1 = conv3d(deconv3_2, 64, 3, name='deconv3_1')

        deconv2_3 = deconv3d(deconv3_1, 64, 3, batch_size, name='deconv2_3')  # 112
        deconv2_2 = conv3d(deconv2_3, 64, 3, name='deconv2_2')
        deconv2_1 = conv3d(deconv2_2, 64, 3, name='deconv2_1')

        deconv1_3 = deconv3d(deconv2_1, 64, 3, batch_size, name='deconv1_3')  # 224
        deconv1_2 = conv3d(deconv1_3, 64, 3, name='deconv1_2')
        deconv1_1 = conv3d(deconv1_2, 64, 3, name='deconv1_1')

        output = conv3d(deconv1_1, 3, 3, name='rgb')
    return output


class TemporalAttention:
    """
    TODO
    add attention to the decoder to generate more realistic frames
    impose attention on time dimension
    """

    def __init__(self, hidden_units, temperature, activation):
        self.hidden = hidden_units
        self.temperature = temperature
        self.activation = activation

    def forward(self, query, key, mode='conv'):
        """
        TODO
        :param query:
        :param key:
        :param mode:
        :return:
        """
        if mode == 'conv':
            q_feature = self.activation(conv3d(query))  # wrong !
            k_feature = self.activation(conv3d(key))
            # v_feature = self.activation(value)
        else:
            q_feature = self.activation(query)
            k_feature = self.activation(key)

        qk_feature = tf.matmul(q_feature, tf.transpose(k_feature, [0, 1, 3, 2]))
        score = tf.nn.softmax(qk_feature, self.temperature)
        return score


def dilated_plus(fea1, fea2, fea3, times=16):
    res = []
    for i in range(times):
        temp = fea1[:, i]
        if i % 3 == 0:
            temp += fea2[:, i//3]
        if i % 5 == 0:
            temp += fea3[:, i//5]
        res.append(temp)
    res = tf.stack(res, axis=1)
    return res
