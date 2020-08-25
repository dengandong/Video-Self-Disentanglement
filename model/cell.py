import tensorflow as tf
# from util import *


class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    """A LSTM cell with convolutions instead of multiplications.

  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

    def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True,
                 data_format='channels_last', reuse=None):
        super(ConvLSTMCell, self).__init__(_reuse=reuse)
        self._kernel = kernel  # kernel_size, e.g. [3, 3]
        self._filters = filters  # numbers of the input channels
        self._forget_bias = forget_bias
        self._activation = activation
        self._normalize = normalize
        self._peephole = peephole
        self._channels = [64, 128, 256]
        if data_format == 'channels_last':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = tf.TensorShape([self._filters] + shape)
            self._feature_axis = 0
            self._data_format = 'NCHW'
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

    @property
    def output_size(self):
        return self._size

    def call(self, x, state):
        c, h = state

        x = tf.concat([x, h], axis=self._feature_axis)
        # n = x.shape[-1].value
        m = 4 * self._filters if self._filters > 1 else 4
        # W = tf.get_variable('kernel', self._kernel + [n, m])  # e.g. [3, 3, n, m]
        # y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
        # if not self._normalize:
        #   y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
        with tf.variable_scope('block'):
            y_conv = conv_in_cell(x, kernel_size=self._kernel,
                                  out_channels=self._channels,
                                  name='conv', _data_format=self._data_format, normalize=self._normalize)

        y = conv_in_cell(y_conv, kernel_size=self._kernel, out_channels=[m],
                         _data_format=self._data_format, normalize=self._normalize, pool=False)

        j, i, f, o = tf.split(y, 4, axis=self._feature_axis)  # (128, 224, 224, 12) --> 4 * (128, 224, 224, 3)

        if self._peephole:
            i += tf.get_variable('W_ci', c.shape[1:]) * c  # c-->(128, 224, 224, 3)
            f += tf.get_variable('W_cf', c.shape[1:]) * c

        if self._normalize:
            j = tf.contrib.layers.layer_norm(j)
            i = tf.contrib.layers.layer_norm(i)
            f = tf.contrib.layers.layer_norm(f)

        f = tf.sigmoid(f + self._forget_bias)
        i = tf.sigmoid(i)
        c = c * f + i * self._activation(j)

        if self._peephole:
            o += tf.get_variable('W_co', c.shape[1:]) * c

        if self._normalize:
            o = tf.contrib.layers.layer_norm(o)
            c = tf.contrib.layers.layer_norm(c)

        o = tf.sigmoid(o)
        h = o * self._activation(c)

        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""

    def __init__(self, shape, filters, kernel, activation=tf.tanh, normalize=True, data_format='channels_last',
                 reuse=None):
        super(ConvGRUCell, self).__init__(_reuse=reuse)
        self._filters = filters
        self._kernel = kernel
        self._activation = activation
        self._normalize = normalize
        if data_format == 'channels_last':
            self._size = tf.TensorShape(shape + [self._filters])
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = tf.TensorShape([self._filters] + shape)
            self._feature_axis = 0
            self._data_format = 'NC'
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def call(self, x, h):
        channels = x.shape[self._feature_axis].value

        with tf.variable_scope('gates'):
            inputs = tf.concat([x, h], axis=self._feature_axis)
            n = channels + self._filters
            m = 2 * self._filters if self._filters > 1 else 2
            W = tf.get_variable('kernel', self._kernel + [n, m])
            y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
            if self._normalize:
                r, u = tf.split(y, 2, axis=self._feature_axis)
                r = tf.contrib.layers.layer_norm(r)
                u = tf.contrib.layers.layer_norm(u)
            else:
                y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
                r, u = tf.split(y, 2, axis=self._feature_axis)
            r, u = tf.sigmoid(r), tf.sigmoid(u)

        with tf.variable_scope('candidate'):
            inputs = tf.concat([x, r * h], axis=self._feature_axis)
            n = channels + self._filters
            m = self._filters
            W = tf.get_variable('kernel', self._kernel + [n, m])
            y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
            if self._normalize:
                y = tf.contrib.layers.layer_norm(y)
            else:
                y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
            h = u * h + (1 - u) * self._activation(y)

        return h, h


def conv_in_cell(x, kernel_size, out_channels, _data_format, name='conv', normalize=True, pool=False):
    n = x.shape[-1].value
    depth = len(out_channels)
    for i in range(depth):
        with tf.variable_scope(name + '_{}'.format(i), reuse=tf.AUTO_REUSE):
            weight = tf.get_variable('weight', shape=kernel_size + [n, out_channels[i]])
            bias = tf.get_variable('bias', [out_channels[i]], initializer=tf.zeros_initializer())
            x = tf.nn.convolution(x, weight, padding='SAME', data_format=_data_format)
            n = x.shape[-1].value
            if not normalize:
                x += bias
    if pool:
        conv_out = tf.nn.avg_pool2d(x, 2, strides=[1, 2, 2, 1], padding='VALID')
    else:
        conv_out = x
    return conv_out


if __name__ == '__main__':
    sequences = tf.random.normal([128, 32, 224, 224, 3])  # N x T x H x W x C
    batch_size, time_steps, height, width, channels = sequences.get_shape()

    f_cell = ConvLSTMCell(shape=[height, width], kernel=[3, 3], filters=3)
    b_cell = ConvLSTMCell(shape=[height, width], kernel=[3, 3], filters=3)

    with tf.variable_scope("bi_conv_lstm", reuse=tf.AUTO_REUSE):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, sequences, dtype=tf.float32)

    fw_output, bw_output = outputs
    output = fw_output + bw_output  # (128, 32, 224, 224, 3)

    fw_state, bw_state = states  # (128, 224, 224, 3)
    fw_cell, fw_hidden = fw_state
    bw_cell, bw_hidden = bw_state
