import numpy as np
from model import *

from model import cell


class MotionEncoder:
    def __init__(self, units=600, encoder_mode='conv2lstm', name='motion_encoder'):
        self.hidden_units = units
        self.encoder_mode = encoder_mode
        self.name = name
        pass

    def forward(self, frames):
        """
        coarsely utilize residual connection to disentangle the input to semantic feature and low-level feature,
        """
        # global low_level_representation, semantic_representation
        batch_size, time_steps, height, width, channels = frames.get_shape()

        with tf.name_scope(name=self.name):
            if self.encoder_mode == 'ConvLSTM':
                conv_lstm_cell = cell.ConvLSTMCell(shape=[height, width], filters=channels, kernel=[3, 3], normalize=True)
                fw_output, fw_state = rnn_inference(frames, conv_lstm_cell, 'forward_bilstm', False, 'conv')
                bw_output, bw_state = rnn_inference(frames, conv_lstm_cell, 'backward_bilstm', True, 'conv')
                # aggregate the output
                encoder_output = fw_output + bw_output  # (128, 32, 224, 224, 6)

                # construct encoder state, but check whether it is valid to directly add the cell and hidden
                encoder_cell = fw_state[0] + bw_state[0]
                encoder_hidden = fw_state[1] + bw_state[1]
                encoder_state = tf.nn.rnn_cell.LSTMStateTuple(encoder_cell, encoder_hidden)
                print(encoder_state)

            # use 3d convolutional block before the bidirectional lstm
            elif self.encoder_mode == 'conv2lstm':
                conv_out = cnn_inference(frames, name='conv')  # [batch, time, 512]
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units)
                fw_output, fw_state = rnn_inference(conv_out, lstm_cell, 'forward_bilstm', False, 'linear')
                bw_output, bw_state = rnn_inference(conv_out, lstm_cell, 'backward_bilstm', True, 'linear')

                # aggregate the output
                encoder_output = fw_output + bw_output  # (128, 32, 1200)

                # construct encoder state, but check whether it is valid to directly add the cell and hidden
                encoder_cell = fw_state[0] + bw_state[0]
                encoder_hidden = fw_state[1] + bw_state[1]
                encoder_state = tf.nn.rnn_cell.LSTMStateTuple(encoder_cell, encoder_hidden)

            elif self.encoder_mode == 'C3D':
                pass

        return encoder_output, encoder_state

    @staticmethod
    def frame_random_miss(frames, random_ratio=0.1, mode='blank'):
        time_index = frames.get_shape()[0]
        spatial_shape = frames.get_shape()[1:]
        sample_number = time_index * random_ratio
        sampled_index = np.random.randint(0, time_index, sample_number)

        if mode == 'blank':
            frames[sampled_index, :, :, :] = tf.zeros(spatial_shape)
        elif mode == 'noisy':
            frames[sampled_index, :, :, :] = tf.random.normal([spatial_shape])
        elif mode == 'mask':
            pass

        return frames


class ContentEncoder:
    def __init__(self, units=600):
        self.hidden_units = units
        pass

    def forward(self, frames):
        """
        eliminate the temporal feature of the input frames by global pooling on time axis
        """
        # global low_level_representation, semantic_representation
        reduce_frames = gap_frames(frames)  # (128, 1, 224, 224, 3)
        project_frames = cnn_2d_inference(reduce_frames, name='pre_conv')  # (128, 1, 512)
        content_code = tf.layers.dense(project_frames, self.hidden_units, activation=tf.nn.relu)
        reduce_reconstruction = cnn_2d_decode_inference(content_code)

        content_code = tf.expand_dims(content_code, axis=1)
        content_code = tf.tile(content_code, [1, 16, 1])

        return content_code, reduce_frames, reduce_reconstruction


class Decoder:
    def __init__(self, hidden_units=600, decoder_mode='conv2lstm', name='decoder'):
        self.decoder_mode = decoder_mode
        self.units = hidden_units
        self.name = name

    def forward(self, enc_output, enc_state):
        with tf.name_scope(self.name):
            if self.decoder_mode == 'ConvLSTM':
                batch_size, time_steps, height, width, channels = enc_output.get_shape()  # (128, 32, 224, 224, 3)
                decoder_cell = cell.ConvLSTMCell([height, width], channels, [3, 3])
                with tf.variable_scope('conv_decode_rnn'):
                    dec_output, dec_state = tf.nn.dynamic_rnn(decoder_cell, enc_output, initial_state=enc_state)
            elif self.decoder_mode == 'conv2lstm':  # enc_output: (128, 32, 600)
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.units)
                with tf.variable_scope('linear_decode_rnn'):
                    dec_output, dec_state = tf.nn.dynamic_rnn(decoder_cell, enc_output, initial_state=enc_state)
                with tf.variable_scope('decode_cnn'):
                    dec_output = cnn_decode_inference(dec_output, 'cnn_decoder')

        return dec_output, dec_state