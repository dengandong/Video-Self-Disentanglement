from model import *


class MotionEncoderC3D:
    def __init__(self, hidden_units=512, name='motion_encoder_2'):
        self.units = hidden_units
        self.name = name
        pass

    def forward(self, input_frames):
        with tf.name_scope(self.name):
            code = cnn_inference(input_frames, name='c3d')
            code = tf.layers.dense(code, self.units)
            return code


class ContentEncoder:
    def __init__(self, batch_size, units=256, name='content_encoder_2'):
        self.hidden_units = units
        self.name = name
        self.batch_size = batch_size
        pass

    def forward(self, frames):
        """
        eliminate the temporal feature of the input frames by global pooling on time axis
        """
        with tf.name_scope(self.name):
            # global low_level_representation, semantic_representation
            reduce_frames = gap_frames(frames)
            project_frames = cnn_2d_inference(reduce_frames, name='gap_conv')  # (128, 1, 512)
            content_code = tf.layers.dense(project_frames, self.hidden_units, activation=tf.nn.relu)
            reduce_reconstruction = cnn_2d_decode_inference(content_code, batch_size=self.batch_size, name='gap_recon')

            content_code = tf.expand_dims(content_code, axis=1)
            content_code = tf.tile(content_code, [1, 16, 1])

            return content_code, reduce_frames, reduce_reconstruction


class Decoder:
    def __init__(self, batch_size, name='decoder_2'):
        self.name = name
        self.batch_size = batch_size
        pass

    def forward(self, code):
        with tf.name_scope(self.name):
            reconstruction = cnn_decode_inference(code, batch_size=self.batch_size, name='deconv_c3d')
            return reconstruction