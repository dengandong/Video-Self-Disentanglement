from model import *


class Discriminator:
    """
    Better use the Dual-Video-Discriminator
    """
    def __init__(self):
        self.hidden_units = 256
        pass

    def forward(self, frames):
        project_frames = cnn_inference(frames, name='discriminator')  # (128, 1, 512)
        content_code = tf.layers.dense(project_frames, self.hidden_units, activation=tf.nn.relu)

        return content_code

    def conv_block(self):
        pass
