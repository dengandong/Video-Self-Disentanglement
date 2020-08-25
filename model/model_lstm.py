import tensorflow as tf

from model.model_base import ModelBase
from model.autoencoder.ae_lstm import MotionEncoder, ContentEncoder, Decoder
from model.discriminator import Discriminator


class ModelLSTM(ModelBase):
    """
    Deterministic phase: reconstruct the input frames from the reorganized codes consisting
    content information and motion information
    """
    name = 'model_lstm'

    def __init__(self):
        super(ModelLSTM, self).__init__()

    def forward(self, input_frames):
        # obtain content code
        self.z_c, self.gap, self.reconstructed_gap = self.CE.forward(input_frames)

        # obtain motion code
        # (128, 16, 128, 128, 6), ((128, 128, 128, 3), (128, 128, 128, 3))
        self.z_m, state = self.ME.forward(input_frames)
        batch_size, time_step, channels = self.z_m.get_shape()
        # reparametrization trick
        self.z_mu = self.z_m[:, :, :channels//2]
        self.z_sigma = tf.sqrt(tf.exp(self.z_m[:, :, channels//2:]))
        # self.z_mu = self.z_m[:, :, :, :, :channels//2]
        # self.z_sigma = tf.sqrt(tf.exp(self.z_m[:, :, :, :, channels//2:]))

        # deterministic code
        self.z = self.z_mu + self.z_sigma * self.z_c
        # deterministic reconstruction
        self.reconstructed_frames = self.G.forward(self.z, state)

        # variational inference
        epsilon = tf.random.normal([batch_size, time_step, channels//2])
        z_v = self.z_mu + self.z_sigma * epsilon
        self.synthesized_frames = self.G.forward(z_v, state)
        self.z_m_1 = self.ME.forward(self.synthesized_frames)
        pass


if __name__ == '__main__':
    inputs = tf.random.normal([128, 16, 128, 128, 3])

    ME = MotionEncoder()
    CE = ContentEncoder()
    G = Decoder()
    D = Discriminator()
    model = ModelLSTM(ME, CE, G, D)

    model.forward(inputs)
    print(model.z_m.get_shape())
    print(model.synthesized_frames.get_shape())
