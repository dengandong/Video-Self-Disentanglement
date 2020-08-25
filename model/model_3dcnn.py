import argparse

from model import *
from model.model_base import ModelBase
from model.autoencoder.ae_3dcnn import MotionEncoderC3D, ContentEncoder, Decoder
from model.discriminator import Discriminator


class Model3D(ModelBase):
    name = 'model_3dcnn'

    def __init__(self,
                 args,
                 motion_encoder,
                 content_encoder,
                 generator,
                 discriminator,
                 is_training=True):
        super(Model3D, self).__init__(is_training=is_training, args=args, discriminator=discriminator)

        self.ME = motion_encoder
        self.CE = content_encoder
        self.G = generator

        self.batch_size = args.batch_size

    def forward(self, input_frames):
        self.z_m = self.ME.forward(input_frames)
        _, time_step, channels = self.z_m.get_shape()
        print('channel:', channels)
        print(self.z_m)
        self.z_mu = self.z_m[:, :, :channels//2]
        self.z_sigma = tf.sqrt(tf.exp(self.z_m[:, :, channels//2:]))

        # deterministic reconstruction part
        self.z_c, self.gap, self.reconstructed_gap = self.CE.forward(input_frames)
        self.z = self.z_mu + self.z_sigma * self.z_c
        self.reconstructed_frames = self.G.forward(self.z)

        # variational inference
        epsilon = tf.random.normal([self.batch_size, time_step, channels//2])
        z_v = self.z_mu + self.z_sigma * epsilon

        # variational generation
        self.synthesized_frames = self.G.forward(z_v)
        self.z_m_1 = self.ME.forward(self.synthesized_frames)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    inputs = tf.random.normal([30, 16, 128, 128, 3])
    ME = MotionEncoderC3D(hidden_units=512)
    CE = ContentEncoder()
    G = Decoder()
    D = Discriminator()
    model = Model3D(args, ME, CE, G, D)

    model.forward(inputs)
    print('motion code:', model.z_m.get_shape())
    print('generated frames:', model.synthesized_frames.get_shape())
