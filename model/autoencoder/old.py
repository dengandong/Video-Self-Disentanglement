from abc import ABC, abstractmethod
import tensorflow as tf


class BaseAE(ABC):
    name = 'base_autoencoder'

    def __init__(self, is_training=True):
        self.is_training = is_training
        pass

    @abstractmethod
    def build_model(self, _input):
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_frames):
        raise NotImplementedError

    @abstractmethod
    def train(self, sess, feed_dict, step, batch_size):
        pass

    @abstractmethod
    def compute_loss(self):
        raise NotImplementedError


class DeterministicAE(BaseAE):
    """
    Deterministic phase: reconstruct the input frames from the reorganized codes consisting
    content information and motion information
    """
    name = 'deterministic_autoencoder'

    def __init__(self, motion_encoder, content_encoder, decoder, input_frames, is_training=True):
        """
        :param motion_encoder:
        :param content_encoder:
        :param decoder: generator
        """
        super(DeterministicAE, self).__init__(is_training=is_training)
        self.motion_encoder = motion_encoder
        self.content_encoder = content_encoder
        self.decoder = decoder

        self.true = input_frames  # input
        self.z_m = None  # motion code
        self.z_c = None  # content code
        self.z = None  # complete code
        self.gap = None  # reduced frames by operating gap on input
        self.reconstructed_gap = None  # reconstructed reduced frames
        self.reconstructed_frames = None  # reconstructed input

        self.loss = None

    def build_model(self, _input, _code):
        pass

    def forward(self, input_frames):
        self.z_c, self.gap, self.reconstructed_gap = self.content_encoder.forward(input_frames)
        self.z_m = self.motion_encoder.forward(input_frames)
        self.z = self.z_m + self.z_c  # any other aggregation methods?
        self.reconstructed_frames = self.decoder.forward(self.z)
        pass

    def train(self, sess, feed_dict, step, batch_size):
        # TODO
        train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.999).minimize(self.loss)
        _loss, _ = sess.run([self.loss, train_op], feed_dict=feed_dict)
        pass

    def compute_loss(self):
        """
        reconstruction loss to make the latent code z capture all the necessary information of the input frames
        """
        reconstruction_loss = tf.reduce_mean(tf.square(self.true, self.reconstructed_frames))
        gap_reconstruction_loss = tf.reduce_mean(tf.square(self.gap, self.reconstructed_gap))
        self.loss = reconstruction_loss + gap_reconstruction_loss
        pass


class FlippedAE(BaseAE):
    """
    Variational process which reconstructs motion code in a flipped auto-encoder
    """
    name = 'flipped_autoencoder'

    def __init__(self, encoder, decoder, discriminator, is_training=True):
        """
        :param encoder: motion_encoder
        :param decoder: generator
        """
        super(FlippedAE, self).__init__(is_training=is_training)
        self.encoder = encoder
        self.decoder = decoder
        self.D = discriminator

        # variables in function forward
        self.z_mu = None
        self.z_sigma = None
        self.z_m_1 = None
        self.synthesized_frames = None

        # model input variables
        self.real_frames = None
        self.input_code = None

        self.loss = None

    def build_model(self, _input, _code):
        self.real_frames = _input
        self.input_code = _code

        self.forward(self.input_code)
        if self.is_training:
            self.compute_loss()
        pass

    def forward(self, z_m):
        batch_size, height, width, channels = z_m.get_shape()  # (32, 224, 224, 3) ---> (32, 28, 28, 256)
        self.z_mu = z_m[:, :, :, :channels]
        self.z_sigma = tf.sqrt(tf.exp(z_m[:, :, :, channels:]))
        epsilon = tf.random.normal([batch_size, height, width, channels])
        z_v = self.z_mu + self.z_sigma * epsilon

        synthesized_frames = self.decoder.forward(z_v)
        self.z_m_1 = self.encoder.forward(synthesized_frames)
        pass

    def train(self, sess, feed_dict, step, batch_size):
        # TODO
        train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.999).minimize(self.loss)
        _loss, _ = sess.run([self.loss, train_op], feed_dict=feed_dict)
        pass

    def compute_loss(self, coefficient=[0.1, 0.1, 1]):
        ad_loss = self.compute_adversarial_loss(self.real_frames, self.synthesized_frames)
        reg_loss = self.compute_regularization_loss()
        code_loss = self.compute_representation_loss()
        self.loss = coefficient[0] * ad_loss + coefficient[1] * reg_loss + coefficient[2] * code_loss
        pass

    def compute_adversarial_loss(self, real, generated):
        """
        adversarial loss to produce realistic synthetic images after decoder
        """
        real_score = self.D.forward(real)
        false_score = self.D.forward(generated)

        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_score), logits=real_score)
        )
        false_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(false_score), logits=false_score)
        )
        ad_loss = real_loss + false_loss
        return ad_loss

    def compute_regularization_loss(self):
        """
        regularization loss to force the posterior distribution closer to Gaussian, i.e., reduce the KL-divergence
        between the two. (VAE loss)
        """
        kl_divergence = tf.reduce_sum(
            tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1, axis=1
        )
        kl_loss = tf.reduce_mean(kl_divergence)
        return kl_loss

    def compute_representation_loss(self):
        """
        representation loss to make all the synthetic code z_m_1 after encoder approximate the z_m_0, i.e.,
        force the encoder to filter all the low-level information
        """
        code_loss = tf.reduce_mean(tf.square(self.z_m_0 - self.z_m_1))
        return code_loss
