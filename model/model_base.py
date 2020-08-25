import os
import time

from abc import ABC, abstractmethod
import tensorflow as tf


class ModelBase(ABC):
    name = 'base model'

    def __init__(self,
                 args,
                 discriminator,
                 is_training=True):

        self.D = discriminator

        self.is_training = is_training

        self.true = None  # input
        self.z_m = None  # motion code
        self.z_mu = None
        self.z_sigma = None
        self.z_c = None  # content code
        self.z = None  # complete code
        self.gap = None  # reduced frames by operating gap on input
        self.reconstructed_gap = None  # reconstructed reduced frames
        self.reconstructed_frames = None  # reconstructed input
        self.synthesized_frames = None
        self.z_m_1 = None  # re-obtained motion code from synthesized frames

        self.train_op = None

        self.loss = None
        self.recon_loss = None
        self.ad_loss = None
        self.reg_loss = None
        self.code_loss = None

        self.lr = args.learning_rate
        self.save_dir = 'model_dir/'
        pass

    def build_model(self, input_frames):
        self.true = input_frames['frames']
        self.forward(self.true)

        if self.is_training:
            self.compute_loss()
        pass

    @abstractmethod
    def forward(self, input_frames):
        raise NotImplementedError

    def train(self, sess, feed_dict, epoch):
        start_time = time.clock()
        ops = [self.train_op, self.loss, self.recon_loss, self.ad_loss, self.reg_loss, self.code_loss]
        _, total_loss, recon_loss, ad_loss, reg_loss, code_loss = sess.run(ops, feed_dict=feed_dict)
        duration = time.clock() - start_time
        print('epoch {}:, total loss = {:.4f}, time cost {:.2f}s'.format(epoch, total_loss, duration))
        print('reconstruction loss = {:.4f}\n '
              'adversarial loss = {:.4f}\n '
              'KL loss = {:.4f}\n '
              'code loss = {:.4f}\n'.format(recon_loss, ad_loss, reg_loss, code_loss))
        pass

    def test(self, sess, feed_dict):
        start_time = time.clock()
        ops = [self.loss, self.recon_loss, self.ad_loss, self.reg_loss, self.code_loss]
        total_loss, recon_loss, ad_loss, reg_loss, code_loss = sess.run(ops, feed_dict=feed_dict)
        duration = time.clock() - start_time
        print('Test: total loss = {:.4f}, time cost {:.2f}s'.format(total_loss, duration))
        print('reconstruction loss = {:.4f}\n '
              'adversarial loss = {:.4f}\n '
              'KL loss = {:.4f}\n '
              'code loss = {:.4f}\n'.format(recon_loss, ad_loss, reg_loss, code_loss))
        pass

    def compute_loss(self, coefficient=None):
        if coefficient is None:
            coefficient = [0.1, 0.1, 1, 0.5]  # vital!
        self.recon_loss = self.compute_reconstruction_loss()
        self.ad_loss = self.compute_adversarial_loss(self.true, self.synthesized_frames)
        self.reg_loss = self.compute_regularization_loss()
        self.code_loss = self.compute_representation_loss()
        self.loss = coefficient[0] * self.ad_loss + \
                    coefficient[1] * self.reg_loss + \
                    coefficient[2] * self.code_loss + \
                    coefficient[3] * self.recon_loss
        # define the operation before the variables initialization
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        pass

    def compute_reconstruction_loss(self):
        """
        reconstruction loss to make the latent code z capture all the necessary information of the input frames
        """
        reconstruction_loss = tf.reduce_mean(tf.square(self.true - self.reconstructed_frames))
        gap_reconstruction_loss = tf.reduce_mean(tf.square(self.gap - self.reconstructed_gap))
        loss = reconstruction_loss + gap_reconstruction_loss
        return loss

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
            tf.square(self.z_mu) + tf.square(self.z_sigma) - tf.log(1e-8 + tf.square(self.z_sigma)) - 1, axis=1
        )
        kl_loss = tf.reduce_mean(kl_divergence)
        return kl_loss

    def compute_representation_loss(self, mode='l2'):
        """
        representation loss to make all the synthetic code z_m_1 after encoder approximate the z_m_0, i.e.,
        force the encoder to filter all the low-level information
        """
        if mode == 'l2':
            return tf.reduce_mean(tf.square(self.z_m - self.z_m_1))
        elif mode == 'le':
            return tf.reduce_mean(tf.abs(self.z_m - self.z_m_1))

    def save_model(self, sess, step):
        name = self.__class__.name
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        ckpt_path = os.path.join(self.save_dir, name, 'model.ckpt')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        saver.save(sess, ckpt_path, global_step=step)
        pass
