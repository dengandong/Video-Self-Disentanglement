import argparse
import tensorflow as tf

from dataloader import UCF101
from model.autoencoder.ae_3dcnn import MotionEncoderC3D, ContentEncoder, Decoder
from model.model_3dcnn import Model3D
# from model.encoders import MotionEncoder, ContentEncoder
# from model.decoder import Decoder
from model.discriminator import Discriminator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_saved_path', type=str, default='data/101frames_16')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=str, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_reduce_mode', type=str, default='step')
    parser.add_argument('--split_type', type=str, default='01')

    parser.add_argument('--save_model_interval', type=int, default=1000)
    parser.add_argument('--test_interval', type=int, default=1000)
    parser.add_argument('--save_test', type=int, default=1000)

    args = parser.parse_args()

    # model definition
    motion_encoder = MotionEncoderC3D()
    content_encoder = ContentEncoder(batch_size=args.batch_size)
    G = Decoder(batch_size=args.batch_size)
    D = Discriminator()
    model = Model3D(args, motion_encoder, content_encoder, G, D)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    # open tf session
    with tf.Session(config=session_config) as sess:
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # initializing dataset
        train_data = UCF101(args, 'train')
        test_data = UCF101(args, 'train')
        train_dataset = train_data.get_dataset(mode='2')
        test_dataset = test_data.get_dataset(mode='2')

        # setup inputs
        handle_pl = tf.placeholder(tf.string, [])
        base_iterator = tf.data.Iterator.from_string_handle(
            handle_pl, train_dataset.output_types, train_dataset.output_shapes)
        inputs = base_iterator.get_next()

        # data iterator initialization
        train_iterator = train_dataset.make_initializable_iterator()
        test_iterator = test_dataset.make_initializable_iterator()
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        # model initialization
        model.build_model(inputs)

        # initializing variables
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        sess.run([global_init, local_init])

        start_step = sess.run(global_step)
        sess.run(train_iterator.initializer)

        # TODO
        for epoch in range(int(start_step), args.num_epochs):
            should_save_model = epoch % args.save_model_interval == 0
            should_test = epoch % args.test_interval == 0

            feed_dict = {handle_pl: train_handle}
            model.train(sess, feed_dict, epoch)

            if should_save_model:
                model.save_model(sess, epoch)
            if should_test:
                feed_dict = {handle_pl: test_handle}
                sess.run(test_iterator.initializer)
                model.test(sess, feed_dict)


if __name__ == '__main__':
    main()
