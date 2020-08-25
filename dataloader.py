import os
import time
import math
import argparse
import tensorflow as tf
import cv2 as cv
import numpy as np


class UCF101:
    def __init__(self, args, mode):

        self.frames_saved_path = args.frames_saved_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.mode = mode
        self.h = 128
        self.w = 128

        if mode == 'train':
            self.video_name_path = 'data/ucfTrainTestlist/trainlist{}.txt'.format(args.split_type)
            self.num_epochs = args.num_epochs
            # self.split_ratio = 0.9
        elif mode == 'test':
            self.video_name_path = 'data/ucfTrainTestlist/test_list{}.txt'.format(args.split_type)
            self.num_epochs = 1

        self.video_name_list = open(self.video_name_path, 'r').readlines()

    def __len__(self):
        return len(self.video_name_list)

    @staticmethod
    def get_sample_shape():
        return {
            'frames': [16, 240, 320, 3],
            'label': []
        }

    @staticmethod
    def get_sample_dtype():
        return {
            'frames': tf.float32,
            'label': tf.int32
        }

    def data_generator(self):
        """

        :return:
        frames_path_list: contains 16 frames' names for a single video
        label: corresponding video labels
        """
        cwd = os.getcwd()
        data_dir = 'data/ucfTrainTestlist'
        class_txt_path = os.path.join(cwd, data_dir, 'classInd.txt')
        label_txt_list = open(class_txt_path, 'r').readlines()
        label_dict = dict()
        for label_name in label_txt_list:
            line = label_name.split()  # class_index class_name
            label_dict[line[1]] = int(line[0])  # {'class_name': class_index}

        video_name_list = self.video_name_list
        frames_path_list = []
        video_label_list = []
        for video_name in video_name_list:
            name = video_name.split()  # video_class/video_name class_index
            video_path_name = name[0]  # video_class/video_name
            video_class_name = video_path_name.split('/')[0]  # video_class

            frames_list = []  # to save frames' absolute path (from 101frames_16) for a single video
            for frame_name in os.listdir(os.path.join(self.frames_saved_path, video_path_name)):
                frames_list.append(os.path.join(self.frames_saved_path, video_path_name, frame_name))
            frames_list.sort()
            frames_path_list.append(
                frames_list)  # the element in image_path_list is a list which contains 16 frames' name
            video_label_list.append(video_class_name)

        numeral_label_list = []
        for class_name in video_label_list:
            numeral_label_list.append(label_dict[class_name])

        index_list = list(range(len(frames_path_list)))
        np.random.shuffle(index_list)
        for index in index_list:
            image_3d = self.image_name_list_to_3d(frames_path_list[index])
            yield {'frames': image_3d, 'label': numeral_label_list[index]}

    def get_dataset(self, mode='1'):

        # frames_path_list_cast = tf.cast(self.frames_path_list, tf.string)  # (num, 16), string
        # label_cast = tf.cast(self.label, tf.int32)
        # dataset = tf.data.Dataset.from_tensor_slices((frames_path_list_cast, label_cast))
        sample_dtype = self.get_sample_dtype()
        sample_shape = self.get_sample_shape()
        dataset = tf.data.Dataset.from_generator(self.data_generator, sample_dtype, sample_shape)

        if mode == '1':
            dataset = dataset.map(self.preprocess_fn)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=self.batch_size)
            dataset = dataset.repeat(self.num_epochs)
        elif mode == '2':
            dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=self.preprocess_fn,
                                                                  batch_size=self.batch_size,
                                                                  num_parallel_calls=8))
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(100, self.num_epochs))
            dataset = dataset.prefetch(buffer_size=self.batch_size)

        return dataset

    def preprocess_fn(self, inputs):
        frames = inputs['frames']
        label = inputs['label']
        # images = tf.image.decode_jpeg(image_3d, channels=3)
        # images = tf.image.resize_image_with_crop_or_pad(image_3d, 224, 224)
        # images = tf.image.random_flip_left_right(images)
        # images = tf.image.random_contrast(images, lower=0.2, upper=1.8)
        images = self.random_crop(frames)
        # images = tf.image.per_image_standardization(images)
        images = images / 255.0
        return {'frames': images, 'label': label}
        # return images, label

    @staticmethod
    def image_name_list_to_3d(img_name_list):
        img_list = []
        for image_name in img_name_list:
            img = cv.imread(image_name)
            img = np.array(img)
            img_list.append(img)
        img_3d = np.stack(img_list, axis=0)  # [16, 320, 240, 3]
        return img_3d

    @staticmethod
    def resize(img, h, w):
        _, img_h, img_w, _ = np.shape(img)
        if img_h < h or img_w < w:
            ref = min(img_h, img_w)
            ratio = h / ref
            img = cv.resize(img, [math.ceil(img_h * ratio), math.ceil(img_w * ratio)])
        return img

    def center_crop(self, img):
        _, img_h, img_w, _ = np.shape(img)
        h, w = self.h, self.w
        img = self.resize(img, h, w)
        crop_img = img[img_h // 2 - h // 2:img_h // 2 - h // 2 + h, img_w // 2 - w // 2:img_w // 2 - w // 2 + w, :]
        return crop_img

    def random_crop(self, img, h=128, w=128):
        print(np.shape(img))
        _, img_h, img_w, _ = np.shape(img)
        start_h = np.random.randint(low=0, high=img_h - h)
        start_w = np.random.randint(low=0, high=img_w - w)
        crop_img = img[:, start_h: start_h + h, start_w: start_w + w, :]
        return crop_img

    @staticmethod
    def standardization(img_3d):
        mean = np.mean(img_3d)
        num_pix = np.prod(np.shape(img_3d))
        std_img = (img_3d - mean) / np.sqrt(num_pix)
        return std_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_saved_path', type=str, default='data/101frames_16')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--split_type', type=str, default='01')
    args = parser.parse_args()

    ucf = UCF101(args, 'train')
    dataset = ucf.get_dataset(mode='2')
    print(len(ucf))

    with tf.Session() as sess:

        handle_pl = tf.placeholder(tf.string, [])
        base_iterator = tf.data.Iterator.from_string_handle(
            handle_pl, dataset.output_types, dataset.output_shapes)
        inputs = base_iterator.get_next()

        iters = dataset.make_initializable_iterator()
        train_handle = sess.run(iters.string_handle())

        sess.run(iters.initializer)
        while True:
            start = time.time()
            x_o, y_o = sess.run([inputs], feed_dict={handle_pl: train_handle})
            print(x_o)
            print(time.time()-start)
