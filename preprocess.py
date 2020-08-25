import os
import cv2 as cv


def count(x):
    res = 0
    while x > 0:
        x = x // 10
        res += 1
    return res


def video2frames(video_path, frames_path, sample_ratio=16):
    """
    Convert the videos into frames according to their ids, and save the correspondent frames.
    """
    video_class_list = os.listdir(video_path)
    for i, video_class in enumerate(video_class_list):
        print('video class {}: {}'.format(i+1, video_class))
        video_class_path = os.path.join(video_path, video_class)
        video_name_list = os.listdir(video_class_path)
        for video_name in video_name_list:
            video_name_path = os.path.join(video_class_path, video_name)
            capture = cv.VideoCapture(video_name_path)
            # fps = capture.get(cv.CAP_PROP_FPS)
            num_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)
            # print('video name:', video_name, '\n', 'fps:', fps, '; number of frames:', num_frames)

            _count = 0
            for i in range(int(num_frames)):
                if _count == sample_ratio:
                    break

                assert num_frames >= sample_ratio

                if i % (num_frames // sample_ratio) == 0:
                    _, frame = capture.read()
                    image_name = '0'*(5-count(i+1)) + '{}'.format(i+1) + '.jpg'
                    save_class_path = os.path.join(frames_path, video_class)
                    if not os.path.exists(save_class_path):
                        os.mkdir(save_class_path)
                    save_name_path = os.path.join(save_class_path, video_name)
                    if not os.path.exists(save_name_path):
                        os.mkdir(save_name_path)
                    cv.imwrite(os.path.join(save_name_path, image_name), frame)
                    _count += 1

            assert len(os.listdir(save_name_path)) == sample_ratio


if __name__ == '__main__':
    video_path = '/Users/dengandong/Desktop/uf/'
    ds_store = '.DS_Store'
    if os.path.exists(os.path.join(video_path, ds_store)):
        os.remove(os.path.join(video_path, ds_store))
    frames_path = '/Users/dengandong/Desktop/101frames_16/'
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)

    video2frames(video_path, frames_path)
