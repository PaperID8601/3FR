import os

import numpy as np
import pandas as pd

from torchvision import transforms as tf
from utils.face_processing import RandomLowQuality, RandomHorizontalFlip
import config as cfg
from utils.io import makedirs
import cv2
from skimage import io
from utils import face_processing as fp, face_processing

# To avoid exceptions when loading truncated image files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def denormalize(tensor):
    # assert(len(tensor.shape[1] == 3)
    if tensor.shape[1] == 3:
        tensor[:, 0] += 0.518
        tensor[:, 1] += 0.418
        tensor[:, 2] += 0.361
    elif tensor.shape[-1] == 3:
        tensor[..., 0] += 0.518
        tensor[..., 1] += 0.418
        tensor[..., 2] += 0.361

def denormalized(tensor):
    # assert(len(tensor.shape[1] == 3)
    if isinstance(tensor, np.ndarray):
        t = tensor.copy()
    else:
        t = tensor.clone()
    denormalize(t)
    return t


def read_openface_detection(lmFilepath, numpy_lmFilepath=None, from_sequence=False, use_cache=True,
                            return_num_faces=False, expected_face_center=None):
    num_faces_in_image = 0
    try:
        if numpy_lmFilepath is not None:
            npfile = numpy_lmFilepath + '.npz'
        else:
            npfile = lmFilepath + '.npz'
        if os.path.isfile(npfile) and use_cache:
            try:
                data = np.load(npfile)
                of_conf, landmarks, pose = [data[arr] for arr in data.files]
                if of_conf > 0:
                    num_faces_in_image = 1
            except:
                print('Could not open file {}'.format(npfile))
                raise
        else:
            if from_sequence:
                lmFilepath = lmFilepath.replace('features', 'features_sequence')
                lmDir, fname = os.path.split(lmFilepath)
                clip_name = os.path.split(lmDir)[1]
                lmFilepath = os.path.join(lmDir, clip_name)
                features = pd.read_csv(lmFilepath + '.csv', skipinitialspace=True)
                frame_num = int(os.path.splitext(fname)[0])
                features = features[features.frame == frame_num]
            else:
                features = pd.read_csv(lmFilepath + '.csv', skipinitialspace=True)
            features.sort_values('confidence', ascending=False, inplace=True)
            selected_face_id = 0
            num_faces_in_image = len(features)
            if num_faces_in_image > 1 and expected_face_center is not None:
                max_face_size = 0
                min_distance = 1000
                for fid in range(len(features)):
                    face = features.iloc[fid]
                    # if face.confidence < 0.2:
                    #     continue
                    landmarks_x = face.as_matrix(columns=['x_{}'.format(i) for i in range(68)])
                    landmarks_y = face.as_matrix(columns=['y_{}'.format(i) for i in range(68)])

                    landmarks = np.vstack((landmarks_x, landmarks_y)).T
                    face_center = landmarks.mean(axis=0)
                    distance = ((face_center - expected_face_center)**2).sum()**0.5
                    if distance < min_distance:
                        min_distance = distance
                        selected_face_id = fid

            try:
                face = features.iloc[selected_face_id]
            except KeyError:
                face = features
            of_conf = face.confidence
            landmarks_x = face.as_matrix(columns=['x_{}'.format(i) for i in range(68)])
            landmarks_y = face.as_matrix(columns=['y_{}'.format(i) for i in range(68)])
            landmarks = np.vstack((landmarks_x, landmarks_y)).T
            pitch = face.pose_Rx
            yaw = face.pose_Ry
            roll = face.pose_Rz
            pose = np.array((pitch, yaw, roll), dtype=np.float32)
            if numpy_lmFilepath is not None:
                makedirs(npfile)
            np.savez(npfile, of_conf, landmarks, pose)
    except IOError as e:
        # raise IOError("\tError: Could not load landmarks from file {}!".format(lmFilepath))
        # pass
        # print(e)
        of_conf = 0
        landmarks = np.zeros((68,2), dtype=np.float32)
        pose = np.zeros(3, dtype=np.float32)

    result = [of_conf, landmarks.astype(np.float32), pose]
    if return_num_faces:
        result += [num_faces_in_image]
    return result


def read_300W_detection(lmFilepath):
    lms = []
    with open(lmFilepath) as f:
        for line in f:
            try:
                x,y = [float(e) for e in line.split()]
                lms.append((x, y))
            except:
                pass
    assert(len(lms) == 68)
    landmarks = np.vstack(lms)
    return landmarks


def build_transform(deterministic, color, daug=0):
    transforms = []
    if not deterministic:
        transforms = [fp.RandomHorizontalFlip(0.5)]
        if daug == 1:
            transforms += [fp.RandomAffine(3, translate=[0.025,0.025], scale=[0.975, 1.025], shear=0, keep_aspect=False)]
        elif daug == 2:
            transforms += [fp.RandomAffine(3, translate=[0.035,0.035], scale=[0.970, 1.030], shear=2, keep_aspect=False)]
        elif daug == 3:
            transforms += [fp.RandomAffine(20, translate=[0.035,0.035], scale=[0.970, 1.030], shear=5, keep_aspect=False)]
        elif daug == 4:
            transforms += [fp.RandomAffine(45, translate=[0.035,0.035], scale=[0.940, 1.030], shear=5, keep_aspect=False)]
        elif daug == 5:
            transforms += [fp.RandomAffine(60, translate=[0.035,0.035], scale=[0.940, 1.030], shear=5, keep_aspect=False)]
        elif daug == 6:
            transforms += [fp.RandomAffine(30, translate=[0.04,0.04], scale=[0.940, 1.050], shear=5, keep_aspect=False)]
    return tf.Compose(transforms)


def get_face(filename, fullsize_img_dir, cropped_img_dir, landmarks, pose=None, bb=None, size=(cfg.CROP_SIZE, cfg.CROP_SIZE),
             use_cache=True, cropper=None):
    filename_noext = os.path.splitext(filename)[0]
    crop_filepath = os.path.join(cropped_img_dir, filename_noext + '.jpg')
    is_cached_crop = False
    if use_cache and os.path.isfile(crop_filepath):
        try:
            img = io.imread(crop_filepath)
        except:
            raise IOError("\tError: Could not cropped image {}!".format(crop_filepath))
        if img.shape[:2] != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        is_cached_crop = True
    else:
        # Load image from dataset
        img_path = os.path.join(fullsize_img_dir, filename)
        try:
            img = io.imread(img_path)
        except:
            raise IOError("\tError: Could not load image {}!".format(img_path))
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        assert(img.shape[2] == 3)

    if (landmarks is None or not landmarks.any()) and not 'crops_celeba' in cropped_img_dir:
            assert(bb is not None)
            # Fall back to bounding box if no landmarks found
            # print('falling back to bounding box')
            crop = face_processing.crop_by_bb(img, face_processing.scale_bb(bb, f=1.075), size=size)
    else:

        if 'crops_celeba' in cropped_img_dir:
            if is_cached_crop:
                crop = img
            else:
                crop = face_processing.crop_celeba(img, size)
        else:
                crop, landmarks, pose = face_processing.crop_face(img,
                                                                  landmarks,
                                                                  img_already_cropped=is_cached_crop,
                                                                  pose=pose,
                                                                  output_size=size,
                                                                  crop_by_eye_mouth_dist=cfg.CROP_BY_EYE_MOUTH_DIST,
                                                                  align_face_orientation=cfg.CROP_ALIGN_ROTATION,
                                                                  crop_square=cfg.CROP_SQUARE)
    if use_cache and not is_cached_crop:
        makedirs(crop_filepath)
        io.imsave(crop_filepath, crop)

    return crop, landmarks, pose