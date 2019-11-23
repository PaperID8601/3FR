import os
import time

import skimage.transform
import cv2
from scipy.ndimage import median_filter
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import numbers

from utils import geometry
from utils import log
from skimage import exposure
import config as cfg
# import torch.nn.functional as F
import torchvision.transforms.functional as F


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def mirror_padding(img):
    EXTEND_BLACK = True
    if EXTEND_BLACK:
        empty = np.zeros_like(img)
        center_row = np.hstack((empty, img, empty))
        flipped_ud = np.flipud(np.hstack((empty, empty, empty)))
        result = np.vstack((flipped_ud, center_row, flipped_ud))
    else:
        s = int(img.shape[0]*0.1)
        k = (s,s)
        # blurred = cv2.blur(cv2.blur(cv2.blur(img, k), k), k)
        blurred = cv2.blur(img, k)
        flipped_lr = np.fliplr(blurred)
        center_row = np.hstack((flipped_lr, img, flipped_lr))
        flipped_ud = np.flipud(np.hstack((flipped_lr, blurred, flipped_lr)))
        result = np.vstack((flipped_ud, center_row, flipped_ud))
    return result


def make_square(bb):
    h = bb[3] - bb[1]
    bb1 = geometry.convertBB2to1(bb)
    bb1[2] = h/2
    return geometry.convertBB1to2(bb1)


class FaceCrop():
    def __init__(self, img, output_size=(cfg.CROP_SIZE, cfg.CROP_SIZE), bbox=None, landmarks=None,
                 img_already_cropped=False, crop_by_eye_mouth_dist=False, align_face_orientation=True, scale=1.0,
                 crop_square=False):
        assert(bbox is not None or landmarks is not None)
        self.output_size = output_size
        self.align_face_orientation = align_face_orientation
        self.M = None
        self.img_already_cropped = img_already_cropped
        self.tl = None
        self.br = None
        self.scale = scale
        if bbox is not None:
            bbox = np.asarray(bbox)
            self.tl = bbox[:2].astype(int)
            self.br = bbox[2:4].astype(int)
        self.lms = landmarks
        self.img = img
        self.angle_x_deg = 0
        if landmarks is not None:
            self.calculate_crop_parameters(img, landmarks, img_already_cropped)

    def __get_eye_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_eye_l, id_eye_r = [36, 39],  [42, 45]
        elif lms.shape[0] == 98:
            id_eye_l, id_eye_r = [60, 64],  [68, 72]
        elif lms.shape[0] == 37:
            id_eye_l, id_eye_r = [13, 16],  [19, 22]
        elif lms.shape[0] == 21:
            id_eye_l, id_eye_r = [6, 8],  [9, 11]
        elif lms.shape[0] == 4:
            id_eye_l, id_eye_r = [0],  [1]
        elif lms.shape[0] == 5:
            # id_eye_l, id_eye_r = [1],  [2]
            id_eye_l, id_eye_r = [0],  [1]
        else:
            raise ValueError("Invalid landmark format!")
        eye_l, eye_r = lms[id_eye_l].mean(axis=0),  lms[id_eye_r].mean(axis=0)  # eye centers
        if eye_r[0] < eye_l[0]:
            eye_r, eye_l = eye_l, eye_r
        return eye_l, eye_r

    def __get_mouth_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_mouth_l, id_mouth_r = 48,  54
        elif lms.shape[0] == 98:
            id_mouth_l, id_mouth_r = 76,  82
        elif lms.shape[0] == 37:
            id_mouth_l, id_mouth_r = 25,  31
        elif lms.shape[0] == 21:
            id_mouth_l, id_mouth_r = 17,  19
        elif lms.shape[0] == 4:
            id_mouth_l, id_mouth_r = 2,  3
        elif lms.shape[0] == 5:
            id_mouth_l, id_mouth_r = 3, 4
        else:
            raise ValueError("Invalid landmark format!")
        return lms[id_mouth_l],  lms[id_mouth_r]  # outer landmarks

    def __get_chin_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_chin = 8
        elif lms.shape[0] == 98:
            id_chin = 16
        else:
            raise ValueError("Invalid landmark format!")
        return lms[id_chin]

    def get_face_center(self, lms, return_scale=False):
        eye_l, eye_r = self.__get_eye_coordinates(lms)
        mouth_l, mouth_r = self.__get_mouth_coordinates(lms)
        chin = self.__get_chin_coordinates(lms)
        eye_c = (eye_l+eye_r)/2
        # vec_nose = eye_c - (mouth_l+mouth_r)/2
        vec_nose = eye_c - chin
        c = eye_c - 0.25*vec_nose
        if return_scale:
            s = int(0.9*np.linalg.norm(vec_nose))
            return c, s
        else:
            return c

    def __calc_rotation_matrix(self, center, eye_l, eye_r, nose_upper, chin):
        def calc_angles_deg(vec):
            vnorm = vec / np.linalg.norm(vec)
            angle_x = np.arcsin(vnorm[1])
            angle_y = np.arcsin(vnorm[0])
            return np.rad2deg(angle_x), np.rad2deg(angle_y)

        vec_eye = eye_r - eye_l
        w = np.linalg.norm(vec_eye)
        vx = vec_eye / w

        if w > 100:
            angle_x = np.arcsin(vx[1])
            self.angle_x_deg = np.rad2deg(angle_x)
        else:
            # profile faces
            vec_nose = nose_upper - chin
            ax, ay = calc_angles_deg(vec_nose)
            self.angle_x_deg = ay
        return cv2.getRotationMatrix2D(tuple(center), self.angle_x_deg, 1.0)

    def __rotate_image(self, img, M):
        return cv2.warpAffine(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)

    def __rotate_landmarks(self, lms, M):
        _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
        return M.dot(_lms_hom.T).T  # apply transformation


    def calculate_crop_parameters(self, img, lms_orig, img_already_cropped):
        self.__img_shape = img.shape
        lms = lms_orig.copy()

        self.face_center, self.face_scale = self.get_face_center(lms, return_scale=True)

        nonzeros= lms[:,0] > 0

        if self.align_face_orientation:
            eye_l, eye_r = self.__get_eye_coordinates(lms)
            chin = self.__get_chin_coordinates(lms)
            nose_upper = lms[51]
            self.M = self.__calc_rotation_matrix(self.face_center, eye_l, eye_r, nose_upper, chin)
            lms = self.__rotate_landmarks(lms, self.M)

        # EYES_ALWAYS_CENTERED = cfg.INPUT_SIZE == 128
        EYES_ALWAYS_CENTERED = False
        if not EYES_ALWAYS_CENTERED:
            cx = (lms[nonzeros,0].min()+lms[nonzeros,0].max())/2
            self.face_center[0] = cx

        crop_by_eye_mouth_dist = False
        if len(lms) == 5:
            crop_by_eye_mouth_dist = True

        crop_by_outline = False

        if crop_by_eye_mouth_dist:
            print(self.face_center)
            print(self.face_scale)
            self.tl = (self.face_center - self.face_scale).astype(int)
            self.br = (self.face_center + self.face_scale).astype(int)
            # pass
        elif crop_by_outline:
            t = lms[nonzeros, 1].min()
            b = lms[nonzeros, 1].max()
            if t > b:
                t, b = b, t
            h = b - t
            assert(h >= 0)

            l = lms[nonzeros, 0].min()
            r = lms[nonzeros, 0].max()
            if l > r:
                l, r = r, l
            w = r - l
            assert(w >= 0)

            s = max(w, h)
            s *= 1.4

            # w = s
            # h = s

            w *= 1.45
            h *= 1.45

            w *= self.scale
            h *= self.scale

            self.face_center = [(r + l) // 2, (b + t) // 2]
            face_center = self.face_center

            face_center[1] += int(h*0.02)

            # h *= 1.45
            # face_center[1] -= int(h*0.025)

            # self.tl = [face_center[0] - s, face_center[1] - s]
            # self.br = [face_center[0] + s, face_center[1] + s]
            self.tl = [face_center[0] - w//2, face_center[1] - h//2]
            self.br = [face_center[0] + w//2, face_center[1] + h//2]

        else:
            # calc height
            t = lms[nonzeros, 1].min()
            b = lms[nonzeros, 1].max()
            if t > b:
                t, b = b, t

            h = b - t
            assert(h >= 0)

            # calc width
            l = lms[nonzeros, 0].min()
            r = lms[nonzeros, 0].max()
            if l > r:
                l, r = r, l
            w = r - l
            assert(w >= 0)

            if len(lms) != 68 and len(lms) != 21 and len(lms) != 98:
                h *= 1.5
                t = t - h/2
                b = b + h/2

            # enlarge a little
            cfg.CROP_MOVE_TOP_FACTOR = 0.25       # move top by factor of face height in respect to eye brows
            cfg.CROP_MOVE_BOTTOM_FACTOR = 0.15   # move bottom by factor of face height in respect to chin bottom point
            min_row, max_row = int(t - cfg.CROP_MOVE_TOP_FACTOR * h), int(b + cfg.CROP_MOVE_BOTTOM_FACTOR * h)

            # calc width
            crop_square = False
            if crop_square:
                s = (max_row - min_row)/2
                min_col, max_col = self.face_center[0] - s, self.face_center[0] + s
            else:
                # min_col, max_col = l, r
                min_col, max_col = int(l - 0.15 * w), int(r + 0.15 * w)

            # in case right eye is actually left of right eye...
            if min_col > max_col:
                min_col, max_col = min_col, max_col

            # crop = img[int(min_row):int(max_row), int(min_col):int(max_col)]
            # plt.imshow(crop)
            # plt.show()

            self.tl = np.array((min_col, min_row))
            self.br = np.array((max_col, max_row))

        # extend area by crop border margins
        scale_factor = cfg.CROP_SIZE / cfg.INPUT_SIZE
        bbox = np.concatenate((self.tl, self.br))
        bbox_crop = geometry.scaleBB(bbox, scale_factor, scale_factor, typeBB=2)
        self.tl = bbox_crop[0:2].astype(int)
        self.br = bbox_crop[2:4].astype(int)


    def apply_to_image(self, img=None, with_hist_norm=cfg.WITH_HIST_NORM):
        if img is None:
            img = self.img

        if self.img_already_cropped:
            h, w = img.shape[:2]
            if (w,h) != self.output_size:
                img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_CUBIC)
            return img

        img_padded = mirror_padding(img)

        h,w = img.shape[:2]
        tl_padded = self.tl + (w,h)
        br_padded = self.br + (w,h)

        # extend image in case mirror padded image is still too smal
        dilate = -np.minimum(tl_padded, 0)
        padding = [
            (dilate[1], dilate[1]),
            (dilate[0], dilate[0]),
             (0,0)
        ]
        try:
            img_padded = np.pad(img_padded, padding, 'constant')
        except TypeError:
            plt.imshow(img)
            plt.show()
        tl_padded += dilate
        br_padded += dilate

        if self.align_face_orientation and self.lms is not None:
            # rotate image
            face_center = self.get_face_center(self.lms, return_scale=False)
            M  = cv2.getRotationMatrix2D(tuple(face_center+(w,h)), self.angle_x_deg, 1.0)
            img_padded = self.__rotate_image(img_padded, M)

        crop = img_padded[tl_padded[1]: br_padded[1], tl_padded[0]: br_padded[0]]

        try:
            resized_crop = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_CUBIC)
        except cv2.error:
            print('img size', img.shape)
            print(self.tl)
            print(self.br)
            print('dilate: ', dilate)
            print('padding: ', padding)
            print('img pad size', img_padded.shape)
            print(tl_padded)
            print(br_padded)
            plt.imshow(img_padded)
            plt.show()
            raise

        # image normalization
        if with_hist_norm:
            p2, p98 = np.percentile(crop, (2, 98))
            resized_crop = exposure.rescale_intensity(resized_crop, in_range=(p2, p98))

        # resized_crop = resized_crop.astype(np.float32)
        np.clip(resized_crop, 0, 255)
        return resized_crop


    def center_on_face(self, img=None, landmarks=None):
        raise(NotImplemented)
        if img is None:
            img = self.img

        tl_padded = np.array(img.shape[:2][::-1])
        br_padded = tl_padded * 2
        img_center = tl_padded * 1.5

        face_center = self.get_face_center(landmarks)
        offset = (self.face_center - img_center).astype(int)

        img_new = img
        landmarks_new = landmarks

        if landmarks is not None:
            if self.align_face_orientation and self.lms is not None:
                landmarks_new = self.__rotate_landmarks(landmarks_new+tl_padded, self.M) - tl_padded
            landmarks_new = landmarks_new - offset

        if img is not None:
            img_padded = mirror_padding(img)
            if self.align_face_orientation and self.lms is not None:
                img_padded = self.__rotate_image(img_padded, self.M)
            tl_padded += offset
            br_padded += offset
            img_new = img_padded[tl_padded[1]: br_padded[1], tl_padded[0]: br_padded[0]]

        return img_new, landmarks_new


    def apply_to_landmarks(self, lms_orig, pose=None):
        if lms_orig is None:
            return lms_orig, pose

        if pose is not None:
            pose_new = np.array(pose).copy()
        else:
            pose_new = None

        lms = lms_orig.copy()

        if self.align_face_orientation and self.lms is not None:
            # rotate landmarks
            # if not self.img_already_cropped:
            #     lms[:,0] += self.img.shape[1]
            #     lms[:,1] += self.img.shape[0]
            self.face_center_orig = self.get_face_center(self.lms, return_scale=False)
            M = cv2.getRotationMatrix2D(tuple(self.face_center_orig), self.angle_x_deg, 1.0)
            lms = self.__rotate_landmarks(lms, M).astype(np.float32)

            # if not self.img_already_cropped:
            #     lms[:,0] -= self.img.shape[1]
            #     lms[:,1] -= self.img.shape[0]

            if pose_new is not None:
                pose_new[2] = 0.0

        # tl = (self.face_center - self.face_scale).astype(int)
        # br = (self.face_center + self.face_scale).astype(int)

        tl = self.tl
        br = self.br
        crop_width = br[0] - tl[0]
        crop_height = br[1] - tl[1]

        lms_new = lms.copy()
        lms_new[:, 0] = (lms_new[:, 0] - tl[0]) * self.output_size[0] / crop_width
        lms_new[:, 1] = (lms_new[:, 1] - tl[1]) * self.output_size[1] / crop_height

        return lms_new, pose_new

    def apply_to_landmarks_inv(self, lms):
        tl = self.tl
        br = self.br
        crop_width = br[0] - tl[0]
        crop_height = br[1] - tl[1]

        lms_new = lms.copy()
        lms_new[:, 0] = lms_new[:, 0] * (crop_width / self.output_size[0]) + tl[0]
        lms_new[:, 1] = lms_new[:, 1] * (crop_height / self.output_size[1]) + tl[1]

        # return lms_new
        if self.M is not None:
            M = cv2.getRotationMatrix2D(tuple(self.face_center_orig), -self.angle_x_deg, 1.0)
            lms_new = self.__rotate_landmarks(lms_new, M)

        return lms_new


class CenterCrop(object):
    """Like tf.CenterCrop, but works works on numpy arrays instead of PIL images."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __crop_image(self, img):
        t = int((img.shape[0] - self.size[0]) / 2)
        l = int((img.shape[1] - self.size[1]) / 2)
        b = t + self.size[0]
        r = l + self.size[1]
        return img[t:b, l:r]

    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
            if landmarks is not None:
                landmarks[...,0] -= int((img.shape[0] - self.size[0]) / 2)
                landmarks[...,1] -= int((img.shape[1] - self.size[1]) / 2)
                landmarks[landmarks < 0] = 0
            return {'image': self.__crop_image(img), 'landmarks': landmarks, 'pose': pose}
        else:
            return self.__crop_image(sample)


    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class RandomRotation(object):
    """Rotate the image by angle.

    Like tf.RandomRotation, but works works on numpy arrays instead of PIL images.

    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle


    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
        angle = self.get_params(self.degrees)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = calc_rotation_matrix(center, angle)
        img_rotated = rotate_image(image, M)
        if landmarks is not None:
            landmarks = rotate_landmarks(landmarks, M).astype(np.float32)
            pose_rotated = pose
            pose_rotated[2] -= np.deg2rad(angle).astype(np.float32)
        return {'image': img_rotated, 'landmarks': landmarks, 'pose': pose}

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ')'
        return format_string


def calc_rotation_matrix(center, degrees):
    return cv2.getRotationMatrix2D(tuple(center), degrees, 1.0)


def rotate_image(img, M):
    return cv2.warpAffine(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


def rotate_landmarks(lms, M):
    _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
    return M.dot(_lms_hom.T).T  # apply transformation


def transform_image(img, M):
    # r = skimage.transform.AffineTransform(rotation=np.deg2rad(45))
    # im_centered = skimage.transform.warp(img, t._inv_matrix).astype(np.float32)
    # im_trans = skimage.transform.warp(im_centered, M._inv_matrix, order=3).astype(np.float32)
    # tmp = skimage.transform.AffineTransform(matrix=M)

    # return  skimage.transform.warp(img, M._inv_matrix, order=3).astype(np.float32)  # Very slow!
    return cv2.warpAffine(img, M.params[:2], img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


def transform_landmarks(lms, M):
    _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
    # t = skimage.transform.AffineTransform(translation=-np.array(img.shape[:2][::-1])/2)
    # m = t._inv_matrix.dot(M.params.dot(t.params))
    # return M.params.dot(_lms_hom.T).T[:,:2]
    return M(lms)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (numbers.Number, tuple))
        self.output_size = output_size

    # @staticmethod
    # def crop(image, scale):
    #     s = random.uniform(*scale)
    #     print(s)
    #     h, w = image.shape[:2]
    #     cy, cx = h//2, w//2
    #     h2, w2 = int(cy*s), int(cx*s)
    #     return image[cy-h2:cy+h2, cx-h2:cx+h2]

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        # if random.random() < self.p:
        #     image = self.crop(image, self.scale)

        h, w = image.shape[:2]

        if isinstance(self.output_size, numbers.Number):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # img = F.resize(image, (new_h, new_w))
        img = cv2.resize(image, dsize=(new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]
            landmarks = landmarks.astype(np.float32)

        return {'image': img, 'landmarks': landmarks, 'pose': pose}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        if landmarks is not None:
            landmarks = landmarks - [left, top]
            landmarks = landmarks.astype(np.float32)

        return {'image': image, 'landmarks': landmarks, 'pose': pose}


class RandomResizedCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, p=1.0, scale=(1.0, 1.0), keep_aspect=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.scale = scale
        self.p = p
        self.keep_aspect = keep_aspect

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        h, w = image.shape[:2]
        s_x = random.uniform(*self.scale)
        if self.keep_aspect:
            s_y = s_x
        else:
            s_y = random.uniform(*self.scale)
        new_w, new_h = int(self.output_size[0] * s_x), int(self.output_size[1] * s_y)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        image = cv2.resize(image, dsize=self.output_size)
        landmarks /= [s_x, s_y]

        return {'image': image, 'landmarks': landmarks.astype(np.float32), 'pose': pose.astype(np.float32)}


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    lm_left_to_right_98 = {
        # outline
        0:32,
        1:31,
        2:30,
        3:29,
        4:28,
        5:27,
        6:26,
        7:25,
        8:24,

        9:23,
        10:22,
        11:21,
        12:20,
        13:19,
        14:18,
        15:17,
        16:16,

        #eyebrows
        33:46,
        34:45,
        35:44,
        36:43,
        37:42,
        38:50,
        39:49,
        40:48,
        41:47,

        #nose
        51:51,
        52:52,
        53:53,
        54:54,

        55:59,
        56:58,
        57:57,

        #eyes
        60:72,
        61:71,
        62:70,
        63:69,
        64:68,
        65:75,
        66:74,
        67:73,
        96:97,

        #mouth outer
        76:82,
        77:81,
        78:80,
        79:79,
        87:83,
        86:84,
        85:85,

        #mouth inner
        88:92,
        89:91,
        90:90,
        95:93,
        94:94,
    }

    lm_left_to_right_68 = {
        # outline
        0:16,
        1:15,
        2:14,
        3:13,
        4:12,
        5:11,
        6:10,
        7:9,
        8:8,

        #eyebrows
        17:26,
        18:25,
        19:24,
        20:23,
        21:22,

        #nose
        27:27,
        28:28,
        29:29,
        30:30,

        31:35,
        32:34,
        33:33,

        #eyes
        36:45,
        37:44,
        38:43,
        39:42,
        40:47,
        41:46,

        #mouth outer
        48:54,
        49:53,
        50:52,
        51:51,
        57:57,
        58:56,
        59:55,

        #mouth inner
        60:64,
        61:63,
        62:62,
        66:66,
        67:65,
    }

    # AFLW
    lm_left_to_right_21 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:16,
        13:15,
        14:14,
        17:19,
        18:18,
        20:20
    }

    # AFLW without ears
    lm_left_to_right_19 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:14,
        13:13,
        15:17,
        16:16,
        18:18
    }

    lm_left_to_right_5 = {
        0:1,
        2:2,
        3:4,
    }

    def __init__(self, p=0.5):

        def build_landmark_flip_map(left_to_right):
            map = left_to_right
            right_to_left = {v:k for k,v in map.items()}
            map.update(right_to_left)
            return map

        self.p = p

        self.lm_flip_map_98 = build_landmark_flip_map(self.lm_left_to_right_98)
        self.lm_flip_map_68 = build_landmark_flip_map(self.lm_left_to_right_68)
        self.lm_flip_map_21 = build_landmark_flip_map(self.lm_left_to_right_21)
        self.lm_flip_map_19 = build_landmark_flip_map(self.lm_left_to_right_19)
        self.lm_flip_map_5 = build_landmark_flip_map(self.lm_left_to_right_5)


    def __call__(self, sample):
        if random.random() < self.p:
            if isinstance(sample, dict):
                img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
                # flip image
                flipped_img = np.fliplr(img).copy()
                # flip landmarks
                non_zeros = landmarks[:,0] > 0
                landmarks[non_zeros, 0] *= -1
                landmarks[non_zeros, 0] += img.shape[1]
                landmarks_new = landmarks.copy()
                if len(landmarks) == 21:
                    lm_flip_map = self.lm_flip_map_21
                if len(landmarks) == 19:
                    lm_flip_map = self.lm_flip_map_19
                elif len(landmarks) == 68:
                    lm_flip_map = self.lm_flip_map_68
                elif len(landmarks) == 5:
                    lm_flip_map = self.lm_flip_map_5
                elif len(landmarks) == 98:
                    lm_flip_map = self.lm_flip_map_98
                else:
                    raise ValueError('Invalid landmark format.')
                for i in range(len(landmarks)):
                    landmarks_new[i] = landmarks[lm_flip_map[i]]
                # flip pose
                if pose is not None:
                    pose[1] *= -1
                return {'image': flipped_img, 'landmarks': landmarks_new, 'pose': pose}

            return np.fliplr(sample).copy()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees=0, translate=None, scale=None, shear=None, resample=False, fillcolor=0, keep_aspect=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.keep_aspect = keep_aspect

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size, keep_aspect):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])

        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (-np.round(random.uniform(-max_dx, max_dx)),
                            -np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale_x = random.uniform(scale_ranges[0], scale_ranges[1])
            if keep_aspect:
                scale_y = scale_x
            else:
                scale_y = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale_x, scale_y = 1.0, 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        M =  skimage.transform.AffineTransform(
            translation=translations,
            shear=np.deg2rad(shear),
            scale=(scale_x, scale_y),
            rotation=np.deg2rad(angle)
        )
        t = skimage.transform.AffineTransform(translation=-np.array(img_size[::-1])/2)
        return skimage.transform.AffineTransform(matrix=t._inv_matrix.dot(M.params.dot(t.params)))


    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
        else:
            img = sample

        M = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.shape[:2], self.keep_aspect)
        img_new = transform_image(img, M)

        if isinstance(sample, dict):
            if landmarks is None:
                landmarks_new = None
            else:
                landmarks_new = transform_landmarks(landmarks, M).astype(np.float32)
            return {'image': img_new, 'landmarks': landmarks_new, 'pose': pose}
        else:
            return img_new


    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)



class RandomLowQuality(object):
    """Reduce image quality by as encoding as low quality jpg.

    Args:
        p (float): probability of the image being recoded. Default value is 0.2
        qmin (float): min jpg quality
        qmax (float): max jpg quality
    """

    def __init__(self, p=0.5, qmin=8, qmax=25):
        self.p = p
        self.qmin = qmin
        self.qmax = qmax

    def _encode(self, img, q):
        return cv2.imencode('.jpg', img, params=[int(cv2.IMWRITE_JPEG_QUALITY), q])

    def _recode(self, img, q):
        return cv2.imdecode(self._encode(img, q)[1], flags=cv2.IMREAD_COLOR)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be recoded .

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return self._recode(img, random.randint(self.qmin, self.qmax))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(sample, dict):
            sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        else:
            sample = F.normalize(sample, self.mean, self.std)
        return sample


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, dict):
            image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            # image = image.transpose((2, 0, 1))
            return {'image': F.to_tensor(image),
                    'landmarks': landmarks,
                    'pose': pose}
        else:
            return F.to_tensor(sample)
            # return torch.from_numpy(sample)


class RandomOcclusion(object):
    def __init__(self):
        pass

    def __add_occlusions(self, img):
        bkg_size = cfg.CROP_BORDER
        min_occ_size = 30
        max_occ_size = cfg.INPUT_SIZE

        cx = random.randint(bkg_size, bkg_size+cfg.INPUT_SIZE)
        cy = random.randint(bkg_size, bkg_size+cfg.INPUT_SIZE)

        w_half = min(img.shape[1]-cx-1, random.randint(min_occ_size, max_occ_size)) // 2
        h_half = min(img.shape[0]-cy-1, random.randint(min_occ_size, max_occ_size)) // 2
        w_half = min(cx, w_half)
        h_half = min(cy, h_half)

        # l = max(0, w_half+1)
        l = 0
        t = random.randint(h_half+1, cfg.INPUT_SIZE)

        # r = l+2*w_half
        r = bkg_size
        b = min(img.shape[0]-1, t+2*h_half)

        cutout = img[t:b, l:r]
        dst_shape = (2*h_half, 2*w_half)

        if cutout.shape[:2] != dst_shape:
            try:
                cutout = cv2.resize(cutout, dsize=dst_shape[::-1], interpolation=cv2.INTER_CUBIC)
            except:
                print('resize error', img.shape, dst_shape, cutout.shape[:2], cy, cx, h_half, w_half)

        try:
            cutout = cv2.blur(cutout, ksize=(5,5))
            img[cy-h_half:cy+h_half, cx-w_half:cx+w_half] = cutout
        except:
            print(img.shape, dst_shape, cutout.shape[:2], cy, cx, h_half, w_half)
        # plt.imshow(img)
        # plt.show()
        return img

    def __call__(self, sample):
        # res_dict = {}
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
            return {'image': self.__add_occlusions(img), 'landmarks': landmarks, 'pose': pose}
            # img = sample['image']
            # res_dict.update(sample)
        else:
            return self.__add_occlusions(sample)
            # img = sample
        # res_dict['image_mod'] = self.__add_occlusions(img)
        # return res_dict


    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)

