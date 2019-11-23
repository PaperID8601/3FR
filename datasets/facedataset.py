import os
import numpy as np

from constants import *
import config as cfg
import torch.utils.data as td
from datasets import ds_utils
import utils.geometry
from torchvision import transforms as tf
from utils import face_processing as fp


class FaceDataset(td.Dataset):

    def __init__(self, root_dir, fullsize_img_dir, root_dir_local=None, train=True, crop_type='tight',
                 color=True, start=None, max_samples=None, deterministic=None, use_cache=True,
                 align_face_orientation=False, return_modified_images=False,
                 return_landmark_heatmaps=True, landmark_sigma=9, landmark_ids=range(68), test_split='fullset',
                 crop_source='bb_detector', daug=0, **kwargs):

        from utils.face_extractor import FaceExtractor
        self.face_extractor = FaceExtractor()

        self.test_split = test_split
        self.train = train
        self.mode = TRAIN if train else VAL
        self.use_cache = use_cache
        self.crop_source = crop_source
        self.crop_type = crop_type
        self.align_face_orientation = align_face_orientation
        self.start = start
        self.max_samples = max_samples
        self.daug = daug
        self.return_modified_images = return_modified_images

        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.landmark_sigma = landmark_sigma
        self.landmark_ids = landmark_ids

        self.deterministic = deterministic
        if self.deterministic is None:
            self.deterministic = self.mode != TRAIN

        self.fullsize_img_dir = fullsize_img_dir

        self.root_dir = root_dir
        self.root_dir_local = root_dir_local if root_dir_local is not None else self.root_dir

        self.cropped_img_dir = os.path.join(self.root_dir_local, 'crops', crop_source)
        self.feature_dir = os.path.join(self.root_dir_local,  'features')
        self.color = color

        self.transform = ds_utils.build_transform(self.deterministic, self.color, daug)

        print("Loading annotations... ")
        self.annotations = self.create_annotations()
        print("  Number of images: {}".format(len(self.annotations)))

        self.init()
        self.select_samples()

        transforms = [fp.CenterCrop(cfg.INPUT_SIZE)]
        transforms += [fp.ToTensor() ]
        transforms += [fp.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]  # VGGFace(2)
        self.crop_to_tensor = tf.Compose(transforms)

    def init(self):
        pass

    def create_annotations(self):
        raise NotImplementedError

    def filter_labels(self, label_dict):
        import collections
        print("Applying filter to labels: {}".format(label_dict))
        for k, v in label_dict.items():
            if isinstance(v, collections.Sequence):
                selected_rows = self.annotations[k].isin(v)
            else:
                selected_rows = self.annotations[k] == v
            self.annotations = self.annotations[selected_rows]
        print("  Number of images: {}".format(len(self.annotations)))

    def select_samples(self):
        print("Limiting number of samples...")
        # limit number of samples
        st,nd = 0, None
        if self.start is not None:
            st = self.start
        if self.max_samples is not None:
            nd = st + self.max_samples
        self.annotations = self.annotations[st:nd]
        print("  Number of images: {}".format(len(self.annotations)))

    @property
    def labels(self):
        return self.annotations.ID.values

    def get_crop_extend_factors(self):
        # return cfg.CROP_MOVE_TOP_FACTOR, cfg.CROP_MOVE_BOTTOM_FACTOR
        return 0, 0

    def get_adjusted_bounding_box(self, l, t, w, h):
        # l,t,w,h = sample.face_x, sample.face_y, sample.face_w, sample.face_h
        r, b = l + w, t + h

        # enlarge bounding box
        if t > b:
            t, b = b, t
        h = b-t
        assert(h >= 0)
        extend_top, extend_bottom = self.get_crop_extend_factors()
        t_new, b_new = int(t - extend_top * h), int(b + extend_bottom * h)

        # set width of bbox same as height
        h_new = b_new - t_new
        cx = (r + l) / 2
        l_new, r_new = cx - h_new/2, cx + h_new/2
        # in case right eye is actually left of right eye...
        if l_new > r_new:
            l_new, r_new = r_new, l_new

        # extend area by crop border margins
        bbox = np.array([l_new, t_new, r_new, b_new], dtype=np.float32)
        scalef = cfg.CROP_SIZE / cfg.INPUT_SIZE
        bbox_crop = utils.geometry.scaleBB(bbox, scalef, scalef, typeBB=2)
        return bbox_crop

    def get_expression(self, sample):
        return np.array([[0,0,0]], dtype=np.float32)

    def get_identity(self, sample):
        return -1

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        raise NotImplementedError
