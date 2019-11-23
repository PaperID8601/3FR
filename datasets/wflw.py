import os
import numpy as np
from utils import vis

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils

from datasets.facedataset import FaceDataset
from landmarks.lmutils import create_landmark_heatmaps, lm98_to_lm68
from torchvision import transforms as tf
from utils import face_processing as fp

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

subsets = ['pose', 'illumination', 'expression', 'make-up', 'occlusion', 'blur']

class WFLW(FaceDataset):
    def __init__(self, root_dir=cfg.WFLW_ROOT, root_dir_local=cfg.WFLW_ROOT_LOCAL,
                 return_landmark_heatmaps=True, landmark_ids=range(98), **kwargs):
        fullsize_img_dir=os.path.join(root_dir, 'WFLW_images')
        super().__init__(root_dir=root_dir, root_dir_local=root_dir_local, fullsize_img_dir=fullsize_img_dir,
                         return_landmark_heatmaps=return_landmark_heatmaps,
                         landmark_ids=landmark_ids, **kwargs)

    def init(self):
        if not self.train:
            if self.test_split in subsets:
                self.filter_labels({self.test_split:1})

    def get_crop_extend_factors(self):
        return 0.0, 0.1  # standard

    def parse_groundtruth_txt(self, gt_txt_file):
        num_lm_cols = 98*2
        columns_names = [
            'x',
            'y',
            'x2' ,
            'y2',
            'pose',
            'expression',
            'illumination',
            'make-up',
            'occlusion',
            'blur',
            'fname'
        ]
        ann = pd.read_csv(gt_txt_file,
                               header=None,
                               sep=' ',
                               usecols=range(num_lm_cols, num_lm_cols+11),
                               names=columns_names)
        ann['w'] = ann['x2'] - ann['x']
        ann['h'] = ann['y2'] - ann['y']

        landmarks = pd.read_csv(gt_txt_file,
                              header=None,
                              sep=' ',
                              usecols=range(0, num_lm_cols)).values

        ann['landmarks'] = [i for i in landmarks.reshape((-1, num_lm_cols//2, 2))]
        return ann

    def create_annotations(self):
        split_name = 'train' if self.train else 'test'
        annotation_filename = os.path.join(self.root_dir_local, '{}_{}.pkl'.format(self.name, split_name))
        if os.path.isfile(annotation_filename):
            # print('Reading pickle file...')
            ann = pd.read_pickle(annotation_filename)
            # print('done.')
        else:
            print('Reading txt file...')
            gt_txt_file = os.path.join(self.root_dir,
                                       'WFLW_annotations',
                                       'list_98pt_rect_attr_train_test',
                                       'list_98pt_rect_attr_'+split_name+'.txt')
            ann = self.parse_groundtruth_txt(gt_txt_file)
            ann.to_pickle(annotation_filename)
            print('done.')
        return ann

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        filename  = sample.fname
        bb = self.get_adjusted_bounding_box(sample.x, sample.y, sample.w, sample.h)
        face_id = int(sample.name)

        if self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = lm98_to_lm68(sample.landmarks.astype(np.float32))
        else:
            # no landmarks -> crop using bounding boxes
            landmarks_for_crop = None

        try:
            crop, landmarks, pose, cropper = self.face_extractor.get_face(filename,
                                                                          self.fullsize_img_dir,
                                                                          self.cropped_img_dir,
                                                                          crop_type=self.crop_type,
                                                                          landmarks=landmarks_for_crop,
                                                                          bb=bb,
                                                                          use_cache=self.use_cache,
                                                                          id=face_id,
                                                                          aligned=self.align_face_orientation)
        except:
            print('Could not load image {}'.format(filename))
            raise

        landmarks_gt = sample.landmarks.astype(np.float32)
        landmarks_gt = cropper.apply_to_landmarks(landmarks_gt)[0]

        cropped_sample = {'image': crop,
                          'landmarks': landmarks_gt,
                          'pose':  np.zeros(3, dtype=np.float32)}
        item = self.transform(cropped_sample)

        em_val_ar = np.array([[-1,0,0]], dtype=np.float32)

        if self.crop_type != 'fullsize':
            result = self.crop_to_tensor(item)
        else:
            result = item

        result.update({
            'fnames': filename,
            'expression': em_val_ar,
            'bb': bb
        })

        if self.return_modified_images:
            mod_transforms = tf.Compose([fp.RandomOcclusion()])
            crop_occ = mod_transforms(item['image'])
            crop_occ = self.crop_to_tensor(crop_occ)
            result['image_mod'] = crop_occ

        if self.return_landmark_heatmaps and self.crop_type != 'fullsize':
            result['lm_heatmaps'] = create_landmark_heatmaps(result['landmarks'], self.landmark_sigma, self.landmark_ids)

        return result


if __name__ == '__main__':
    from utils.nn import Batch,to_numpy
    import utils.common
    from landmarks import lmconfig as lmcfg

    utils.common.init_random(3)
    lmcfg.config_landmarks('wflw')

    ds = WFLW(train=True, deterministic=True, use_cache=True, daug=0)
    # ds.filter_labels({'pose':0, 'blur':0, 'occlusion':1})
    dl = td.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    cfg.WITH_LANDMARK_LOSS = False

    for data in dl:
        batch = Batch(data, gpu=False)
        images = vis._to_disp_images(batch.images, denorm=True)
        # lms = lmutils.convert_landmarks(to_numpy(batch.landmarks), lmutils.LM98_TO_LM68)
        lms = batch.landmarks
        images = vis.add_landmarks_to_images(images, lms, draw_wireframe=False, color=(0,255,0), radius=3)
        vis.vis_square(images, nCols=1, fx=1., fy=1., normalize=False)
