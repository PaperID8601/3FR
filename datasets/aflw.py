import os
import numpy as np
from utils import vis

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils
from landmarks.lmutils import create_landmark_heatmaps
from landmarks import lmconfig as lmcfg

from constants import *
import utils.geometry

from torchvision import transforms as tf
from utils import face_processing as fp

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AFLW(td.Dataset):

    def __init__(self, root_dir=cfg.AFLW_ROOT, train=True, color=True, start=None,
                 max_samples=None, deterministic=None, use_cache=True,
                 daug=0, return_modified_images=False, test_split='full', align_face_orientation=True,
                 return_landmark_heatmaps=False, landmark_sigma=9, landmark_ids=range(19), **kwargs):

        assert test_split in ['full', 'frontal']
        from utils.face_extractor import FaceExtractor
        self.face_extractor = FaceExtractor()

        self.use_cache = use_cache
        self.align_face_orientation = align_face_orientation

        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.return_modified_images = return_modified_images
        self.landmark_sigma = landmark_sigma
        self.landmark_ids = landmark_ids

        self.mode = TRAIN if train else VAL

        self.root_dir = root_dir
        root_dir_local = cfg.AFLW_ROOT_LOCAL
        self.fullsize_img_dir = os.path.join(root_dir, 'data/flickr')
        self.cropped_img_dir = os.path.join(root_dir_local, 'crops')
        self.feature_dir = os.path.join(root_dir_local,  'features')
        self.color = color

        annotation_filename = os.path.join(cfg.AFLW_ROOT_LOCAL, 'alfw.pkl')
        self.annotations_original = pd.read_pickle(annotation_filename)
        print("Number of images: {}".format(len(self.annotations_original)))

        self.frontal_only = test_split == 'frontal'
        self.make_split(train, self.frontal_only)

        # limit number of samples
        st,nd = 0, None
        if start is not None:
            st = start
        if max_samples is not None:
            nd = st+max_samples
        self.annotations = self.annotations[st:nd]

        if deterministic is None:
            deterministic = self.mode != TRAIN
        self.transform = ds_utils.build_transform(deterministic, True, daug)

        transforms = [fp.CenterCrop(cfg.INPUT_SIZE)]
        transforms += [fp.ToTensor() ]
        transforms += [fp.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]  # VGGFace(2)
        self.crop_to_tensor = tf.Compose(transforms)

        print("Number of images: {}".format(len(self)))
        # print("Number of identities: {}".format(self.annotations.id.nunique()))

    @property
    def labels(self):
        return self.annotations.ID.values

    @property
    def heights(self):
        return self.annotations.face_h.values

    @property
    def widths(self):
        return self.annotations.face_w.values

    def make_split(self, train, only_frontal):
        import scipy.io
        # Additional annotations from http://mmlab.ie.cuhk.edu.hk/projects/compositional.html
        annots = scipy.io.loadmat(os.path.join(cfg.AFLW_ROOT_LOCAL, 'AFLWinfo_release.mat'))

        train_ids, test_ids = annots['ra'][0][:20000] - 1, annots['ra'][0][20000:] - 1
        ids = annots['ra'][0] - 1


        # merge original and additional annotations

        lms = annots['data'][ids]
        lms = np.dstack((lms[:,:19], lms[:, 19:]))
        lms_list = [l for l in lms]
        mask_new = annots['mask_new'][ids]

        # mask_all_lms_visible = np.stack(mask_new).min(axis=1) == 1

        bbox = annots['bbox'][ids]
        x1, x2, y1, y2 = bbox[:,0], bbox[:,1], bbox[:, 2], bbox[:, 3]
        fnames = [f[0][0] for f in annots['nameList'][ids]]
        annotations_additional = pd.DataFrame({
                                               'fname':fnames,
                                               'ra': ids,
                                               'landmarks_full':lms_list,
                                               'masks': [m for m in mask_new],
                                               'face_x': x1,
                                               'face_y': y1,
                                               'face_w': x2 - x1,
                                               'face_h': y2 - y1
            })

        ad = annotations_additional
        ao = self.annotations_original

        # self.annotations_test = self.annotations_original[self.annotations.fname.isin(fnames)]
        pd.set_option('display.expand_frame_repr', False)
        self.annotations = pd.merge(ad, ao, on=['fname',
                                                'face_x',
                                                'face_y',
                                                'face_w',
                                                'face_h'
                                                ])
        self.annotations = self.annotations.sort_values('ra')


        split_ids = train_ids if train else test_ids
        self.annotations = self.annotations[self.annotations.ra.isin(split_ids)]

        if not train and only_frontal:
            mask_all_lms_visible = np.stack(self.annotations.masks.values).min(axis=1) == 1
            self.annotations = self.annotations[mask_all_lms_visible]
            print(len(self.annotations))


    def get_bounding_box(self, sample):
        l,t,w,h = sample.face_x, sample.face_y, sample.face_w, sample.face_h
        r, b = l + w, t + h

        # enlarge bounding box
        if t > b:
            t, b = b, t
        h = b-t
        assert(h >= 0)
        t_new, b_new = int(t - 0.05 * h), int(b + 0.08 * h)

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

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        # assert sample.fname == sample.fname_full

        face_id = sample.ra
        filename  = sample.fname
        bb = self.get_bounding_box(sample)

        try:
            crop, landmarks, pose, cropper = self.face_extractor.get_face(filename,
                                                                          self.fullsize_img_dir,
                                                                          self.cropped_img_dir,
                                                                          bb=bb,
                                                                          use_cache=self.use_cache,
                                                                          id=face_id)
        except:
            print(filename)
            raise

        landmarks, _ = cropper.apply_to_landmarks(sample.landmarks_full)
        # vis.show_landmarks(crop, landmarks, title='lms aflw', wait=0, color=(0,0,255))

        cropped_sample = {'image': crop,
                          'landmarks': landmarks.astype(np.float32),
                          # 'landmarks': np.zeros((68,2), dtype=np.float32),
                          'pose':  np.zeros(3, dtype=np.float32)}
        item = self.transform(cropped_sample)

        em_val_ar = np.array([[-1,0,0]], dtype=np.float32)

        result = self.crop_to_tensor(item)

        result.update({
            'fnames': filename,
            'expression': em_val_ar,
            'id': 0
        })

        if self.return_modified_images:
            mod_transforms = tf.Compose([fp.RandomOcclusion()])
            crop_occ = mod_transforms(item['image'])
            crop_occ = self.crop_to_tensor(crop_occ)
            result['image_mod'] = crop_occ

        if self.return_landmark_heatmaps:
            result['lm_heatmaps'] = create_landmark_heatmaps(result['landmarks'], self.landmark_sigma, self.landmark_ids)

        return result



if __name__ == '__main__':

    from utils.nn import Batch
    import utils.common

    utils.common.init_random()

    lmcfg.config_landmarks('aflw')

    ds = AFLW(train=True, deterministic=True, use_cache=True)
    dl = td.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    cfg.WITH_LANDMARK_LOSS = False

    for data in dl:
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        ds_utils.denormalize(inputs)
        imgs = vis.add_landmarks_to_images(inputs.numpy(), batch.landmarks.numpy(), radius=3, color=(0,255,0))
        print(batch.fnames)
        vis.vis_square(imgs, nCols=1, fx=1.0, fy=1.0, normalize=False)
