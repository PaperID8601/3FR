import os
import cv2
import numpy as np

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils

# from constants import *
from utils.nn import Batch
from landmarks.lmutils import create_landmark_heatmaps
import utils.geometry
from torchvision import transforms as tf
from utils import face_processing as fp

# Ignore warnings
import warnings
import landmarks.lmconfig as lmcfg

warnings.filterwarnings("ignore")


class W300(td.Dataset):

    CROP_SOURCES = ['bb_detector', 'bb_ground_truth', 'lm_ground_truth', 'lm_cnn', 'lm_openface']

    def __init__(self, root_dir=cfg.W300_ROOT, train=True,
                 transform=None, color=True, start=None, max_samples=None,
                 deterministic=None, align_face_orientation=cfg.CROP_ALIGN_ROTATION,
                 crop_type='tight', test_split='challenging', detect_face=False, use_cache=True,
                 crop_source='bb_detector', daug=0, return_modified_images=False,
                 return_landmark_heatmaps=False, landmark_sigma=3, landmark_ids=range(68), **kwargs):

        assert(crop_type in ['fullsize', 'tight','loose'])
        test_split = test_split.lower()
        assert(test_split in ['common', 'challenging', '300w', 'full'])
        assert(crop_source in W300.CROP_SOURCES)
        lmcfg.config_landmarks('300w')

        self.start = start
        self.max_samples = max_samples
        self.use_cache = use_cache
        self.crop_source = crop_source
        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.return_modified_images = return_modified_images
        self.landmark_sigma = landmark_sigma
        self.landmark_ids = landmark_ids

        self.root_dir = root_dir
        self.local_root_dir = cfg.W300_ROOT_LOCAL
        self.color = color
        self.transform = transform
        self.fullsize_img_dir = os.path.join(self.root_dir, 'images')
        self.align_face_orientation = align_face_orientation
        self.detect_face = detect_face
        self.crop_type = crop_type
        self.cropped_img_dir = os.path.join(cfg.W300_ROOT_LOCAL, 'crops', crop_source)

        self.feature_dir_cnn = os.path.join(cfg.W300_ROOT_LOCAL, 'features_cnn')
        self.feature_dir_of = os.path.join(cfg.W300_ROOT_LOCAL, 'features_of')

        self.bounding_box_dir = os.path.join(cfg.W300_ROOT, 'Bounding Boxes')

        self.split = 'train' if train else test_split
        self.build_annotations(self.split)
        print("Num. images: {}".format(len(self)))

        # limit number of samples
        st,nd = 0, None
        if start is not None:
            st = start
        if max_samples is not None:
            nd = st+max_samples
        self.annotations = self.annotations[st:nd]

        if deterministic is None:
            deterministic = not train
        if self.crop_type == 'tight':
            self.transform = ds_utils.build_transform(deterministic, True, daug)
        elif self.crop_type == 'fullsize':
            self.transform = lambda x:x

        from utils.face_extractor import FaceExtractor
        self.face_extractor = FaceExtractor()

        transforms = [fp.CenterCrop(cfg.INPUT_SIZE)]
        transforms += [fp.ToTensor() ]
        transforms += [fp.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]  # VGGFace(2)
        self.crop_to_tensor = tf.Compose(transforms)


    def build_annotations(self, split):
        import scipy.io
        import glob

        split_defs = {
            'train': [
                ('train/afw', 'afw'),
                ('train/helen', 'helen_trainset'),
                ('train/lfpw', 'lfpw_trainset')
            ],
            'common': [
                ('test/common/helen', 'helen_testset'),
                ('test/common/lfpw', 'lfpw_testset')
            ],
            'challenging': [
                ('test/challenging/ibug', 'ibug')
            ],
            'full': [
                ('test/common/helen', 'helen_testset'),
                ('test/common/lfpw', 'lfpw_testset'),
                ('test/challenging/ibug', 'ibug')
            ],
            '300w': [
                ('test/300W/01_Indoor', None),
                ('test/300W/01_Outdoor', None)
            ]
        }

        ann = []

        bboxes = []
        for id, subset in enumerate(split_defs[split]):

            im_dir, bbox_file_suffix = subset

            # get image file paths and read GT landmarks
            ext = "*.jpg"
            if 'lfpw' in im_dir or '300W' in im_dir:
                ext = "*.png"
            for img_file in sorted(glob.glob(os.path.join(self.fullsize_img_dir, im_dir, ext))):

                path_abs_noext = os.path.splitext(img_file)[0]
                filename_noext =  os.path.split(path_abs_noext)[1]
                path_rel_noext = os.path.join(im_dir, filename_noext)
                filename = os.path.split(img_file)[1]
                path_rel = os.path.join(im_dir, filename)

                # load landmarks from *.pts files
                landmarks =  ds_utils.read_300W_detection(path_abs_noext+'.pts')
                ann.append({'imgName': str(filename), 'fname': path_rel, 'landmarks': landmarks})

            # load supplied detected bounding boxes from MAT file
            if bbox_file_suffix is not None:
                subset_bboxes = scipy.io.loadmat(os.path.join(self.bounding_box_dir, 'bounding_boxes_{}.mat'.format(bbox_file_suffix)))
                for item in subset_bboxes['bounding_boxes'][0]:
                    imgName, bb_detector, bb_ground_truth = item[0][0]
                    bboxes.append({'imgName': str(imgName[0]), 'bb_detector': bb_detector[0], 'bb_ground_truth': bb_ground_truth[0]})


        self._annotations = pd.DataFrame(ann)
        if len(bboxes) > 0:
            df_bboxes = pd.DataFrame(bboxes)
            self._annotations = self._annotations.merge(df_bboxes, on='imgName', how='left')

    @property
    def labels(self):
        return None

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, new_annots):
        self._annotations = new_annots

    def __len__(self):
        return len(self.annotations)

    def get_bounding_box(self, sample):
        bb = sample.bb_detector if self.crop_source == 'bb_detector' else sample.bb_ground_truth

        # enlarge bounding box
        l,t,r,b = bb
        if t > b:
            t, b = b, t
        h = b-t
        assert(h >= 0)

        t_new, b_new = int(t - cfg.CROP_MOVE_TOP_FACTOR * h), int(b + cfg.CROP_MOVE_BOTTOM_FACTOR * h)
        # t_new, b_new = int(t - 0.27 * h), int(b + 0.17 * h)

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


    def __getitem__(self, idx):
        def get_landmarks_for_crop():
            pose = np.zeros(3, dtype=np.float32)
            if self.crop_source == 'lm_openface':
                openface_filepath = os.path.join(self.feature_dir_of, os.path.splitext(filename)[0])
                est_face_center = landmarks_gt.mean(axis=0)
                of_conf, landmarks_of, pose = ds_utils.read_openface_detection(openface_filepath, expected_face_center=est_face_center)
                if of_conf < 0.01:
                    landmarks_of = landmarks_gt
                else:
                    # landmarks_of, pose = self.cropper.apply_crop_to_landmarks(landmarks_of, pose)
                    landmarks_of[:,0] -= cfg.CROP_BORDER
                    landmarks_of[:,1] -= cfg.CROP_BORDER
                landmarks  = landmarks_of
            elif self.crop_source == 'lm_cnn':
                try:
                    landmarks = np.load(os.path.join(self.feature_dir_cnn, os.path.splitext(filename)[0]+'.npy'))
                except FileNotFoundError:
                    landmarks = None
            elif self.crop_source == 'lm_ground_truth':
                landmarks = landmarks_gt
            else:
                # no landmarks -> crop using bounding boxes
                landmarks = None
            return landmarks, pose

        sample = self.annotations.iloc[idx]
        filename = sample.fname
        landmarks_gt = sample.landmarks.astype(np.float32)
        bbox = self.get_bounding_box(sample) if not self.split == '300w' else None

        landmarks_for_crop, pose = get_landmarks_for_crop()

        crop, landmarks, pose, cropper = self.face_extractor.get_face(filename, self.fullsize_img_dir, self.cropped_img_dir,
                                                                      bb=bbox, landmarks=landmarks_for_crop, pose=pose,
                                                                      use_cache=self.use_cache,
                                                                      detect_face=self.detect_face,
                                                                      crop_type=self.crop_type,
                                                                      aligned=self.align_face_orientation)

        landmarks_gt, _ = cropper.apply_to_landmarks(landmarks_gt)
        # self.show_landmarks(crop, landmarks_gt)


        cropped_sample = {'image': crop, 'landmarks': landmarks_gt, 'pose': pose}
        item = self.transform(cropped_sample)


        em_val_ar = np.array([[-1,0,0]], dtype=np.float32)

        result = self.crop_to_tensor(item)

        result.update({
            'fnames': filename,
            'expression': em_val_ar
        })

        if self.return_modified_images:
            mod_transforms = tf.Compose([fp.RandomOcclusion()])
            crop_occ = mod_transforms(item['image'])
            crop_occ = self.crop_to_tensor(crop_occ)
            result['image_mod'] = crop_occ

        # add landmark heatmaps if landmarks enabled
        if self.return_landmark_heatmaps:
            result['lm_heatmaps'] = create_landmark_heatmaps(result['landmarks'], self.landmark_sigma, self.landmark_ids)

        return result

    def show_landmarks(self, img, landmarks):
        for lm in landmarks:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 3, (0, 0, 255), -1)
        cv2.imshow('landmarks', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)



if __name__ == '__main__':

    from utils import vis
    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    ds = W300(train=True, deterministic=True, use_cache=True,
              test_split='challenging', daug=0, align_face_orientation=False,
              return_modified_images=False)
    dl = td.DataLoader(ds, batch_size=50, shuffle=False, num_workers=0)
    print(ds)

    cfg.WITH_LANDMARK_LOSS = False

    for data in dl:
        batch = Batch(data, gpu=False)
        inputs = batch.images.clone()
        imgs = vis._to_disp_images(inputs, denorm=True)
        imgs = vis.add_landmarks_to_images(imgs, batch.landmarks, radius=3, color=(0,255,0))
        # imgs = vis.add_landmarks_to_images(imgs, data['landmarks_of'].numpy(), color=(1,0,0))
        vis.vis_square(imgs, nCols=5, fx=1, fy=1, normalize=False)