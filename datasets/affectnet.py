import os
import cv2
import numpy as np

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils
from utils import face_processing as fp

from constants import *
from utils.nn import to_numpy, Batch
from landmarks.lmutils import create_landmark_heatmaps
from utils.face_extractor import FaceExtractor
import utils.geometry
from torchvision import transforms as tf

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

CLASS_NAMES = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
MAX_IMAGES_PER_EXPRESSION = 1000000


class AffectNet(td.Dataset):

    classes = CLASS_NAMES
    colors = ['tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:red', 'tab:blue']
    markers = ['s', 'o', '>', '<', '^', 'v', 'P', 'd']

    def __init__(self, root_dir=cfg.AFFECTNET_ROOT, train=True,
                 transform=None, crop_type='tight', color=True, start=None, max_samples=None,
                 outlier_threshold=None, deterministic=None, use_cache=True,
                 detect_face=False, align_face_orientation=False, min_conf=cfg.MIN_OPENFACE_CONFIDENCE, daug=0,
                 return_landmark_heatmaps=False, landmark_sigma=9, landmark_ids=range(68),
                 return_modified_images=False, crop_source='lm_openface', **kwargs):
        assert(crop_type in ['fullsize', 'tight', 'loose'])
        assert(crop_source in ['bb_ground_truth', 'lm_ground_truth', 'lm_cnn', 'lm_openface'])

        self.face_extractor = FaceExtractor()

        self.mode = TRAIN if train else VAL

        self.crop_source = crop_source
        self.use_cache = use_cache
        self.detect_face = detect_face
        self.align_face_orientation = align_face_orientation
        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.return_modified_images = return_modified_images
        self.landmark_sigma = landmark_sigma
        self.landmark_ids = landmark_ids

        self.start = start
        self.max_samples = max_samples

        self.root_dir = root_dir
        self.crop_type = crop_type
        self.color = color
        self.outlier_threshold = outlier_threshold
        self.transform = transform
        self.fullsize_img_dir = os.path.join(self.root_dir, 'cropped_Annotated')
        crop_folder = 'crops'
        if cfg.INPUT_SIZE == 128:
            crop_folder += '_128'
        self.cropped_img_dir = os.path.join(self.root_dir, crop_folder, crop_source)
        self.feature_dir = os.path.join(self.root_dir, 'features')

        annotation_filename = 'training' if train else 'validation'
        path_annotations_mod = os.path.join(root_dir, annotation_filename + '.mod.pkl')
        if os.path.isfile(path_annotations_mod):
            print('Reading pickle file...')
            self._annotations = pd.read_pickle(path_annotations_mod)
        else:
            print('Reading CSV file...')
            self._annotations = pd.read_csv(os.path.join(root_dir, annotation_filename+'.csv'))
            print('done.')

            # drop non-faces
            self._annotations = self._annotations[self._annotations.expression < 8]

            # Samples in annotation file are somewhat clustered by expression.
            # Shuffle to create a more even distribution.
            # NOTE: deterministic, always creates the same order
            if train:
                from sklearn.utils import shuffle
                self._annotations = shuffle(self._annotations, random_state=2)

                # remove samples with inconsistent expression<->valence/arousal values
                self._remove_outliers()

            poses = []
            confs = []
            landmarks = []
            for cnt, filename in enumerate(self._annotations.subDirectory_filePath):
                if cnt % 1000 == 0:
                    print(cnt)
                filename_noext = os.path.splitext(filename)[0]
                conf, lms, pose = ds_utils.read_openface_detection(os.path.join(self.feature_dir, filename_noext))
                poses.append(pose)
                confs.append(conf)
                landmarks.append(lms)
            self._annotations['pose'] = poses
            self._annotations['conf'] = confs
            self._annotations['landmarks_of'] = landmarks
            # self.annotations.to_csv(path_annotations_mod, index=False)
            self._annotations.to_pickle(path_annotations_mod)

        # There is (at least) one missing image in the dataset. Remove by checking face width:
        self._annotations = self._annotations[self._annotations.face_width > 0]

        self.rebalance_classes()

        if deterministic is None:
            deterministic = self.mode != TRAIN
        self.transform = ds_utils.build_transform(deterministic, self.color, daug)

        transforms = [fp.CenterCrop(cfg.INPUT_SIZE)]
        transforms += [fp.ToTensor() ]
        transforms += [fp.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]  # VGGFace(2)
        self.crop_to_tensor = tf.Compose(transforms)


    def filter_labels(self, label_dict=None, label_dict_exclude=None):
        if label_dict is not None:
            print("Applying include filter to labels: {}".format(label_dict))
            for k, v in label_dict.items():
                self.annotations = self.annotations[self.annotations[k] == v]
        if label_dict_exclude is not None:
            print("Applying exclude filter to labels: {}".format(label_dict_exclude))
            for k, v in label_dict_exclude.items():
                self.annotations = self.annotations[self.annotations[k] != v]
        print("  Number of images: {}".format(len(self.annotations)))


    def rebalance_classes(self, max_images_per_class=MAX_IMAGES_PER_EXPRESSION):
        if self.mode == TRAIN:
            # balance class sized if neccessary
            print('Limiting number of images to {} per class...'.format(max_images_per_class))
            # self._annotations = self._annotations.groupby('expression').head(5000)
            from sklearn.utils import shuffle
            self._annotations['cls_idx'] = self._annotations.groupby('expression').cumcount()
            self._annotations = shuffle(self._annotations)
            self._annotations_balanced = self._annotations[self._annotations.cls_idx < max_images_per_class]
            print(len(self._annotations_balanced))
        else:
            self._annotations_balanced = self._annotations

        # limit number of samples
        st,nd = 0, None
        if self.start is not None:
            st = self.start
        if self.max_samples is not None:
            nd = st+self.max_samples
        self._annotations_balanced = self._annotations_balanced[st:nd]

    @property
    def labels(self):
        return self.annotations['expression'].values

    @property
    def heights(self):
        return self.annotations.face_height.values

    @property
    def widths(self):
        return self.annotations.face_width.values

    @property
    def annotations(self):
        return self._annotations_balanced

    @annotations.setter
    def annotations(self, new_annots):
        self._annotations_balanced = new_annots

    def print_stats(self):
        print(self._stats_repr())

    def _stats_repr(self):
        labels = self.annotations.expression
        fmt_str =  "    Class sizes:\n"
        for id in np.unique(labels):
            count = len(np.where(labels == id)[0])
            fmt_str += "      {:<6} ({:.2f}%)\t({})\n".format(count, 100.0*count/self.__len__(), self.classes[id])
        fmt_str += "    --------------------------------\n"
        fmt_str += "      {:<6}\n".format(len(labels))
        return fmt_str

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.mode)
        # fmt_str += '    Root Location: {}\n'.format(self.root_dir)
        # tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        fmt_str += self._stats_repr()
        return fmt_str

    def __len__(self):
        return len(self.annotations)

    def get_class_sizes(self):
        groups = self.annotations.groupby(by='expression')
        return groups.size().values

    def parse_landmarks(self, landmarks):
        try:
            vals = [float(s) for s in landmarks.split(';')]
            return np.array([(x, y) for x, y in zip(vals[::2], vals[1::2])], dtype=np.float32)
        except:
            raise ValueError("Invalid landmarks {}".format(landmarks))

    def get_bounding_box(self, sample):
        l,t,w,h = sample.face_x, sample.face_y, sample.face_width, sample.face_height
        r, b = l + w, t + h

        # enlarge bounding box
        if t > b:
            t, b = b, t
        h = b-t
        assert(h >= 0)
        t_new, b_new = int(t + 0.05 * h), int(b + 0.25 * h)

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
        sample = self.annotations.iloc[idx]
        filename = sample.subDirectory_filePath
        pose = sample.pose
        bb = None
        landmarks_for_crop = None
        landmarks_to_return = self.parse_landmarks(sample.facial_landmarks)

        if self.crop_source == 'bb_ground_truth':
            bb = self.get_bounding_box(sample)
        elif self.crop_source == 'lm_ground_truth':
            landmarks_for_crop = landmarks_to_return
        elif self.crop_source == 'lm_openface':
            of_conf, landmarks_for_crop = sample.conf, sample.landmarks_of
            # if OpenFace didn't detect a face, fall back to AffectNet landmarks
            if sample.conf <= 0.1:
                try:
                    landmarks_for_crop = self.parse_landmarks(sample.facial_landmarks)
                except ValueError:
                    pass

        try:
            crop, landmarks, pose, cropper = self.face_extractor.get_face(filename, self.fullsize_img_dir,
                                                                          self.cropped_img_dir, landmarks=landmarks_for_crop,
                                                                          bb=bb, pose=pose, use_cache=self.use_cache,
                                                                          detect_face=False, crop_type=self.crop_type,
                                                                          aligned=self.align_face_orientation)
        except AssertionError:
            print(filename)
            raise


        landmarks, _ = cropper.apply_to_landmarks(landmarks_to_return)
        # vis.show_landmarks(crop, landmarks, title='lms affectnet', wait=0, color=(0,0,255))

        cropped_sample = {'image': crop, 'landmarks': landmarks, 'pose': pose}
        item = self.transform(cropped_sample)

        em_val_ar = np.array([[sample.expression, sample.valence, sample.arousal]], dtype=np.float32)

        result = self.crop_to_tensor(item)

        result.update({
            'id': 0,
            'fnames': filename,
            'expression': em_val_ar
        })

        if self.return_modified_images:
            mod_transforms = tf.Compose([fp.RandomOcclusion()])
            crop_occ = mod_transforms(item['image'])
            crop_occ = self.crop_to_tensor(crop_occ)
            result['image_mod'] = crop_occ

        if self.return_landmark_heatmaps:
            result['lm_heatmaps'] = create_landmark_heatmaps(result['landmarks'], self.landmark_sigma, self.landmark_ids)

        return result

    def get_face(self, filename, size=(cfg.CROP_SIZE, cfg.CROP_SIZE), use_cache=True):
        sample = self._annotations.loc[self._annotations.subDirectory_filePath == filename].iloc[0]
        landmarks = sample.landmarks_of.astype(np.float32)
        pose = sample.pose

        # if OpenFace didn't detect a face, fall back to AffectNet landmarks
        if sample.conf <= 0.9:
            landmarks = self.parse_landmarks(sample.facial_landmarks)

        crop, landmarks, pose, _ = self.face_extractor.get_face(filename, self.fullsize_img_dir, self.cropped_img_dir,
                                                                crop_type='tight', landmarks=landmarks, pose=pose,
                                                                use_cache=True, detect_face=False, size=size)

        return crop, landmarks, pose

    def show_landmarks(self, img, landmarks):
        for lm in landmarks:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 3, (0, 0, 255), -1)
        cv2.imshow('landmarks', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)


if __name__ == '__main__':
    import argparse
    from utils import vis

    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', default=False, type=bool)
    parser.add_argument('--st', default=None, type=int)
    parser.add_argument('--nd', default=None, type=int)
    args = parser.parse_args()

    import torch
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    ds = AffectNet(train=True, start=0, align_face_orientation=False, use_cache=False, crop_source='bb_ground_truth')
    dl = td.DataLoader(ds, batch_size=10, shuffle=False, num_workers=0)
    # print(ds)

    for data in dl:
        # data = next(iter(dl))
        batch = Batch(data, gpu=False)

        gt = to_numpy(batch.landmarks)
        ocular_dists_inner = np.sqrt(np.sum((gt[:, 42] - gt[:, 39]) ** 2, axis=1))
        ocular_dists_outer = np.sqrt(np.sum((gt[:, 45] - gt[:, 36]) ** 2, axis=1))
        ocular_dists = np.vstack((ocular_dists_inner, ocular_dists_outer)).mean(axis=0)
        print(ocular_dists)

        inputs = batch.images.clone()
        ds_utils.denormalize(inputs)
        imgs = vis.add_landmarks_to_images(inputs.numpy(), batch.landmarks.numpy())
        vis.vis_square(imgs, nCols=10, fx=1.0, fy=1.0, normalize=False)