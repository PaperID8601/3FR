import os
import time
import numpy as np

from utils import log, vis, face_processing

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils
from landmarks.lmutils import create_landmark_heatmaps

from constants import *
from utils.face_extractor import FaceExtractor
import utils.geometry
from torchvision import transforms as tf
from utils import face_processing as fp

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class VggFace2(td.Dataset):

    def __init__(self, root_dir=cfg.VGGFACE2_ROOT, train=True, color=True, start=None,
                 max_samples=None, deterministic=None, min_conf=cfg.MIN_OPENFACE_CONFIDENCE, use_cache=True,
                 crop_source='bb_ground_truth', detect_face=False, align_face_orientation=True,
                 return_landmark_heatmaps=False, return_modified_images=False,
                 daug=0, landmark_sigma=None, landmark_ids=None, **kwargs):

        assert(crop_source in ['bb_ground_truth', 'lm_ground_truth', 'lm_cnn', 'lm_openface'])

        self.mode = TRAIN if train else VAL

        self.face_extractor = FaceExtractor()
        self.use_cache = use_cache
        self.detect_face = detect_face
        self.align_face_orientation = align_face_orientation
        self.color = color
        self.crop_source = crop_source
        self.return_landmark_heatmaps = return_landmark_heatmaps
        self.return_modified_images = return_modified_images
        self.landmark_sigma = landmark_sigma
        self.landmark_ids = landmark_ids

        self.root_dir = root_dir
        root_dir_local = cfg.VGGFACE2_ROOT_LOCAL
        split_subfolder = 'train' if train else 'test'
        crop_folder = 'crops'
        if cfg.INPUT_SIZE == 128:
            crop_folder += '_128'
        self.cropped_img_dir = os.path.join(root_dir_local, split_subfolder, crop_folder, crop_source)
        self.fullsize_img_dir = os.path.join(root_dir, split_subfolder, 'imgs')
        self.feature_dir = os.path.join(root_dir_local, split_subfolder, 'features')
        annotation_filename = 'loose_bb_{}.csv'.format(split_subfolder)
        # annotation_filename = 'loose_landmark_{}.csv'.format(split_subfolder)

        # self.path_annotations_mod = os.path.join(root_dir_local, annotation_filename + '.mod_full_of.pkl')
        self.path_annotations_mod = os.path.join(root_dir_local, annotation_filename + '.mod_full.pkl')
        if os.path.isfile(self.path_annotations_mod):
            print('Reading pickle file...')
            self.annotations = pd.read_pickle(self.path_annotations_mod)
            print('done.')
        else:
            print('Reading CSV file...')
            self.annotations = pd.read_csv(os.path.join(self.root_dir, 'bb_landmark', annotation_filename))
            print('done.')

            of_confs, poses, landmarks = [], [], []
            self.annotations = self.annotations[0:4000000]
            self.annotations = self.annotations[self.annotations.H > 80]
            print("Number of images: {}".format(len(self)))

            def get_face_height(lms):
                return lms[8,1] - lms[27,1]

            read_openface_landmarks = True
            if read_openface_landmarks:
                for cnt, filename in enumerate(self.annotations.NAME_ID):
                    filename_noext = os.path.splitext(filename)[0]

                    bb = self.annotations.iloc[cnt][1:5].values
                    expected_face_center = [bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0]

                    conf, lms, pose, num_faces  = ds_utils.read_openface_detection(os.path.join(self.feature_dir, filename_noext),
                                                                       expected_face_center=expected_face_center,
                                                                       use_cache=True, return_num_faces=True)

                    if num_faces > 1:
                        print("Deleting extracted crop for {}...".format(filename))
                        cache_filepath = os.path.join(self.cropped_img_dir, 'tight', filename + '.jpg')
                        if os.path.isfile(cache_filepath):
                            os.remove(cache_filepath)

                    of_confs.append(conf)
                    landmarks.append(lms)
                    poses.append(pose)
                    if (cnt+1) % 10000 == 0:
                        log.info(cnt+1)
                self.annotations['pose'] = poses
                self.annotations['of_conf'] = of_confs
                self.annotations['landmarks_of'] = landmarks

            # assign new continuous ids to persons (0, range(n))
            print("Creating id labels...")
            _ids = self.annotations.NAME_ID
            _ids = _ids.map(lambda x: int(x.split('/')[0][1:]))
            self.annotations['ID'] = _ids

            self.annotations.to_pickle(self.path_annotations_mod)

        min_face_height = 100
        print('Removing faces with height <={:.2f}px...'.format(min_face_height))
        self.annotations = self.annotations[self.annotations.H > min_face_height]
        print("Number of images: {}".format(len(self)))

        # limit number of samples
        st,nd = 0, None
        if start is not None:
            st = start
        if max_samples is not None:
            nd = st+max_samples
        self.annotations = self.annotations[st:nd]

        if deterministic is None:
            deterministic = self.mode != TRAIN
        self.transform = ds_utils.build_transform(deterministic, self.color, daug)

        print("Number of images: {}".format(len(self)))
        print("Number of identities: {}".format(self.annotations.ID.nunique()))

    @property
    def labels(self):
        return self.annotations.ID.values

    @property
    def heights(self):
        return self.annotations.H.values

    @property
    def widths(self):
        return self.annotations.W.values

    def get_bounding_box(self, sample):
        bb = sample[1:5].values.copy()

        # convert from x,y,w,h to x1,y1,x2,y2
        bb[2:] += bb[:2]

        # enlarge bounding box
        l,t,r,b = bb
        if t > b:
            t, b = b, t
        h = b-t
        assert(h >= 0)
        t_new, b_new = int(t + 0.05 * h), int(b + 0.1 * h)

        # set width of bbox same as height
        h_new = b_new - t_new
        cx = (r + l) / 2
        l_new, r_new = cx - h_new/2, cx + h_new/2
        # in case right eye is actually left of right eye...
        if l_new > r_new:
            l_new, r_new = r_new, l_new

        bbox = np.array([l_new, t_new, r_new, b_new], dtype=np.float32)
        scalef = cfg.CROP_SIZE / cfg.INPUT_SIZE
        bbox_crop = utils.geometry.scaleBB(bbox, scalef, scalef, typeBB=2)
        return bbox_crop

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        filename, id  = sample[0], sample.ID
        bb = None
        landmarks_for_crop = None

        if self.crop_source == 'bb_ground_truth':
            bb = self.get_bounding_box(sample)
            pose = np.zeros(3, dtype=np.float32)
        else:
            of_conf, landmarks_for_crop  = sample.of_conf, sample.landmarks_of
            pose = sample.pose

        try:
            crop, landmarks, pose, cropper = self.face_extractor.get_face(filename+'.jpg', self.fullsize_img_dir,
                                                                          self.cropped_img_dir, landmarks=landmarks_for_crop,
                                                                          bb=bb, pose=pose, use_cache=self.use_cache,
                                                                          detect_face=False, crop_type='tight',
                                                                          aligned=self.align_face_orientation)
        except:
            print(filename)
            raise
            # return self.__getitem__(random.randint(0,len(self)-1))

        try:
            landmarks, _ = cropper.apply_to_landmarks(sample.landmarks)
        except AttributeError:
            landmarks = np.zeros((68,2))

        # vis.show_landmarks(crop, landmarks, title='lms', wait=0, color=(0,0,255))

        cropped_sample = {'image': crop, 'landmarks': landmarks.astype(np.float32), 'pose': pose}

        item = self.transform(cropped_sample)

        transforms = [fp.CenterCrop(cfg.INPUT_SIZE)]
        transforms += [fp.ToTensor() ]
        transforms += [fp.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]  # VGGFace(2)
        transforms = tf.Compose(transforms)

        result = transforms(item)

        result.update({
            'id': id,
            'fnames': filename,
            'expression': np.array([[0,0,0]], dtype=np.float32),
        })

        if self.return_modified_images:
            mod_transforms = tf.Compose([fp.RandomOcclusion()])
            crop_occ = mod_transforms(item['image'])
            crop_occ = transforms(crop_occ)
            result['image_mod'] = crop_occ

        # add landmark heatmaps if landmarks enabled
        if self.return_landmark_heatmaps:
            result['lm_heatmaps'] = create_landmark_heatmaps(item['landmarks'], self.landmark_sigma, self.landmark_ids)
        return result


def extract_features(split, st=None, nd=None):
    """ Extract facial features (landmarks, pose,...) from images """
    import glob
    assert(split in ['train', 'test'])
    person_dirs = sorted(glob.glob(os.path.join(cfg.VGGFACE2_ROOT, split, 'imgs', '*')))[st:nd]
    # print(os.path.join(cfg.VGGFACE2_ROOT, split, 'imgs', '*'))
    for cnt, img_dir in enumerate(person_dirs):
        folder_name = os.path.split(img_dir)[1]
        out_dir = os.path.join(cfg.VGGFACE2_ROOT_LOCAL, split, 'features', folder_name)
        log.info("{}/{}".format(cnt, len(person_dirs)))
        face_processing.run_open_face(img_dir, out_dir, is_sequence=False)


def extract_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--st', default=None, type=int)
    parser.add_argument('--nd', default=None, type=int)
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    extract_features(args.split, st=args.st, nd=args.nd)



if __name__ == '__main__':
    # extract_main()
    # exit()

    from utils.nn import Batch
    import utils.common as util
    util.init_random()

    ds = VggFace2(train=True, deterministic=True, use_cache=True, align_face_orientation=False,
                  crop_source='bb_ground_truth', return_modified_images=True)
    micro_batch_loader = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    t = time.perf_counter()
    for iter, data in enumerate(micro_batch_loader):
        print('t load:', time.perf_counter() - t)
        t = time.perf_counter()
        batch = Batch(data, gpu=False)
        print('t Batch:', time.perf_counter() - t)
        images = ds_utils.denormalized(batch.images)
        f = 1.0
        vis.vis_square(images, fx=f, fy=f, normalize=False, nCols=10, wait=0)
