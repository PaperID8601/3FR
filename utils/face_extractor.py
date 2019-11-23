import os
from skimage import io
import numpy as np
import cv2
import config as cfg
from utils import face_processing
from utils.geometry import bboxRelOverlap2, enlargeBB, convertBB2to1, convertBB1to2, scaleBB
from utils.io import makedirs
from skimage import exposure
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageEnhance
import utils.common

file_ext_crops = '.jpg'

# draw the bounding box of the face along with the associated
# probability
def draw_bbox(image, bb, confidence=None):
    (startX, startY, endX, endY) = bb.astype(np.int)
    color = (0,0,255)
    if confidence is not None:
        color = (255,0,0)
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

class NoFacesDetected(ValueError):
    pass

def save_bbox(bbox, crop_filepath):
    x1,y1, x2,y2 = bbox
    bbox_filepath = crop_filepath.replace(file_ext_crops, '.bbx')
    makedirs(bbox_filepath)
    np.savetxt(bbox_filepath, np.array([x1, y1, x2-x1, y2-y1], dtype=np.int), fmt='%d')

class FaceExtractor():
    def __init__(self):
        # modelFile = os.path.join(cfg.MODEL_DIR, "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        # configFile = os.path.join(cfg.MODEL_DIR, "face_detector/deploy.prototxt")
        # self.face_det = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        pass

    def find_face_in_image(self, image, gt_landmarks=None, return_square_bbox=True, show=False):
        if gt_landmarks is not None:
            tl = gt_landmarks.min(axis=0)
            br = gt_landmarks.max(axis=0)
            h, w = br - tl
            tl = tl - (h * 0.50, w * 0.5)
            br = br + (h * 0.50, w * 0.5)

            tl = np.clip(tl.astype(int), a_min=0, a_max=None)
            br = np.clip(br.astype(int), a_min=0, a_max=None)
            roi_tl = tl
            roi = image[tl[1]: br[1], tl[0]: br[0]]
        else:
            roi = image

        disp_image = roi.copy()
        padding = (np.array(roi.shape[:2]) * 0.5).astype(int)
        roi = np.pad(roi, [(padding[0],padding[0]), (padding[1], padding[1]), (0,0)], 'constant')

        (h, w) = roi.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_det.setInput(blob)
        detections = self.face_det.forward()

        highest_confidence = detections[0, 0, 0, 2]
        if highest_confidence < 0.1:
            return np.array([0,0, w, h])

        sorted_det_ids = np.argsort(detections[0,0,:,2]).ravel()[::-1]

        max_overlap = 0
        best_box = None
        best_confidence = 0

        if len(sorted_det_ids) == 0:
            raise NoFacesDetected()

        # loop over the detections
        for i in sorted_det_ids:
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # print(startX, startY, endX, endY)
            box = np.array([startX, startY, endX, endY])
            box[:2] -= padding[::-1]
            box[2:] -= padding[::-1]
            if best_box is None:
                best_box = box
                best_confidence = confidence

            # if endX > roi.shape[1]:
            #     continue
            #     startX -= endX - roi.shape[1]
            #     endX = roi.shape[1]
            # if endY > roi.shape[0]:
            #     continue
            #     startY -= endY - roi.shape[0]
            #     endY = roi.shape[0]
            # print(startX, startY, endX, endY)

            draw_bbox(disp_image, best_box, confidence=best_confidence)

            if gt_landmarks is not None:
                # gt_box = np.array((tl[0], tl[1] , br[0], br[1]))
                # draw_bbox(disp_image, gt_box)

                box[0] += tl[0]
                box[1] += tl[1]
                box[2] += tl[0]
                box[3] += tl[1]
                tl = gt_landmarks.min(axis=0)
                br = gt_landmarks.max(axis=0)
                gt_box = np.array((tl[0], tl[1] , br[0], br[1]))
                overlap = bboxRelOverlap2(box, gt_box)
                if overlap > max_overlap:
                    # print(overlap)
                    max_overlap = overlap
                    best_box = box
                    best_confidence = confidence
                if show:
                    gt_box[0] -= roi_tl[0]
                    gt_box[1] -= roi_tl[1]
                    gt_box[2] -= roi_tl[0]
                    gt_box[3] -= roi_tl[1]
                    draw_bbox(disp_image, gt_box)
            else:
                best_box = box
                best_confidence = confidence
                break

        # reduce face height to make bbox more similar to bboxes based on landmarks. These bboxes don't contain
        # the forehead.
        h = best_box[3] - best_box[1]
        best_box[1] += h * 0.25

        if return_square_bbox:
            h = best_box[3] - best_box[1]
            bb1 = convertBB2to1(best_box)
            bb1[2] = h/2
            best_box = convertBB1to2(bb1)

        if show:
            cv2.imshow('Dets', cv2.cvtColor(disp_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey()

        return best_box

    def get_loose_crop(self, img=None, landmarks=None, bb=None, detect_face=False):
        if detect_face:
            assert(img is not None)
            try:
                bb = self.find_face_in_image(img, gt_landmarks=landmarks)
            except NoFacesDetected:
                print("No faces detected in image")
                bb = np.array([0, 0, img.shape[1], img.shape[0]])
            bb = scaleBB(bb, cfg.LOOSE_BBOX_SCALE, cfg.LOOSE_BBOX_SCALE, typeBB=2).astype(np.int)
        elif landmarks is not None:
            bb = face_processing.get_bbox_from_landmarks(landmarks, loose=True)
            # cv2.rectangle(img, tuple(bb[:2]), tuple(bb[2:]), (255,255,255))
            # cv2.imshow('bb', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # vis.show_landmarks(img, landmarks)
        elif bb is not None:
            bb = enlargeBB(bb, 0.3, 0.3, typeBB=2)
        else:
            raise ValueError('Either landmarks or bb must be supplied or detect_face must be True!')
        return bb


    def get_face(self, filename, fullsize_img_dir, cropped_img_root, crop_type='tight',
                 landmarks=None, pose=None, bb=None, size=(cfg.CROP_SIZE, cfg.CROP_SIZE),
                 use_cache=True, detect_face=False, aligned=False, id=None):

        # assert(not detect_face or crop_type == 'loose')

        load_fullsize = False
        loose_bbox = None

        if crop_type=='fullsize':
            load_fullsize = True
        else:
            crop_dir = crop_type
            if detect_face:
                crop_dir += '_det'
            if not aligned:
                crop_dir += '_noalign'
            filename_noext = os.path.splitext(filename)[0]
            if id is not None:
                filename_noext += '.{:07d}'.format(id)
            cache_filepath = os.path.join(cropped_img_root, crop_dir, filename_noext + file_ext_crops)

            is_cached_crop = False
            if use_cache and os.path.isfile(cache_filepath):
                # Load cached crops
                try:
                    img = io.imread(cache_filepath)
                except:
                    print("\tError: Could load not cropped image {}!".format(cache_filepath))
                    print("\tDeleting file and loading fullsize image.")
                    os.remove(cache_filepath)
                    load_fullsize = True

                is_cached_crop = True
                if crop_type=='loose':
                    [x,y, w,h] = np.loadtxt(cache_filepath.replace(file_ext_crops, '.bbx'))
                    loose_bbox = np.array([x,y, x+w, y+h], dtype=int)
            else:
                load_fullsize = True

        assert(detect_face or landmarks is not None or bb is not None)

        if load_fullsize:
            # Load fullsize image from dataset
            # t = time.time()
            img_path = os.path.join(fullsize_img_dir, filename)
            try:
                img = io.imread(img_path)
            except:
                raise IOError("\tError: Could not load image {}!".format(img_path))
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                print(filename, "converting RGBA to RGB...")
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            assert img.shape[2] == 3, "{}, invalid format: {}".format(img_path, img.shape)
            # print(time.time()-t)

            # def plot_landmarks():
            #     plt.imshow(img, interpolation='nearest')
            #     scs = []
            #     legendNames = []
            #     # for i in range(0,21):
            #     for i in range(0,68):
            #         sc = plt.scatter(landmarks[i,0], landmarks[i,1], s=10.0)
            #         scs.append(sc)
            #         legendNames.append("{}".format(i))
            #     plt.legend(scs, legendNames, scatterpoints=1, loc='best', ncol=4, fontsize=7)
            #     plt.show()

            # import vis
            # vis.show_landmarks(img, landmarks, wait=10)

            if crop_type == 'fullsize':
                fc = face_processing.FaceCrop(img,
                                              bbox=[0,0,img.shape[2], img.shape[1]],
                                              output_size=(img.shape[2], img.shape[1]))
                return img, landmarks, pose, fc

            if crop_type == 'loose':
                loose_bbox = self.get_loose_crop(img, landmarks=landmarks, detect_face=detect_face)

                fc = face_processing.FaceCrop(img, bbox=loose_bbox, output_size=size)
                loose_crop = fc.apply_to_image(with_hist_norm=False)
                if use_cache:
                    save_bbox(loose_bbox, cache_filepath)
                    io.imsave(cache_filepath, loose_crop)


        if loose_bbox is not None:
            cropper = face_processing.FaceCrop(img, bbox=loose_bbox, output_size=size,
                                               img_already_cropped=is_cached_crop)
        elif (landmarks is None or not landmarks.any()):
            assert (bb is not None)
            # Fall back to bounding box if no landmarks found
            # print('falling back to bounding box')
            # crop = face_processing.crop_by_bb(img, face_processing.scale_bb(bb, f=1.075), size=size)
            cropper = face_processing.FaceCrop(img, bbox=bb, img_already_cropped=is_cached_crop)
            # cropper = face_processing.FaceCrop(img, bbox=bb, img_already_cropped=False)
        else:
            cropper = face_processing.FaceCrop(img, landmarks=landmarks, img_already_cropped=is_cached_crop,
                                               align_face_orientation=aligned, output_size=size)

        try:
            crop = cropper.apply_to_image()
        except cv2.error:
            print('Could not crop image {} (load_fullsize=={}).'.format(filename, load_fullsize))
        landmarks, pose = cropper.apply_to_landmarks(landmarks, pose)

        if use_cache and not is_cached_crop:
            makedirs(cache_filepath)
            io.imsave(cache_filepath, crop)

        crop = np.minimum(crop, 255)
        crop = cv2.medianBlur(crop, ksize=3)
        return crop, landmarks, pose, cropper








