import cv2
import os
import numpy as np

import torch
import torch.nn as nn

import config as cfg
import networks.invresnet
from datasets import ds_utils
from utils import vis
from networks.archs import D_net_gauss, Discriminator
from networks import resnet_ae, archs
import utils.nn

from utils.nn import to_numpy
import landmarks.lmconfig as lmcfg
import landmarks.lmutils
import matplotlib.pyplot as plt


def calc_acc(outputs, labels):
    assert(outputs.shape[1] == 8)
    assert(len(outputs) == len(labels))
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels)
    acc = corrects.double()/float(outputs.size(0))
    return acc.item()

def pearson_dist(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    r = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)))
    return 1 - r.abs().mean()


def resize_image_batch(X, target_size):
    resize = lambda im: cv2.resize(im, dsize=tuple(target_size), interpolation=cv2.INTER_CUBIC)
    X = X.cpu()
    imgs = [i.permute(1, 2, 0).numpy() for i in X]
    imgs = [resize(i) for i in imgs]
    tensors = [torch.from_numpy(i).permute(2, 0, 1) for i in imgs]
    return torch.stack(tensors)


def load_net(modelname):
    modelfile = os.path.join(cfg.SNAPSHOT_DIR, modelname)
    net = SAAE()
    print("Loading model {}...".format(modelfile))
    utils.nn.read_model(modelfile, 'saae', net)
    meta = utils.nn.read_meta(modelfile)
    print("Model trained for {} iterations.".format(meta['total_iter']))
    return net


class SAAE(nn.Module):
    def __init__(self, pretrained_encoder=False):
        super(SAAE, self).__init__()

        self.z_dim = cfg.EMBEDDING_DIMS
        input_channels = 3

        self.Q = resnet_ae.resnet18(pretrained=pretrained_encoder,
                                    num_classes=self.z_dim,
                                    input_size=cfg.INPUT_SIZE,
                                    input_channels=input_channels,
                                    layer_normalization=cfg.ENCODER_LAYER_NORMALIZATION).cuda()

        decoder_class = networks.invresnet.InvResNet

        num_blocks = [cfg.DECODER_PLANES_PER_BLOCK] * 4
        self.P = decoder_class(networks.invresnet.InvBasicBlock,
                               num_blocks,
                               input_dims=self.z_dim,
                               output_size=cfg.INPUT_SIZE,
                               output_channels=input_channels,
                               layer_normalization=cfg.DECODER_LAYER_NORMALIZATION).cuda()

        self.D_z = D_net_gauss(self.z_dim).cuda()
        self.D = Discriminator().cuda()

        self.LMH = networks.invresnet.LandmarkHeadV2(networks.invresnet.InvBasicBlock,
                                                     [cfg.DECODER_PLANES_PER_BLOCK]*4,
                                                     output_size=lmcfg.HEATMAP_SIZE,
                                                     output_channels=lmcfg.NUM_LANDMARK_HEATMAPS,
                                                     layer_normalization='batch').cuda()

        def count_parameters(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        print("Trainable params Q: {:,}".format(count_parameters(self.Q)))
        print("Trainable params P: {:,}".format(count_parameters(self.P)))
        # print("Trainable params D_z: {:,}".format(count_parameters(self.D_z)))
        # print("Trainable params D: {:,}".format(count_parameters(self.D)))
        print("Trainable params LMH: {:,}".format(count_parameters(self.LMH)))

        self.total_iter = 0
        self.iter = 0
        self.z = None
        self.images = None
        self.current_dataset = None


    def z_vecs(self):
        return [to_numpy(self.z)]

    def heatmaps_to_landmarks(self, hms):
        lms = np.zeros((len(hms), lmcfg.NUM_LANDMARKS, 2), dtype=int)
        if hms.shape[1] > 3:
            # print(hms.max())
            for i in range(len(hms)):
                if hms.shape[1] not in [19, 68, 98]:
                    _, lm_coords = landmarks.lmutils.decode_heatmaps(to_numpy(hms[i]))
                    lms[i] = lm_coords
                else:
                    heatmaps = to_numpy(hms[i])
                    for l in range(len(heatmaps)):
                        hm = heatmaps[lmcfg.LANDMARK_ID_TO_HEATMAP_ID[l]]
                        # hm = cv2.blur(hm, (9,9))
                        # hm = cv2.medianBlur(hm, 9,9)
                        lms[i, l, :] = np.unravel_index(np.argmax(hm, axis=None), hm.shape)[::-1]
        elif hms.shape[1] == 3:
            hms = to_numpy(hms)
            def get_score_plane(h, lm_id, cn):
                v = utils.nn.lmcolors[lm_id, cn]
                hcn = h[cn]
                hcn[hcn < v-2] = 0; hcn[hcn > v+5] = 0
                return hcn
            hms *= 255
            for i in range(len(hms)):
                hm = hms[i]
                for l in landmarks.config.LANDMARKS:
                    lm_score_map = get_score_plane(hm, l, 0) * get_score_plane(hm, l, 1) * get_score_plane(hm, l, 2)
                    lms[i, l, :] = np.unravel_index(np.argmax(lm_score_map, axis=None), lm_score_map.shape)[::-1]
        lm_scale = lmcfg.HEATMAP_SIZE / cfg.INPUT_SIZE
        return lms / lm_scale

    def landmarks_pred(self):
        try:
            if self.landmark_heatmaps is not None:
                return self.heatmaps_to_landmarks(self.landmark_heatmaps)
        except AttributeError:
            pass
        return None

    def detect_landmarks(self, X):
        X_recon = self.forward(X)
        X_lm_hm = self.LMH(self.P)
        X_lm_hm = landmarks.lmutils.decode_heatmap_blob(X_lm_hm)
        X_lm_hm = landmarks.lmutils.smooth_heatmaps(X_lm_hm)
        lm_preds = to_numpy(self.heatmaps_to_landmarks(X_lm_hm))
        return X_recon, lm_preds, X_lm_hm

    def forward(self, X):
        self.z = self.Q(X)
        outputs = self.P(self.z)
        self.landmark_heatmaps = None
        if outputs.shape[1] > 3:
            self.landmark_heatmaps = outputs[:,3:]
        return outputs[:,:3]


def vis_reconstruction(net, inputs, ids=None, clips=None, poses=None, emotions=None, landmarks=None, landmarks_pred=None,
                       pytorch_ssim=None, fx=0.5, fy=0.5, ncols=10):
    net.eval()
    cs_errs = None
    with torch.no_grad():
        X_recon = net(inputs)

        if pytorch_ssim is not None:
            cs_errs = np.zeros(len(inputs))
            for i in range(len(cs_errs)):
                cs_errs[i] = 1 - pytorch_ssim(inputs[i].unsqueeze(0), X_recon[i].unsqueeze(0)).item()

    inputs_resized = inputs
    landmarks_resized = landmarks
    if landmarks is not None:
        landmarks_resized = landmarks.cpu().numpy().copy()
        landmarks_resized[...,0] *= inputs_resized.shape[3]/inputs.shape[3]
        landmarks_resized[...,1] *= inputs_resized.shape[2]/inputs.shape[2]

    return vis.draw_results(inputs_resized, X_recon, net.z_vecs(),
                            landmarks=landmarks_resized,
                            landmarks_pred=landmarks_pred,
                            cs_errs=cs_errs,
                            fx=fx, fy=fy, ncols=ncols)



