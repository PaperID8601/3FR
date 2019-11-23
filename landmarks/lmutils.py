import numpy as np
import config as cfg
import landmarks.lmconfig as lmcfg
from utils.nn import to_numpy, to_image
import utils.nn as nn
from utils import vis
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import torch
from datasets import ds_utils

layers = []

outline = range(0, 17)
eyebrow_l = range(17, 22)
eyebrow_r = range(22, 27)
nose = range(27, 31)
nostrils = range(31, 36)
eye_l = range(36, 42)
eye_r = range(42, 48)
mouth = range(48, 68)


components = [outline, eyebrow_l, eyebrow_r, nose, nostrils, eye_l, eye_r, mouth]

new_layers = []
for idx in range(20):
    lm_ids = []
    for comp in components[1:]:
        if len(comp) > idx:
            lm = comp[idx]
            lm_ids.append(lm)
    new_layers.append(lm_ids)

outline_layers = [[lm] for lm in range(17)]

layers = components + new_layers + outline_layers

hm_code_mat = np.zeros((len(layers), 68), dtype=bool)
for l, lm_ids in enumerate(layers):
    hm_code_mat[l, lm_ids] = True


def generate_colors(n, r, g, b, dim):
    ret = []
    step = [0,0,0]
    step[dim] =  256 / n
    for i in range(n):
        r += step[0]
        g += step[1]
        b += step[2]
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

_colors = generate_colors(17, 220, 0, 0, 2) + \
          generate_colors(10, 0, 240, 0, 0) + \
          generate_colors(9, 0, 0, 230, 1) + \
          generate_colors(12, 100, 255, 0, 2) + \
          generate_colors(20, 150, 0, 255, 2)
# lmcolors = np.array(_colors)
np.random.seed(0)
lmcolors = np.random.randint(0,255,size=(68,3))
lmcolors = lmcolors / lmcolors.sum(axis=1).reshape(-1,1)*255


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x-mean)**2 / (2 * sigma**2))


def make_landmark_template(wnd_size, sigma):
    X, Y = np.mgrid[-wnd_size//2:wnd_size//2, -wnd_size//2:wnd_size//2]
    Z = np.sqrt(X**2 + Y**2)
    N = gaussian(Z, 0, sigma)
    # return (N/N.max())**2  # square to make sharper
    return (N/N.max())


def _fill_heatmap_layer(dst, lms, lm_id, lm_heatmap_window, wnd_size):
    posx, posy = min(lms[lm_id,0], lmcfg.HEATMAP_SIZE-1), min(lms[lm_id,1], lmcfg.HEATMAP_SIZE-1)

    img_size = lmcfg.HEATMAP_SIZE
    l = int(posx - wnd_size/2)
    t = int(posy - wnd_size/2)
    r = l + wnd_size
    b = t + wnd_size

    src_l = max(0, -l)
    src_t = max(0, -t)
    src_r = min(wnd_size, wnd_size-(r-img_size))
    src_b = min(wnd_size, wnd_size-(b-img_size))

    try:
        cn = lmcfg.LANDMARK_ID_TO_HEATMAP_ID[lm_id]
        wnd = lm_heatmap_window[src_t:src_b, src_l:src_r]
        weight = 1.0
        dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)] = np.maximum(
            dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)], wnd*weight)
    except:
        pass


def __get_code_mat(num_landmarks):
    def to_binary(n, ndigits):
        bits =  np.array([bool(int(x)) for x in bin(n)[2:]])
        assert len(bits) <= ndigits
        zero_pad_bits = np.zeros(ndigits, dtype=bool)
        zero_pad_bits[-len(bits):] = bits
        return zero_pad_bits

    n_enc_layers = int(np.ceil(np.log2(num_landmarks)))

    # get binary code for each heatmap id
    codes = [to_binary(i+1, ndigits=n_enc_layers) for i in range(num_landmarks)]
    return np.vstack(codes)


def convert_to_encoded_heatmaps(hms):

    def merge_layers(hms):
        hms = hms.max(axis=0)
        return hms/hms.max()

    num_landmarks = len(hms)
    n_enc_layers = len(hm_code_mat)

    # create compressed heatmaps by merging layers according to transpose of binary code mat
    encoded_hms = np.zeros((n_enc_layers, hms.shape[1], hms.shape[2]))
    for l in range(n_enc_layers):
        selected_layer_ids = hm_code_mat[l,:]
        encoded_hms[l] = merge_layers(hms[selected_layer_ids].copy())
    decode_heatmaps(encoded_hms)
    return encoded_hms


def convert_to_hamming_encoded_heatmaps(hms):

    def merge_layers(hms):
        hms = hms.max(axis=0)
        return hms/hms.max()

    num_landmarks = len(hms)
    n_enc_layers = int(np.ceil(np.log2(num_landmarks)))
    code_mat = __get_code_mat(num_landmarks)

    # create compressed heatmaps by merging layers according to transpose of binary code mat
    encoded_hms = np.zeros((n_enc_layers, hms.shape[1], hms.shape[2]))
    for l in range(n_enc_layers):
        selected_layer_ids = code_mat[:, l]
        encoded_hms[l] = merge_layers(hms[selected_layer_ids].copy())
    # decode_heatmaps(encoded_hms)
    return encoded_hms


def decode_heatmap_blob(hms):
    assert len(hms.shape) == 4
    if hms.shape[1] == lmcfg.NUM_LANDMARK_HEATMAPS: # no decoding necessary
        return hms
    assert hms.shape[1] == len(hm_code_mat)
    hms68 = np.zeros((hms.shape[0], 68, hms.shape[2], hms.shape[3]), dtype=np.float32)
    for img_idx in range(len(hms)):
        hms68[img_idx] = decode_heatmaps(to_numpy(hms[img_idx]))[0]
    return hms68


def decode_heatmaps(hms):
    import cv2
    def get_decoded_heatmaps_for_layer(hms, lm):
        show = False
        enc_layer_ids = code_mat[:, lm]
        heatmap = np.ones_like(hms[0])
        for i in range(len(enc_layer_ids)):
            pos = enc_layer_ids[i]
            layer = hms[i]
            if pos:
                if show:
                    fig, ax = plt.subplots(1,4)
                    print(i, pos)
                    ax[0].imshow(heatmap, vmin=0, vmax=1)
                    ax[1].imshow(layer, vmin=0, vmax=1)
                # mask = layer.copy()
                # mask[mask < 0.1] = 0
                # heatmap *= mask
                heatmap *= layer
                if show:
                    # ax[2].imshow(mask, vmin=0, vmax=1)
                    ax[3].imshow(heatmap, vmin=0, vmax=1)

        return heatmap

    num_landmarks = 68

    # get binary code for each heatmap id
    code_mat = hm_code_mat

    decoded_hms = np.zeros((num_landmarks, hms.shape[1], hms.shape[1]))

    show = False
    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(code_mat)
        fig_dec, ax_dec = plt.subplots(7, 10)
        fig, ax = plt.subplots(5,9)
        for i in range(len(hms)):
            ax[i//9, i%9].imshow(hms[i])

    lms = np.zeros((68,2), dtype=int)
    # lmid_to_show = 16

    for lm in range(0,68):

        heatmap = get_decoded_heatmaps_for_layer(hms, lm)

        decoded_hms[lm] = heatmap
        heatmap = cv2.blur(heatmap, (5, 5))
        lms[lm, :] = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[::-1]

        if show:
            ax_dec[lm//10, lm%10].imshow(heatmap)

    if show:
        plt.show()

    return decoded_hms, lms



def create_landmark_heatmaps(lms, sigma, landmarks_to_use):
    landmark_target = lmcfg.LANDMARK_TARGET
    lm_wnd_size = int(sigma * 5)
    lm_heatmap_window = make_landmark_template(lm_wnd_size, sigma)
    # lm_heatmap_window_outer = make_landmark_template(lm_wnd_size, sigma+2.0)

    nchannels = len(landmarks_to_use)
    if landmark_target == 'colored':
        nchannels = 3

    hms = np.zeros((nchannels, lmcfg.HEATMAP_SIZE, lmcfg.HEATMAP_SIZE))
    lm_scale = lmcfg.HEATMAP_SIZE / cfg.INPUT_SIZE
    lms_rescaled = lms * lm_scale
    for l in landmarks_to_use:
        wnd = lm_heatmap_window
        _fill_heatmap_layer(hms, lms_rescaled, l, wnd, lm_wnd_size)

    if landmark_target == 'single_channel':
        hms = hms.max(axis=0)
        hms /= hms.max()
    elif landmark_target == 'colored':
        # face_weights = face_weights.max(axis=0)
        hms = hms.clip(0,255)
        hms /= 255
    elif landmark_target == 'hamming':
        # hms = convert_to_hamming_encoded_heatmaps(hms)
        hms = convert_to_encoded_heatmaps(hms)
    return hms.astype(np.float32)


def calc_landmark_nme_per_img(gt_lms, pred_lms, ocular_norm='pupil', landmarks_to_eval=None):
    norm_dists = calc_landmark_nme(gt_lms, pred_lms, ocular_norm)
    # norm_dists = np.clip(norm_dists, a_min=None, a_max=20.0)
    if landmarks_to_eval is None:
        landmarks_to_eval = range(norm_dists.shape[1])
    return np.mean(np.array([norm_dists[:,l] for l in landmarks_to_eval]).T, axis=1)


def get_pupil_dists(gt):
    ocular_dists_inner = np.sqrt(np.sum((gt[:, 42] - gt[:, 39])**2, axis=1))
    ocular_dists_outer = np.sqrt(np.sum((gt[:, 45] - gt[:, 36])**2, axis=1))
    return np.vstack((ocular_dists_inner, ocular_dists_outer)).mean(axis=0)


def get_landmark_confs(X_lm_hm):
    return np.clip(to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2), a_min=0, a_max=1)


def calc_landmark_nme(gt_lms, pred_lms, ocular_norm='pupil'):
    def reformat(lms):
        lms = to_numpy(lms)
        if len(lms.shape) == 2:
            lms = lms.reshape((1,-1,2))
        return lms
    gt = reformat(gt_lms)
    pred = reformat(pred_lms)
    assert(len(gt.shape) == 3)
    assert(len(pred.shape) == 3)
    if gt.shape[1] == 19:
        # assert face_sizes is not None, "Face sizes a needed for error normalization on AFLW."
        ocular_dists = np.ones(gt.shape[0], dtype=np.float32) * cfg.INPUT_SIZE / 1.13
    elif gt.shape[1] == 98:
        ocular_dists = np.sqrt(np.sum((gt[:, 72] - gt[:, 60])**2, axis=1))
    else:
        if ocular_norm == 'pupil':
            ocular_dists = get_pupil_dists(gt)
        elif ocular_norm == 'outer':
            ocular_dists = np.sqrt(np.sum((gt[:, 45] - gt[:, 36])**2, axis=1))
        elif ocular_norm is None or ocular_norm == 'none':
            ocular_dists = np.ones((len(gt),1)) * 100.0 #* cfg.INPUT_SIZE
        else:
            raise ValueError("Ocular norm {} not defined!".format(ocular_norm))
    norm_dists = np.sqrt(np.sum((gt - pred)**2, axis=2)) / ocular_dists.reshape(len(gt), 1)
    return norm_dists * 100


def calc_landmark_failure_rate(nmes, th=10.0):
    img_nmes = nmes.mean(axis=1)
    assert len(img_nmes) == len(nmes)
    return np.count_nonzero(img_nmes > th) / len(img_nmes.ravel())


def calc_landmark_ncc(X, X_recon, lms):
    input_images = vis._to_disp_images(X, denorm=True)
    recon_images = vis._to_disp_images(X_recon, denorm=True)
    nimgs = len(input_images)
    nlms = len(lms[0])
    wnd_size = int(cfg.INPUT_SCALE_FACTOR * 15)
    nccs = np.zeros((nimgs, nlms), dtype=np.float32)
    img_shape = input_images[0].shape
    for i in range(nimgs):
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(img_shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(img_shape[1]-1, x+wnd_size//2)
            wnd1 = input_images[i][t:b, l:r]
            wnd2 = recon_images[i][t:b, l:r]
            ncc = ((wnd1-wnd1.mean()) * (wnd2-wnd2.mean())).mean() / (wnd1.std() * wnd2.std())
            nccs[i, lid] = ncc
    return np.clip(np.nan_to_num(nccs), a_min=-1, a_max=1)


def calc_landmark_ssim_score(X, X_recon, lms,
                             wnd_size=int(cfg.INPUT_SCALE_FACTOR * 16)):
    input_images = vis._to_disp_images(X, denorm=True)
    recon_images = vis._to_disp_images(X_recon, denorm=True)
    data_range = 255.0 if input_images[0].dtype == np.uint8 else 1.0
    nimgs = len(input_images)
    nlms = len(lms[0])
    scores = np.zeros((nimgs, nlms), dtype=np.float32)
    for i in range(nimgs):
        S = compare_ssim(input_images[i], recon_images[i], data_range=data_range, multichannel=True, full=True)[1]
        S = S.mean(axis=2)
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(S.shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(S.shape[1]-1, x+wnd_size//2)
            wnd = S[t:b, l:r]
            scores[i, lid] = wnd.mean()
    return np.nan_to_num(scores)


def calc_landmark_cs_error(X, X_recon, lms, torch_ssim, training=False,
                           wnd_size=int(cfg.INPUT_SCALE_FACTOR * 16)):
    nimgs = len(X)
    nlms = len(lms[0])
    errs = torch.zeros((nimgs, nlms), requires_grad=training).cuda()
    for i in range(len(X)):
        torch_ssim(X[i].unsqueeze(0), X_recon[i].unsqueeze(0))
        cs_map = torch_ssim.cs_map[0].mean(dim=0)
        map_size = cs_map.shape[0]
        margin = (cfg.INPUT_SIZE - map_size) // 2
        S = torch.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), requires_grad=training).cuda()
        S[margin:-margin, margin:-margin] = cs_map
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(S.shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(S.shape[1]-1, x+wnd_size//2)
            wnd = S[t:b, l:r]
            errs[i, lid] = 1 - wnd.mean()
    return errs


def calc_landmark_recon_error(X, X_recon, lms, return_maps=False, reduction='mean'):
    assert len(X.shape) == 4
    assert reduction in ['mean', 'none']
    X = to_numpy(X)
    X_recon = to_numpy(X_recon)
    mask = np.zeros((X.shape[0], X.shape[2], X.shape[3]), dtype=np.float32)
    radius = cfg.INPUT_SIZE * 0.05
    for img_id in range(len(mask)):
        for lm in lms[img_id]:
            cv2.circle(mask[img_id], (int(lm[0]), int(lm[1])), radius=int(radius), color=1, thickness=-1)
    err_maps = np.abs(X - X_recon).mean(axis=1) * 255.0
    masked_err_maps = err_maps * mask

    debug = False
    if debug:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(vis.to_disp_image((X * mask[:,np.newaxis,:,:].repeat(3, axis=1))[0], denorm=True))
        ax[1].imshow(vis.to_disp_image((X_recon * mask[:,np.newaxis,:,:].repeat(3, axis=1))[0], denorm=True))
        ax[2].imshow(masked_err_maps[0])
        plt.show()

    if reduction == 'mean':
        err = masked_err_maps.sum() / (mask.sum() * 3)
    else:
        # err = masked_err_maps.mean(axis=2).mean(axis=1)
        err = masked_err_maps.sum(axis=2).sum(axis=1) / (mask.reshape(len(mask), -1).sum(axis=1) * 3)

    if return_maps:
        return err, masked_err_maps
    else:
        return err


def to_single_channel_heatmap(lm_heatmaps):
    if lmcfg.LANDMARK_TARGET == 'colored':
        mc = [to_image(lm_heatmaps[0])]
    elif lmcfg.LANDMARK_TARGET == 'single_channel':
        mc = [to_image(lm_heatmaps[0, 0])]
    else:
        mc = to_image(lm_heatmaps.max(axis=1))
    return mc


#
# Visualizations
#

def show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1.0):

    vmax = 1.0
    rows_heatmaps = []
    if gt_heatmaps is not None:
        vmax = gt_heatmaps.max()
        if len(gt_heatmaps[0].shape) == 2:
            gt_heatmaps = [vis.color_map(hm, vmin=0, vmax=vmax, cmap=plt.cm.jet) for hm in gt_heatmaps]
        nCols = 1 if len(gt_heatmaps) == 1 else nimgs
        rows_heatmaps.append(cv2.resize(vis.make_grid(gt_heatmaps, nCols=nCols, padval=0), None, fx=f, fy=f))

    disp_pred_heatmaps = pred_heatmaps
    if len(pred_heatmaps[0].shape) == 2:
        disp_pred_heatmaps = [vis.color_map(hm, vmin=0, vmax=vmax, cmap=plt.cm.jet) for hm in pred_heatmaps]
    nCols = 1 if len(pred_heatmaps) == 1 else nimgs
    rows_heatmaps.append(cv2.resize(vis.make_grid(disp_pred_heatmaps, nCols=nCols, padval=0), None, fx=f, fy=f))

    cv2.imshow('Landmark heatmaps', cv2.cvtColor(np.vstack(rows_heatmaps), cv2.COLOR_RGB2BGR))


def visualize_batch(images, landmarks, X_recon, X_lm_hm, lm_preds_max,
                    lm_heatmaps=None, images_mod=None, lm_preds_cnn=None, ds=None, wait=0, ssim_maps=None,
                    landmarks_to_draw=lmcfg.ALL_LANDMARKS, ocular_norm='outer', horizontal=False, f=1.0,
                    overlay_heatmaps_input=False,
                    overlay_heatmaps_recon=False,
                    clean=False):

    gt_color = (0,255,0)
    pred_color = (0,0,255)

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    nme_per_lm = None
    if landmarks is None:
        # print('num landmarks', lmcfg.NUM_LANDMARKS)
        lm_gt = np.zeros((nimgs, lmcfg.NUM_LANDMARKS, 2))
    else:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs]
        nme_per_lm = calc_landmark_nme(lm_gt, lm_preds_max[:nimgs], ocular_norm=ocular_norm)
        lm_ssim_errs = 1 - calc_landmark_ssim_score(images, X_recon[:nimgs], lm_gt)

    lm_confs = None
    # show landmark heatmaps
    pred_heatmaps = None
    if X_lm_hm is not None:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
        gt_heatmaps = None
        if lm_heatmaps is not None:
            gt_heatmaps = to_single_channel_heatmap(to_numpy(lm_heatmaps[:nimgs]))
            gt_heatmaps = np.array([cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in gt_heatmaps])
        show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1)
        lm_confs = to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2)

    # resize images for display and scale landmarks accordingly
    lm_preds_max = lm_preds_max[:nimgs] * f
    if lm_preds_cnn is not None:
        lm_preds_cnn = lm_preds_cnn[:nimgs] * f
    lm_gt *= f

    input_images = vis._to_disp_images(images[:nimgs], denorm=True)
    if images_mod is not None:
        disp_images = vis._to_disp_images(images_mod[:nimgs], denorm=True)
    else:
        disp_images = vis._to_disp_images(images[:nimgs], denorm=True)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]

    recon_images = vis._to_disp_images(X_recon[:nimgs], denorm=True)
    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]

    # overlay landmarks on input images
    if pred_heatmaps is not None and overlay_heatmaps_input:
        disp_images = [vis.overlay_heatmap(disp_images[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]
    if pred_heatmaps is not None and overlay_heatmaps_recon:
        disp_X_recon = [vis.overlay_heatmap(disp_X_recon[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]


    #
    # Show input images
    #
    disp_images = vis.add_landmarks_to_images(disp_images, lm_gt[:nimgs], color=gt_color)
    disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_max[:nimgs], lm_errs=nme_per_lm,
                                              color=pred_color, draw_wireframe=False, gt_landmarks=lm_gt,
                                              draw_gt_offsets=True)

    # disp_images = vis.add_landmarks_to_images(disp_images, lm_gt[:nimgs], color=(1,1,1), radius=1,
    #                                           draw_dots=True, draw_wireframe=True, landmarks_to_draw=landmarks_to_draw)
    # disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_max[:nimgs], lm_errs=nme_per_lm,
    #                                           color=(1.0, 0.0, 0.0),
    #                                           draw_dots=True, draw_wireframe=True, radius=1,
    #                                           gt_landmarks=lm_gt, draw_gt_offsets=False,
    #                                           landmarks_to_draw=landmarks_to_draw)


    #
    # Show reconstructions
    #
    X_recon_errs = 255.0 * torch.abs(images - X_recon[:nimgs]).reshape(len(images), -1).mean(dim=1)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon[:nimgs], errors=X_recon_errs, format_string='{:>4.1f}')

    # modes of heatmaps
    # disp_X_recon = [overlay_heatmap(disp_X_recon[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]
    if not clean:
        lm_errs_max = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.LANDMARKS_NO_OUTLINE)
        lm_errs_max_outline = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.LANDMARKS_ONLY_OUTLINE)
        lm_errs_max_all = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.ALL_LANDMARKS)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max, loc='br-2', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_outline, loc='br-1', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_all, loc='br', format_string='{:>5.2f}', vmax=15)
    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_gt, color=gt_color, draw_wireframe=True)

    # disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_max[:nimgs],
    #                                            color=pred_color, draw_wireframe=False,
    #                                            lm_errs=nme_per_lm, lm_confs=lm_confs,
    #                                            lm_rec_errs=lm_ssim_errs, gt_landmarks=lm_gt,
    #                                            draw_gt_offsets=True, draw_dots=True)

    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_max[:nimgs],
                                               color=pred_color, draw_wireframe=True,
                                               gt_landmarks=lm_gt, draw_gt_offsets=True, lm_errs=nme_per_lm,
                                               draw_dots=True, radius=2)

    def add_confs(disp_X_recon, lmids, loc):
        means = lm_confs[:,lmids].mean(axis=1)
        colors = vis.color_map(to_numpy(1-means), cmap=plt.cm.jet, vmin=0.0, vmax=0.4)
        return vis.add_error_to_images(disp_X_recon, means, loc=loc, format_string='{:>4.2f}', colors=colors)

    # disp_X_recon = add_confs(disp_X_recon, lmcfg.LANDMARKS_NO_OUTLINE, 'bm-2')
    # disp_X_recon = add_confs(disp_X_recon, lmcfg.LANDMARKS_ONLY_OUTLINE, 'bm-1')
    # disp_X_recon = add_confs(disp_X_recon, lmcfg.ALL_LANDMARKS, 'bm')

    # print ssim errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, multichannel=True)
    if not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                               vmax=0.8, vmin=0.2)
    # print ssim torch errors
    if ssim_maps is not None and not clean:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, ssim_maps.reshape(len(ssim_maps), -1).mean(axis=1),
                                               loc='bl-2', format_string='{:>4.2f}', vmin=0.0, vmax=0.4)

    rows = [vis.make_grid(disp_images, nCols=nimgs, normalize=False)]
    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    if ssim_maps is not None:
        disp_ssim_maps = to_numpy(ds_utils.denormalized(ssim_maps)[:nimgs].transpose(0, 2, 3, 1))
        for i in range(len(disp_ssim_maps)):
            disp_ssim_maps[i] = vis.color_map(disp_ssim_maps[i].mean(axis=2), vmin=0.0, vmax=2.0)
        grid_ssim_maps = vis.make_grid(disp_ssim_maps, nCols=nimgs, fx=f, fy=f)
        cv2.imshow('ssim errors', cv2.cvtColor(grid_ssim_maps, cv2.COLOR_RGB2BGR))

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=2)
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'Predicted Landmarks '
    if ds is not None:
        wnd_title += ds.__class__.__name__
    cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)


def visualize_random_faces(net, nimgs=10, wait=10, f=1.0):
    z_random = torch.randn(nimgs, net.z_dim).cuda()
    with torch.no_grad():
        X_gen_vis = net.P(z_random)[:, :3]
        X_lm_hm = net.LMH(net.P)
    pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
    pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
    disp_X_gen = to_numpy(ds_utils.denormalized(X_gen_vis).permute(0, 2, 3, 1))
    disp_X_gen = (disp_X_gen * 255).astype(np.uint8)
    # disp_X_gen = [vis.overlay_heatmap(disp_X_gen[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]

    grid_img = vis.make_grid(disp_X_gen, nCols=nimgs//2)
    cv2.imshow("random faces", cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)

# LM68_TO_LM96 = 1
LM98_TO_LM68 = 2

def convert_landmarks(lms, code):
    cvt_func = {
        LM98_TO_LM68: lm98_to_lm68,
    }
    if len(lms.shape) == 3:
        new_lms = []
        for i in range(len(lms)):
            new_lms.append(cvt_func[code](lms[i]))
        return np.array(new_lms)
    elif len(lms.shape) == 2:
        return cvt_func[code](lms)
    else:
        raise ValueError


def lm98_to_lm68(lm98):
    def copy_lms(offset68, offset98, n):
        lm68[range(offset68, offset68+n)] = lm98[range(offset98, offset98+n)]

    assert len(lm98), "Cannot convert landmarks to 68 points!"
    lm68 = np.zeros((68,2), dtype=np.float32)

    # outline
    # for i in range(17):
    lm68[range(17)] = lm98[range(0,33,2)]

    # left eyebrow
    copy_lms(17, 33, 5)
    # right eyebrow
    copy_lms(22, 42, 5)
    # nose
    copy_lms(27, 51, 9)

    # eye left
    lm68[36] = lm98[60]
    lm68[37] = lm98[61]
    lm68[38] = lm98[63]
    lm68[39] = lm98[64]
    lm68[40] = lm98[65]
    lm68[41] = lm98[67]

    # eye right
    lm68[36+6] = lm98[60+8]
    lm68[37+6] = lm98[61+8]
    lm68[38+6] = lm98[63+8]
    lm68[39+6] = lm98[64+8]
    lm68[40+6] = lm98[65+8]
    lm68[41+6] = lm98[67+8]

    copy_lms(48, 76, 20)

    return lm68


def is_good_landmark(confs, rec_errs=None):
    if rec_errs is not None:
        low_errors = rec_errs < 0.25
        # if isinstance(low_error):
        #     low_erres
        confs *= np.array(low_errors).astype(int)
    return confs > 0.8



def visualize_batch_CVPR(images, landmarks, X_recon, X_lm_hm, lm_preds,
                         lm_heatmaps=None, ds=None, wait=0, horizontal=False, f=1.0, radius=2):

    gt_color = (0,255,0)
    pred_color = (0,255,255)

    nimgs = min(10, len(images))
    images = nn.atleast4d(images)[:nimgs]
    if landmarks is None:
        print('num landmarks', lmcfg.NUM_LANDMARKS)
        lm_gt = np.zeros((nimgs, lmcfg.NUM_LANDMARKS, 2))
    else:
        lm_gt = nn.atleast3d(to_numpy(landmarks))[:nimgs]

    # show landmark heatmaps
    pred_heatmaps = None
    if X_lm_hm is not None:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
        gt_heatmaps = None
        if lm_heatmaps is not None:
            gt_heatmaps = to_single_channel_heatmap(to_numpy(lm_heatmaps[:nimgs]))
            gt_heatmaps = np.array([cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in gt_heatmaps])
        show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1)
        lm_confs = to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2)

    # resize images for display and scale landmarks accordingly
    lm_preds = lm_preds[:nimgs] * f
    lm_gt *= f

    rows = []

    disp_images = vis._to_disp_images(images[:nimgs], denorm=True)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]
    rows.append(vis.make_grid(disp_images, nCols=nimgs, normalize=False))

    recon_images = vis._to_disp_images(X_recon[:nimgs], denorm=True)
    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    # recon_images = vis._to_disp_images(X_recon[:nimgs], denorm=True)
    disp_X_recon_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_pred = vis.add_landmarks_to_images(disp_X_recon_pred, lm_preds, color=pred_color,radius=radius)
    rows.append(vis.make_grid(disp_X_recon_pred, nCols=nimgs))

    disp_X_recon_gt = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_gt = vis.add_landmarks_to_images(disp_X_recon_gt, lm_gt, color=gt_color, radius=radius)
    rows.append(vis.make_grid(disp_X_recon_gt, nCols=nimgs))

    # overlay landmarks on images
    disp_X_recon_hm = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]
    disp_X_recon_hm = [vis.overlay_heatmap(disp_X_recon_hm[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]
    rows.append(vis.make_grid(disp_X_recon_hm, nCols=nimgs))

    # input images with prediction (and ground truth)
    disp_images_pred = vis._to_disp_images(images[:nimgs], denorm=True)
    disp_images_pred = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images_pred]
    # disp_images_pred = vis.add_landmarks_to_images(disp_images_pred, lm_gt, color=gt_color, radius=radius)
    disp_images_pred = vis.add_landmarks_to_images(disp_images_pred, lm_preds, color=pred_color, radius=radius)
    rows.append(vis.make_grid(disp_images_pred, nCols=nimgs))

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=2)
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'recon errors '
    if ds is not None:
        wnd_title += ds.__class__.__name__
    cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)


def smooth_heatmaps(hms):
    # assert isinstance(hms, np.ndarray)
    assert(len(hms.shape) == 4)
    hms = to_numpy(hms)
    for i in range(hms.shape[0]):
        for l in range(hms.shape[1]):
            hms[i,l] = cv2.blur(hms[i,l], (9,9), borderType=cv2.BORDER_CONSTANT)
            # hms[i,l] = cv2.GaussianBlur(hms[i,l], (9,9), sigmaX=9, borderType=cv2.BORDER_CONSTANT)
    return hms
