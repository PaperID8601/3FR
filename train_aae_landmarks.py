import time

import landmarks.lmconfig as lmcfg
import datetime
import cv2
import os
import pandas as pd

import numpy as np

import torch
import torch.utils.data as td
import torch.nn.modules.distance
import torch.optim as optim
import torch.nn.functional as F
from datasets import affectnet, vggface2, multi, w300, wflw, aflw
from constants import TRAIN, VAL
from utils import log
import config as cfg

from utils.nn import to_numpy, Batch
from train_aae_unsupervised import AAETraining
from landmarks import lmutils


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

eps = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASETS = {
    'affectnet': affectnet.AffectNet,
    'multi': multi.MultiFaceDataset,
    'vggface2': vggface2.VggFace2,
    '300w': w300.W300,
    'aflw': aflw.AFLW,
    'wflw': wflw.WFLW
}


def minmax_scale(a):
    return (a - a.min()) / (a.max() - a.min())


def _avg(list_, key):
    return np.mean([d[key] for d in list_])


class AAELandmarkTraining(AAETraining):

    def __init__(self, datasets, args, session_name='debug', **kwargs):
        args.reset = False
        super().__init__(datasets, args, session_name, train_autoencoder=True, **kwargs)
        self.optimizer_lm_head = optim.Adam(self.saae.LMH.parameters(), lr=args.lr_heatmaps, betas=(0.9,0.999))
        self.optimizer_Q = optim.Adam(self.saae.Q.parameters(), lr=0.00002, betas=(0.9, 0.999))
        self.optimizer_P = optim.Adam(self.saae.P.parameters(), lr=0.00002, betas=(0.9, 0.999))

    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats).mean().to_dict()
        current = stats[-1]

        str_stats = ['[{ep}][({i}/{iters_per_epoch}] '
                     'l_Q={avg_loss_Q:.3f}  '
                     'l_rec={avg_loss_recon:.3f} '
                     'ssim={avg_ssim:.3f} '
                     # 'ssim_torch={avg_ssim_torch:.3f} '
                     # 'l_act={avg_loss_activations:.3f} '
                     # 'z_mu={avg_z_recon_mean: .3f} '
                     'l_lms={avg_loss_lms:.4f} '
                     'err_lms={avg_err_lms_max:.2f}/{avg_err_lms_max_outline:.2f}/{avg_err_lms_max_all:.2f} '
                     # 'l_cnn={avg_loss_lms_cnn:.4f} '
                     # 'err_cnn={avg_err_lms_cnn:.2f}/{avg_err_lms_cnn_outline:.2f}/{avg_err_lms_cnn_all:.2f} '
                     # 'l_D_z={avg_loss_D_z:.3f} '
                     # 'l_E={avg_loss_E:.3f} '
                     # 'l_D={avg_loss_D:.3f} '
                     # 'l_G={avg_loss_G:.3f} '
                     '{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {total_time})'][0]
        log.info(str_stats.format(
            ep=current['epoch'] + 1, i=current['iter'] + 1, iters_per_epoch=self.iters_per_epoch,
            avg_loss_Q=means.get('loss_Q', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_ssim=1.0 - means.get('ssim', -1),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_activations=means.get('loss_activations', -1),
            avg_loss_lms=means.get('loss_lms', -1),
            avg_loss_lms_cnn=means.get('loss_lms_cnn', -1),
            avg_loss_bump=means.get('loss_bump', -1),
            avg_loss_E=means.get('loss_E', -1),
            avg_loss_D_z=means.get('loss_D_z', -1),
            avg_loss_D=means.get('loss_D', -1),
            avg_loss_G=means.get('loss_G', -1),
            avg_loss_D_real=means.get('err_real', -1),
            avg_loss_D_fake=means.get('err_fake', -1),
            avg_z_l1=means.get('z_l1', -1),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            avg_err_lms_max = means.get('lm_errs_max', np.zeros(1)).mean(),
            avg_err_lms_max_outline = means.get('lm_errs_max_outline', np.zeros(1)).mean(),
            avg_err_lms_max_all = means.get('lm_errs_max_all', np.zeros(1)).mean(),
            avg_err_lms_cnn = means.get('lm_errs_cnn', np.zeros(1)).mean(),
            avg_err_lms_cnn_outline = means.get('lm_errs_cnn_outline', np.zeros(1)).mean(),
            avg_err_lms_cnn_all = means.get('lm_errs_cnn_all', np.zeros(1)).mean(),
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time()))
        ))

    def print_eval_metrics(self, nmes, show=False):
        def ced_curve(nmes):
            Y = []
            X = np.linspace(0, 10, 50)
            for th in X:
                recall = 1.0 - lmutils.calc_landmark_failure_rate(nmes, th)
                recall *= 1/len(X)
                Y.append(recall)
            return X,Y

        def auc(recalls):
            return np.sum(recalls)

        # for err_scale in np.linspace(0.1, 1, 10):
        for err_scale in [1.0]:
            # print('\nerr_scale', err_scale)
            # print(np.clip(lm_errs_max_all, a_min=0, a_max=10).mean())

            fr = lmutils.calc_landmark_failure_rate(nmes*err_scale)
            X, Y = ced_curve(nmes)

            log.info('NME:   {:>6.3f}'.format(nmes.mean()*err_scale))
            log.info('FR@10: {:>6.3f} ({})'.format(fr*100, np.sum(nmes.mean(axis=1) > 10)))
            log.info('AUC:   {:>6.4f}'.format(auc(Y)))

            if show:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1,2)
                axes[0].plot(X, Y)
                print(nmes.mean(axis=1).shape)
                print(nmes.mean(axis=1).max())
                axes[1].hist(nmes.mean(axis=1), bins=20)
                plt.show()


    def _print_epoch_summary(self, epoch_stats, epoch_starttime):
        means = pd.DataFrame(epoch_stats).mean().to_dict()

        try:
            nmes = np.concatenate([s.get('nmes', np.zeros(0)) for s in self.epoch_stats])
            # nmes_cnn = np.concatenate([s.get('nmes_cnn', np.zeros(0)) for s in self.epoch_stats])
        except:
            nmes = None
        try:
            nccs = np.concatenate([s.get('nccs', np.zeros(0)) for s in self.epoch_stats])
        except:
            nncs = None

        # lm_errs_max = nmes.mean(axis=1)
        lm_errs_max = np.concatenate([stats.get('lm_errs_max', np.zeros(0)) for stats in self.epoch_stats])
        lm_errs_max_outline = np.concatenate([stats.get('lm_errs_max_outline', np.zeros(0)) for stats in self.epoch_stats])
        lm_errs_max_all = np.concatenate([stats.get('lm_errs_max_all', np.zeros(0)) for stats in self.epoch_stats])

        lm_errs_cnn = np.concatenate([stats.get('lm_errs_cnn', np.zeros(0)) for stats in self.epoch_stats])
        lm_errs_cnn_outline = np.concatenate([stats.get('lm_errs_cnn_outline', np.zeros(0)) for stats in self.epoch_stats])
        lm_errs_cnn_all = np.concatenate([stats.get('lm_errs_cnn_all', np.zeros(0)) for stats in self.epoch_stats])


        duration = int(time.time() - epoch_starttime)
        log.info("{}".format('-' * 120))
        str_stats = ['Train:       '
                     'l_Q={avg_loss_Q:.3f} '
                     'l_rec={avg_loss_recon:.3f} '
                     'ssim={avg_ssim:.3f} '
                     # 'ssim_torch={avg_ssim_torch:.3f} '
                     # 'z_mu={avg_z_recon_mean:.3f} '
                     'l_lms={avg_loss_lms:.4f} '
                     'err_lms={avg_err_lms_max:.3f}/{avg_err_lms_max_outline:.3f}/{avg_err_lms_max_all:.3f} '
                     # 'l_cnn={avg_loss_lms_cnn:.4f} '
                     # 'err_cnn={avg_err_lms_cnn:.3f}/{avg_err_lms_cnn_outline:.3f}/{avg_err_lms_cnn_all:.3f} '
                     # 'l_D={avg_loss_D:.4f} '
                     # 'l_G={avg_loss_G:.4f} '
                     '\tT: {time_epoch}'][0]
        log.info(str_stats.format(
            iters_per_epoch=self.iters_per_epoch,
            avg_loss_Q=means.get('loss_Q', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_ssim=1.0 - means.get('ssim', -1),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_lms=means.get('loss_lms', -1),
            avg_loss_lms_cnn=means.get('loss_lms_cnn', -1),
            # avg_err_lms_max = lm_errs_max.mean(),
            # avg_err_lms_max_outline = lm_errs_max_outline.mean(),
            # avg_err_lms_max_all = lm_errs_max_all.mean(),
            avg_err_lms_max = np.mean(lm_errs_max),
            avg_err_lms_max_outline = np.mean(lm_errs_max_outline),
            avg_err_lms_max_all = np.mean(lm_errs_max_all),
            avg_err_lms_cnn = lm_errs_cnn.mean(),
            avg_err_lms_cnn_outline = lm_errs_cnn_outline.mean(),
            avg_err_lms_cnn_all = lm_errs_cnn_all.mean(),
            avg_loss_E=means.get('loss_E', -1),
            avg_loss_D_z=means.get('loss_D_z', -1),
            avg_loss_D=means.get('loss_D', -1),
            avg_loss_G=means.get('loss_G', -1),
            avg_loss_D_real=means.get('err_real', -1),
            avg_loss_D_fake=means.get('err_fake', -1),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time())),
            time_epoch=str(datetime.timedelta(seconds=duration))))
        try:
            recon_errors = np.concatenate([stats['l1_recon_errors'] for stats in self.epoch_stats])
            rmse = np.sqrt(np.mean(recon_errors ** 2))
            log.info("RMSE: {} ".format(rmse))
        except:
            print("no l1_recon_error")

        if self.args.eval and nmes is not None:
            self.print_eval_metrics(nmes, show=self.args.benchmark)
            max_nme = 10
            lm_errs_max_all = np.clip(lm_errs_max_all, a_min=0, a_max=max_nme)
            print("Clip to {} NME: {}".format(max_nme, lm_errs_max_all.mean()))


    def eval_epoch(self):
        log.info("")
        log.info("Starting evaluation of '{}'...".format(self.session_name))
        log.info("")

        self.training = False
        self.time_start_eval = time.time()
        epoch_starttime = time.time()
        self.epoch_stats = []
        self.saae.eval()

        self._run_epoch(self.datasets[VAL], train_autoencoder=False, eval=True)
        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_starttime)
        return self.epoch_stats

    def train(self, num_epochs):

        log.info("")
        log.info("Starting training session '{}'...".format(self.session_name))
        log.info("")

        while self.epoch < num_epochs:
            log.info('')
            log.info('Epoch {}/{}'.format(self.epoch + 1, num_epochs))
            log.info('=' * 10)

            self.training = True
            self.epoch_stats = []
            epoch_starttime = time.time()
            self.saae.train(self.train_autoencoder)

            self._run_epoch(self.datasets[TRAIN], self.train_autoencoder)

            # save model every few epochs
            if (self.epoch + 1) % self.snapshot_interval == 0:
                log.info("*** saving snapshot *** ")
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_starttime)

            if self._is_eval_epoch():
                self.eval_epoch()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def _run_epoch(self, dataset, train_autoencoder, eval=False):
        batchsize = self.args.batchsize_eval if eval else self.batch_size

        self.iters_per_epoch = int(len(dataset) / batchsize)
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0
        dataloader = td.DataLoader(dataset, batch_size=batchsize, shuffle=not eval, num_workers=self.workers,
                                   drop_last=not eval)

        for data in dataloader:
            self._run_batch(data, train_autoencoder, eval=eval)
            self.total_iter += 1
            self.saae.total_iter = self.total_iter
            self.iter_in_epoch += 1


    def _run_batch(self, data, train_autoencoder, eval=False, ds=None):

        time_dataloading = time.time() - self.iter_starttime
        time_proc_start = time.time()
        iter_stats = {'time_dataloading': time_dataloading}

        batch = Batch(data, eval=eval)

        self.saae.zero_grad()
        self.saae.eval()

        input_images = batch.images_mod if batch.images_mod is not None else batch.images

        with torch.set_grad_enabled(self.args.train_encoder):
            z_sample = self.saae.Q(input_images)

        iter_stats.update({'z_recon_mean': z_sample.mean().item()})

        #######################
        # Reconstruction phase
        #######################
        with torch.set_grad_enabled(self.args.train_encoder and not eval):
            X_recon = self.saae.P(z_sample)

        with torch.no_grad():
            diff = torch.abs(batch.images - X_recon) * 255
            loss_recon_l1 = torch.mean(diff)
            loss_Q = loss_recon_l1 * cfg.W_RECON
            iter_stats['loss_recon'] =  loss_recon_l1.item()
            l1_dist_per_img = diff.reshape(len(batch.images), -1).mean(dim=1)
            iter_stats['l1_recon_errors'] = to_numpy(l1_dist_per_img)

        #######################
        # Landmark predictions
        #######################
        train_lmhead = not eval and not args.train_coords
        lm_preds_max = None
        with torch.set_grad_enabled(train_lmhead):
            self.saae.LMH.train(train_lmhead)
            X_lm_hm = self.saae.LMH(self.saae.P)
            if batch.lm_heatmaps is not None:
                loss_lms = F.mse_loss(batch.lm_heatmaps, X_lm_hm) * 100 * 3

            if (eval or self._is_printout_iter()):
                # expensive, so only calculate when every N iterations
                X_lm_hm = lmutils.decode_heatmap_blob(X_lm_hm)
                X_lm_hm = lmutils.smooth_heatmaps(X_lm_hm)
                lm_preds_max = self.saae.heatmaps_to_landmarks(X_lm_hm)

            iter_stats.update({'loss_Q': loss_Q.item()})
            if not args.train_coords:
                iter_stats.update({'loss_lms': loss_lms.item()})

            if (eval or self._is_printout_iter()):
                lm_gt = to_numpy(batch.landmarks)
                lm_errs_max = lmutils.calc_landmark_nme_per_img(lm_gt,
                                                                lm_preds_max,
                                                                ocular_norm=self.args.ocular_norm,
                                                                landmarks_to_eval=lmcfg.LANDMARKS_NO_OUTLINE)
                lm_errs_max_outline = lmutils.calc_landmark_nme_per_img(lm_gt,
                                                                        lm_preds_max,
                                                                        ocular_norm=self.args.ocular_norm,
                                                                        landmarks_to_eval=lmcfg.LANDMARKS_ONLY_OUTLINE)
                lm_errs_max_all = lmutils.calc_landmark_nme_per_img(lm_gt,
                                                                    lm_preds_max,
                                                                    ocular_norm=self.args.ocular_norm,
                                                                    landmarks_to_eval=lmcfg.ALL_LANDMARKS)

                nmes = lmutils.calc_landmark_nme(lm_gt, lm_preds_max, ocular_norm=self.args.ocular_norm)
                # nccs = lmutils.calc_landmark_ncc(batch.images, X_recon, lm_gt)

                iter_stats.update({'lm_errs_max': lm_errs_max,
                                   'lm_errs_max_all': lm_errs_max_all,
                                   'lm_errs_max_outline': lm_errs_max_outline,
                                   'nmes': nmes,
                                   # 'nccs': nccs
                                   })

        if train_lmhead:
            if self.args.train_encoder:
                loss_lms = loss_lms * 80.0
            loss_lms.backward()
            self.optimizer_lm_head.step()
            if self.args.train_encoder:
                self.optimizer_Q.step()

        # statistics
        iter_stats.update({'epoch': self.epoch, 'timestamp': time.time(),
                           'iter_time': time.time() - self.iter_starttime,
                           'time_processing': time.time() - time_proc_start,
                           'iter': self.iter_in_epoch, 'total_iter': self.total_iter, 'batch_size': len(batch)})
        self.iter_starttime = time.time()

        self.epoch_stats.append(iter_stats)

        # print stats every N mini-batches
        if self._is_printout_iter():
            self._print_iter_stats(self.epoch_stats[-self.print_interval:])

        # Batch visualization
        #
        if self._is_printout_iter():
            f = 2.0 / cfg.INPUT_SCALE_FACTOR
            # lmutils.visualize_random_faces(self.saae, 20, 0)
            lmutils.visualize_batch(batch.images, batch.landmarks, X_recon, X_lm_hm, lm_preds_max,
                                    lm_heatmaps=batch.lm_heatmaps,
                                    images_mod=batch.images_mod,
                                    ds=ds, wait=self.wait,
                                    landmarks_to_draw=lmcfg.ALL_LANDMARKS,
                                    ocular_norm=args.ocular_norm,
                                    f=f,
                                    clean=False,
                                    overlay_heatmaps_input=False,
                                    overlay_heatmaps_recon=False)


def run(args):

    from utils.common import init_random

    if args.seed is not None:
        init_random(args.seed)
    # log.info(json.dumps(vars(args), indent=4))

    datasets = {}
    for phase, dsnames, num_samples in zip((TRAIN, VAL),
                                           (args.dataset_train, args.dataset_val),
                                           (args.train_count, args.val_count)):
        train = phase == TRAIN
        datasets[phase] = DATASETS[dsnames[0]](train=train,
                                               max_samples=num_samples,
                                               use_cache=args.use_cache,
                                               start=args.st,
                                               test_split=args.test_split,
                                               align_face_orientation=args.align,
                                               crop_source=args.crop_source,
                                               return_landmark_heatmaps=lmcfg.PREDICT_HEATMAP and not args.train_coords,
                                               # return_landmark_heatmaps=True,
                                               return_modified_images=args.mod and train,
                                               landmark_sigma=args.sigma,
                                               # landmark_ids=lmcfg.LANDMARKS,
                                               daug=args.daug)
        print(datasets[phase])

    fntr = AAELandmarkTraining(datasets, args, session_name=args.sessionname, batch_size=args.batchsize,
                       macro_batch_size=args.macro_batchsize,
                       snapshot_interval=args.save_freq, snapshot=args.resume, workers=args.workers,
                       wait=args.wait)

    torch.backends.cudnn.benchmark = True
    if args.eval:
        fntr.eval_epoch()
    else:
        fntr.train(num_epochs=args.epochs)


if __name__ == '__main__':

    import sys
    np.set_printoptions(linewidth=np.inf)

    # Disable traceback on Ctrl+c
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    import argparse

    bool_str = lambda x: (str(x).lower() in ['true', '1'])

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--sessionname', default=None, type=str, help='output filename (without ext)')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH', help='path to snapshot (default: None)')
    parser.add_argument('-z','--embedding-dims', default=cfg.EMBEDDING_DIMS, type=int, help='dimenionality of embedding ')

    # Training
    parser.add_argument('--macro-batchsize', default=20, type=int, metavar='N', help='macro batch size')
    parser.add_argument('-b', '--batchsize', default=50, type=int, metavar='N', help='batch size (default: 100)')
    parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print every N steps')
    parser.add_argument('--save-freq', default=1, type=int, metavar='N', help='save snapshot every N epochs')
    parser.add_argument('--tag', default=None, type=str, help='description')
    parser.add_argument('-e', '--epochs', default=10000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--finetuning', type=bool_str, default=False, help='finetune first generator layers')
    parser.add_argument('--train-encoder', type=bool_str, default=False, help='include encoder update in landmark training ')

    # Evaluation
    parser.add_argument('--eval', default=False, action='store_true', help='run evaluation instead of training')
    parser.add_argument('--val-count', default=800, type=int, help='number of test images')
    parser.add_argument('--batchsize-eval', default=50, type=int, metavar='N', help='batch size for evaluation')
    parser.add_argument('--print-freq-eval', default=1, type=int, metavar='N', help='print every N steps')
    parser.add_argument('--eval-freq', default=1, type=int, metavar='N', help='evaluate every N steps')
    parser.add_argument('--benchmark', default=False, action='store_true', help='run performance evaluation on test set')

    # Dataset
    parser.add_argument('--dataset', default=['wflw'], type=str, help='dataset for training and testing', choices=DATASETS, nargs='+')
    # parser.add_argument('--dataset-train', default=['wflw'], type=str, help='dataset for training.', choices=DATASETS, nargs='+')
    # parser.add_argument('--dataset-val', default=['wflw'], type=str, help='dataset for training.', choices=DATASETS, nargs='+')
    parser.add_argument('--test-split', default='full', type=str, help='test set split for 300W/AFLW/WFLW',
                        choices=['challenging', 'common', '300w', 'full', 'frontal']+wflw.subsets)
    parser.add_argument('--train-count', default=10000000, type=int, help='number of training images')
    parser.add_argument('--st', default=None, type=int, help='skip first n training images')
    parser.add_argument('--use-cache', type=bool_str, default=True, help='use cached crops')
    parser.add_argument('--crop-source', type=str, default='bb_detector', choices=w300.W300.CROP_SOURCES)
    parser.add_argument('--align', type=bool_str, default=cfg.CROP_ALIGN_ROTATION, help='rotate crop so eyes are horizontal')
    parser.add_argument('--daug', type=int, default=0, help='state of data augmentation for training')
    parser.add_argument('--mod', type=bool_str, default=False, help='create modified copies of input images')

    # Landmarks
    parser.add_argument('--lr-heatmaps', default=0.001, type=float, help='learning rate for landmark heatmap outputs')
    parser.add_argument('--sigma', default=7, type=float, help='size of landmarks in heatmap')
    parser.add_argument('-n', '--ocular-norm', default=lmcfg.LANDMARK_OCULAR_NORM, type=str, help='how to normalize landmark errors', choices=['pupil', 'outer', 'none'])
    # parser.add_argument('--lm-ids-eval', default=lmcfg.LANDMARKS_TO_EVALUATE, type=int, help='which landmark IDs to evaluate')

    parser.add_argument('--train-coords', type=bool_str, default=False, help='train landmark coordinate regression')
    parser.add_argument('--lr-lm-coords', default=0.001, type=float, help='learning rate for landmark coordinate regression')

    # visualization
    parser.add_argument('--show-random-faces', default=False, action='store_true')
    parser.add_argument('--wait', default=10, type=int)

    args = parser.parse_args()

    args.color = True

    args.dataset_train = args.dataset
    args.dataset_val = args.dataset

    lmcfg.config_landmarks(args.dataset[0])

    if args.eval:
        log.info('Switching to evaluation mode...')
        args.train_ae = False
        args.batchsize_eval = 10
        args.wait = 0
        args.workers = 0
        args.print_freq_eval = 1
        args.epochs = 1

    if args.benchmark:
        log.info('Switching to benchmark mode...')
        args.eval = True
        args.train_ae = False
        args.batchsize_eval = 50
        args.wait = 10
        args.workers = 4
        args.print_freq_eval = 20
        args.epochs = 1
        args.val_count = None

    if args.sessionname is None:
        if args.resume:
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = 'debug'

    lmcfg.LANDMARK_SIGMA = args.sigma
    cfg.WITH_LANDMARK_LOSS = True
    cfg.EMBEDDING_DIMS = args.embedding_dims

    run(args)
