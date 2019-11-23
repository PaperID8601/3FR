import time

import utils
import datetime
import cv2
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.utils.data as td
import torch.nn.modules.distance
import torch.optim as optim
import torch.nn.functional as F
import utils.nn

from datasets import ds_utils
from datasets import multi, affectnet, vggface2, w300, aflw, wflw
from constants import TRAIN, VAL
import config as cfg
from utils import vis
from networks import saae
from networks.saae import vis_reconstruction
from utils.nn import to_numpy, Batch, set_requires_grad
from skimage.measure import compare_ssim
from metrics import ssim as pytorch_msssim
import landmarks.lmconfig as lmcfg
from utils.common import init_random
import landmarks.lmutils as lmutils
import utils.log as log

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
    'wflw': wflw.WFLW,
}


def weights_init(m):
    from torch import nn as nn
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)


def visualize_middle_activations(xa, xb):
    fig, ax = plt.subplots(1,2)
    imga = vis.make_grid(to_numpy(xa[0, :36]), nCols=6)
    imgb = vis.make_grid(to_numpy(xb[0, :36]), nCols=6)
    ax[0].imshow(imga)
    ax[1].imshow(imgb)
    plt.show()


class AAETraining(object):

    def __init__(self, datasets, args, session_name='debug', snapshot_dir=cfg.SNAPSHOT_DIR,
                 lr=0.00002, batch_size=100, snapshot=None, snapshot_interval=5,
                 workers=6, macro_batch_size=20, state_disentanglement=0,
                 train_autoencoder=True, wait=10):

        self.training = True
        self.args = args
        self.session_name = session_name
        self.datasets = datasets
        self.macro_batch_size = macro_batch_size
        self.batch_size = batch_size
        self.workers = workers
        self.ssim = pytorch_msssim.SSIM(window_size=31)
        load_resnet_pretrained = snapshot is None
        self.saae = saae.SAAE(load_resnet_pretrained)
        self.wait = wait

        # Set learning rates
        print("Learning rate: {}".format(lr))

        # Set optimizators
        Q_params = list(filter(lambda p: p.requires_grad, self.saae.Q.parameters()))
        self.optimizer_Q = optim.Adam(Q_params, lr=lr, betas=(0.0, 0.999))
        self.optimizer_P = optim.Adam(self.saae.P.parameters(), lr=lr, betas=(0.0, 0.999))

        betas = (0.0, 0.999)
        self.optimizer_D_z = optim.Adam(self.saae.D_z.parameters(), lr=lr, betas=betas)
        self.optimizer_D = optim.Adam(self.saae.D.parameters(), lr=lr*0.5, betas=betas)

        self.BCE_stable = torch.nn.BCEWithLogitsLoss().cuda()

        self.snapshot_dir = snapshot_dir
        self.total_iter = 0
        self.epoch = 0
        self.best_score = 999

        self.snapshot_interval = snapshot_interval

        self.state_disentanglement = state_disentanglement
        self.train_autoencoder = train_autoencoder

        if cfg.ENCODING_DISTRIBUTION == 'normal':
            self.enc_rand = torch.randn
            self.enc_rand_like = torch.randn_like
        elif cfg.ENCODING_DISTRIBUTION == 'uniform':
            self.enc_rand = torch.rand
            self.enc_rand_like = torch.rand_like
        else:
            raise ValueError()

        self.total_training_time_previous = 0
        self.time_start_training = time.time()

        if snapshot is not None:
            log.info("Resuming session {} from snapshot {}...".format(self.session_name, snapshot))
            self._load_snapshot(snapshot)

        # reset discriminator
        if args.reset:
            self.saae.D.apply(weights_init)

        # save some samples to visualize the training progress
        self.n_fixed = 20
        dl = td.DataLoader(datasets[VAL], batch_size=self.n_fixed, shuffle=False, num_workers=1)
        data_val = next(iter(dl))
        self.fixed_batch_train = None
        self.fixed_batch_val = Batch(data_val, n=self.n_fixed)

    def get_sample_weights(self, dataset):
        bbox_aspect_ratios = dataset.widths / dataset.heights
        print('Num. profile images: ', np.count_nonzero(bbox_aspect_ratios < 0.65))
        _weights = np.ones_like(bbox_aspect_ratios, dtype=np.float32)
        _weights[bbox_aspect_ratios < 0.65] = 10
        return _weights

    def create_weighted_sampler(self, dataset):
        sample_weights = self.get_sample_weights(dataset)
        return torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def create_weighted_cross_entropy_loss(self, affectnet_dataset):
        _weights = 1.0 / affectnet_dataset.get_class_sizes()
        if _weights[7] > 1.0: _weights[7] = 0
        _weights = _weights.astype(np.float32)
        _weights /= np.sum(_weights)
        log.info("AffectNet weights: {}".format(_weights))
        return torch.nn.CrossEntropyLoss(weight=torch.from_numpy(_weights).to(device))
        # return torch.nn.CrossEntropyLoss()


    def _save_snapshot(self, is_best=False):
        def write_model(out_dir, model_name, model):
            filepath_mdl = os.path.join(out_dir, model_name+'.mdl')
            snapshot = {
                        'epoch': self.epoch,
                        'total_iter': self.total_iter,
                        'total_time': self.total_training_time(),
                        'best_score': self.best_score,
                        'arch': type(model).__name__,
                        'state_dict': model.state_dict(),
                        }
            utils.io.makedirs(filepath_mdl)
            torch.save(snapshot, filepath_mdl)

        def write_meta(out_dir):
            with open(os.path.join(out_dir, 'meta.json'), 'w') as outfile:
                data = {'epoch': self.epoch+1,
                        'total_iter': self.total_iter,
                        'total_time': self.total_training_time(),
                        'best_score': self.best_score}
                json.dump(data, outfile)

        model_data_dir = os.path.join(self.snapshot_dir, self.session_name)
        model_snap_dir =  os.path.join(model_data_dir, '{:05d}'.format(self.epoch+1))
        write_model(model_snap_dir, 'saae', self.saae)
        # write_model(model_snap_dir, 'encoder', self.saae.Q.model)
        write_meta(model_snap_dir)

        # save a copy of this snapshot as the best one so far
        if is_best:
            utils.io.copy_files(src_dir=model_snap_dir, dst_dir=model_data_dir, pattern='*.mdl')

    def _load_snapshot(self, snapshot_name, data_dir=None):
        if data_dir is None:
            data_dir = self.snapshot_dir

        model_snap_dir = os.path.join(data_dir, snapshot_name)
        utils.nn.read_model(model_snap_dir, 'saae', self.saae)
        # saae.read_model(model_snap_dir, 'encoder', self.saae.Q.model)

        meta = utils.nn.read_meta(model_snap_dir)
        self.epoch = meta['epoch']
        self.total_iter = meta['total_iter']
        self.total_training_time_previous = meta.get('total_time', 0)
        self.best_score = meta['best_score']
        self.saae.total_iter = self.total_iter
        str_training_time = str(datetime.timedelta(seconds=self.total_training_time()))
        log.info("Model {} trained for {} iterations ({}).".format(snapshot_name, self.total_iter, str_training_time))

    def _is_snapshot_iter(self):
        return (self.total_iter+1) % self.snapshot_interval == 0 and (self.total_iter+1) > 0

    @property
    def print_interval(self):
        return self.args.print_freq if self.training else self.args.print_freq_eval

    def _is_printout_iter(self):
        return (self.iter_in_epoch+1) % self.print_interval == 0

    def _is_eval_epoch(self):
        return (self.epoch+1) % self.args.eval_freq == 0

    def _training_time(self):
        return int(time.time() - self.time_start_training)

    def total_training_time(self):
        return self.total_training_time_previous + self._training_time()

    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats).mean().to_dict()
        current = stats[-1]
        ssim_scores = current['ssim'].mean()

        str_stats = ['[{ep}][({i}/{iters_per_epoch}] '
                     'l_Q={avg_loss_Q:.3f}  '
                     'l_rec={avg_loss_recon:.3f} '
                     'l_ssim={avg_ssim_torch:.3f}({avg_ssim:.2f}) '
                     'l_lmrec={avg_lms_recon:.3f} '
                     'l_lmssim={avg_lms_ssim:.2f} '
                     'l_lmcs={avg_lms_cs:.2f} '
                     'l_lmncc={avg_lms_ncc:.2f} '
                     # 'l_act={avg_loss_activations:.3f} '
                     'z_mu={avg_z_recon_mean: .3f} ']
        str_stats[0] += [
            # 'l_D_z={avg_loss_D_z:.3f} '
            # 'l_E={avg_loss_E:.3f} '
            'l_D={avg_loss_D:.3f} '
            'l_G={avg_loss_G:.3f}({avg_loss_G_rec:.3f}/{avg_loss_G_gen:.3f}) '
            '{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {epoch_time})'][0]
        log.info(str_stats[0].format(
            ep=current['epoch']+1, i=current['iter']+1, iters_per_epoch=self.iters_per_epoch,
            avg_loss_Q=means.get('loss_Q', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_lms_recon=means.get('landmark_recon_errors', -1),
            avg_lms_ssim=means.get('landmark_ssim_scores', -1),
            avg_lms_ncc=means.get('landmark_ncc_errors', -1),
            avg_lms_cs=means.get('landmark_cs_errors', -1),
            avg_ssim=ssim_scores.mean(),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_activations=means.get('loss_activations', -1),
            avg_loss_F=means.get('loss_F', -1),
            avg_loss_E=means.get('loss_E', -1),
            avg_loss_D_z=means.get('loss_D_z', -1),
            avg_loss_D=means.get('loss_D', -1),
            avg_loss_G=means.get('loss_G', -1),
            avg_loss_G_rec=means.get('loss_G_rec', -1),
            avg_loss_G_gen=means.get('loss_G_gen', -1),
            avg_loss_D_real=means.get('err_real', -1),
            avg_loss_D_fake=means.get('err_fake', -1),
            avg_z_l1=means.get('z_l1', -1),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            avg_loss_disent=means.get('loss_disent', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter+1,
            epoch_time=str(datetime.timedelta(seconds=self._training_time()))
        ))

    def _print_epoch_summary(self, epoch_stats, epoch_starttime):
        means = pd.DataFrame(epoch_stats).mean().to_dict()
        try:
            ssim_scores = np.concatenate([stats['ssim'] for stats in self.epoch_stats if 'ssim' in stats])
        except:
            ssim_scores = np.array(0)
        duration = int(time.time() - epoch_starttime)

        log.info("{}".format('-'*140))
        str_stats = ['Train:         '
                 'l_Q={avg_loss_Q:.3f} '
                 'l_rec={avg_loss_recon:.3f} '
                 'l_ssim={avg_ssim_torch:.3f}({avg_ssim:.3f}) '
                 'l_lmrec={avg_lms_recon:.3f} '
                 'l_lmssim={avg_lms_ssim:.3f} '
                 'l_lmcs={avg_lms_cs:.3f} '
                 'l_lmncc={avg_lms_ncc:.3f} '
                 'z_mu={avg_z_recon_mean:.3f} ']
        str_stats[0] += [
            # 'l_D_z={avg_loss_D_z:.4f} '
            # 'l_E={avg_loss_E:.4f}  '
            'l_D={avg_loss_D:.4f} '
            'l_G={avg_loss_G:.4f} '
            '\tT: {epoch_time} ({total_time})'][0]
        log.info(str_stats[0].format(
            iters_per_epoch=self.iters_per_epoch,
            avg_loss_Q=means.get('loss_Q', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_lms_recon=means.get('landmark_recon_errors', -1),
            avg_lms_ssim=means.get('landmark_ssim_scores', -1),
            avg_lms_ncc=means.get('landmark_ncc_errors', -1),
            avg_lms_cs=means.get('landmark_cs_errors', -1),
            avg_ssim=ssim_scores.mean(),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_F=means.get('loss_F', -1),
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
            total_iter=self.total_iter+1, total_time=str(datetime.timedelta(seconds=self._training_time())),
            totatl_time= str(datetime.timedelta(seconds=self.total_training_time())),
            epoch_time=str(datetime.timedelta(seconds=duration))))
        try:
            recon_errors = np.concatenate([stats['l1_recon_errors'] for stats in self.epoch_stats])
            rmse = np.sqrt(np.mean(recon_errors**2))
            log.info("RMSE: {} ".format(rmse))
        except:
            print("no l1_recon_error")


    def eval_epoch(self):
        log.info("")
        log.info("Starting evaluation of '{}'...".format(self.session_name))
        log.info("")

        self.time_start_eval = time.time()
        epoch_starttime = time.time()

        self.epoch_stats = []
        self.saae.eval()
        self.training = False

        ds = self.datasets[VAL]

        self._run_epoch(ds, train_autoencoder=False, eval=True)

        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_starttime)

        # time_elapsed = time.time() - self.time_start_eval
        # print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def train(self, num_epochs):

        log.info("")
        log.info("Starting training session '{}'...".format(self.session_name))
        log.info("")

        while self.epoch < num_epochs:
            log.info('')
            log.info('Epoch {}/{}'.format(self.epoch+1, num_epochs))
            log.info('=' * 10)

            self.epoch_stats = []
            epoch_starttime = time.time()
            self.saae.train(self.train_autoencoder)
            self.training = True

            self._run_epoch(self.datasets[TRAIN], self.train_autoencoder)

            # save model every few epochs
            if (self.epoch+1) % self.snapshot_interval == 0 \
                    and (self.train_autoencoder or self.state_disentanglement == 2 or args.train_coords):
                log.info("*** saving snapshot *** ")
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_starttime)

            if self._is_eval_epoch() and cfg.INPUT_SCALE_FACTOR != 4:
                self.eval_epoch()

            # save visualizations to disk
            if (self.epoch+1) % 1 == 0:
                self.reconstruct_fixed_samples()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def _run_epoch(self, dataset, train_autoencoder, eval=False):
        batchsize = self.args.batchsize_eval if eval else self.batch_size

        self.iters_per_epoch = int(len(dataset)/batchsize)
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0

        if eval:
            dataloader = td.DataLoader(dataset, batch_size=batchsize, num_workers=self.workers)
        else:
            # sampler = self.create_weighted_sampler(dataset)
            sampler = None
            # dataloader = td.DataLoader(dataset, batch_size=batchsize, num_workers=self.workers,
            #                            drop_last=True, sampler=sampler)
            dataloader = td.DataLoader(dataset, batch_size=batchsize, num_workers=self.workers,
                                       drop_last=True, shuffle=True)

        if isinstance(dataset, affectnet.AffectNet):
            self.saae.weighted_CE_loss = self.create_weighted_cross_entropy_loss(dataset)

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

        if self.fixed_batch_train is None:
            self.fixed_batch_train = Batch(data, n=self.n_fixed)

        self.saae.zero_grad()
        y_ones = torch.FloatTensor(len(batch)).fill_(1).cuda()

        with torch.set_grad_enabled(train_autoencoder and cfg.TRAIN_ENCODER):

            #######################
            # Encoding
            #######################

            input_images = batch.images_mod if batch.images_mod is not None else batch.images
            z_sample = self.saae.Q(input_images)
            self.saae.z = z_sample

            if (train_autoencoder or self._is_printout_iter()) and cfg.WITH_ZGAN and cfg.TRAIN_ENCODER:

                #######################
                # AAE regularization phase
                #######################
                self.saae.Q.zero_grad()

                # Discriminator
                if self.iter_in_epoch % 2 == 0:
                    self.saae.D_z.zero_grad()
                    z_real = self.enc_rand_like(z_sample).to(device)

                    D_real = self.saae.D_z(z_real)
                    D_fake = self.saae.D_z(z_sample.detach())
                    loss_D_z = -torch.mean(torch.log(D_real + eps) + torch.log(1 - D_fake + eps))
                    if train_autoencoder and cfg.TRAIN_ENCODER:
                        loss_D_z.backward()
                        self.optimizer_D_z.step()
                    iter_stats.update({'loss_D_z': loss_D_z.item()})

                # Encoder gaussian loss
                self.saae.D_z.zero_grad()
                D_fake = self.saae.D_z(z_sample)
                loss_E = -torch.mean(torch.log(D_fake + eps))

                if train_autoencoder and cfg.TRAIN_ENCODER:
                    loss_E.backward(retain_graph=True)
                    self.optimizer_Q.step()

                iter_stats.update({'loss_E': loss_E.item()})

        self.saae.iter = self.iter_in_epoch
        self.saae.images = batch.images
        self.saae.current_dataset = ds

        z_recon = z_sample

        iter_stats.update({'z_recon_mean':  z_recon.mean().item(),
                           'z_l1': F.l1_loss(z_sample, z_recon).item()})

        #######################
        # Reconstruction phase
        #######################
        with torch.set_grad_enabled(train_autoencoder):
            if train_autoencoder or self._is_printout_iter() or eval or self.args.train_coords:
                X_target = batch.images
                loss_Q = torch.zeros(1, requires_grad=True).cuda()
                self.saae.P.zero_grad()

                with_recon_loss = True
                X_recon = None
                if with_recon_loss:
                    # reconstruct images
                    if cfg.TRAIN_ENCODER:
                        outputs = self.saae.P(z_recon)
                    else:
                        outputs = self.saae.P(z_recon.detach())

                    X_recon = outputs[:, :3]   # take RGB layers

                    #######################
                    # Reconstruction loss
                    #######################

                    l1_dists_recon = (torch.abs(batch.images - X_recon)).mean(dim=1) * 255
                    loss_recon_l1 = torch.mean(l1_dists_recon)
                    iter_stats['l1_recon_errors'] = to_numpy(255.0 * torch.abs(batch.images - X_recon).reshape(len(batch.images), -1).mean(dim=1))

                    if eval and batch.landmarks is not None:
                        iter_stats['landmark_ssim_scores'] = lmutils.calc_landmark_ssim_score(X_target, X_recon, batch.landmarks).mean()
                        iter_stats['landmark_recon_errors'] = lmutils.calc_landmark_recon_error(X_target, X_recon, batch.landmarks).mean()
                        iter_stats['landmark_ncc_errors'] = lmutils.calc_landmark_ncc(X_target, X_recon, batch.landmarks).mean()
                        iter_stats['landmark_cs_errors'] = lmutils.calc_landmark_cs_error(X_target, X_recon, batch.landmarks,
                                                                                          torch_ssim=self.ssim).mean()

                    loss_Q += loss_recon_l1 * cfg.W_RECON
                    iter_stats.update({'loss_recon': loss_recon_l1.item()})

                #######################
                # SSIM loss
                #######################
                cs_error_maps = []
                store_cs_maps = self._is_printout_iter() # get ssim error maps for visualization
                if cfg.WITH_SSIM_LOSS or eval:
                    ssim_torch = torch.zeros(len(batch), requires_grad=True).cuda()
                    for i in range(len(batch)):
                        ssim_torch[i] = 1.0 - self.ssim(batch.images[i].unsqueeze(0), X_recon[i].unsqueeze(0))
                        if store_cs_maps:
                            cs_error_maps.append(1.0 - to_numpy(self.ssim.cs_map))
                    loss_ssim = ssim_torch.mean() * cfg.W_SSIM
                    loss_Q = 0.5 * loss_Q + 0.5 * loss_ssim
                    iter_stats['ssim_torch'] = loss_ssim.item()
                    if len(cs_error_maps) > 0:
                        cs_error_maps = np.vstack(cs_error_maps)
                if len(cs_error_maps) == 0:
                    cs_error_maps = None

                if eval or self._is_printout_iter():
                    ssim = np.zeros(len(batch))
                    input_images = vis._to_disp_images(batch.images, denorm=True)
                    recon_images = vis._to_disp_images(X_recon, denorm=True)
                    for i in range(len(batch)):
                        data_range = 255.0 if input_images[0].dtype == np.uint8 else 1.0
                        ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=data_range, multichannel=True)
                    iter_stats['ssim'] = ssim

                iter_stats.update({'loss_Q': loss_Q.item()})

                if train_autoencoder and cfg.TRAIN_DECODER:
                    loss_Q.backward(retain_graph=True)

                if cfg.WITH_GAN and cfg.TRAIN_DECODER and self.iter_in_epoch%1 == 0:
                    if  self.iter_in_epoch % cfg.UPDATE_DISCRIMINATOR_FREQ == 0:
                        w_gen = 0.25
                        # #######################
                        # # GAN discriminator phase
                        # #######################
                        self.saae.D.zero_grad()
                        err_real = self.saae.D(X_target)
                        err_fake = self.saae.D(X_recon.detach())
                        # from sklearn.utils import shuffle
                        # err_fake = err_fake[shuffle(range(len(err_fake)))]
                        if cfg.WITH_GEN_LOSS:
                            err_fake_gen = self.saae.D(X_gen.detach())
                        assert(len(err_real) == len(X_target))
                        if cfg.RGAN:
                            if cfg.WITH_GEN_LOSS:
                                loss_D_gen = self.BCE_stable(err_real - err_fake_gen, y_ones)
                            loss_D = self.BCE_stable(err_real - err_fake, y_ones)
                        else:
                            if cfg.WITH_GEN_LOSS:
                                loss_D_gen = -torch.mean(torch.log(err_real + eps) + torch.log(1.0 - err_fake_gen + eps))
                            loss_D = -torch.mean(torch.log(err_real + eps) + torch.log(1.0 - err_fake + eps))
                        if cfg.WITH_GEN_LOSS:
                            loss_D = loss_D*(1-w_gen) + loss_D_gen*w_gen
                        if train_autoencoder :
                            # if loss_D > 0.7:
                                loss_D.backward()
                                self.optimizer_D.step()

                        iter_stats.update({'loss_D': loss_D.item(), 'err_real': err_real.mean().item()})

                    #######################
                    # Generator loss
                    #######################
                    if self.iter_in_epoch % cfg.UPDATE_ENCODER_FREQ == 0:
                        self.saae.D.zero_grad()
                        set_requires_grad(self.saae.D, False)

                        # X_recon = self.saae.P(self.saae.Q(batch.images))[:,:3]

                        err_G_random = self.saae.D(X_recon)
                        if cfg.WITH_GEN_LOSS:
                            err_G_gen = self.saae.D(X_gen)
                        if cfg.RGAN:
                            err_real = self.saae.D(X_target)
                            if cfg.WITH_GEN_LOSS:
                                loss_G_gen = self.BCE_stable(err_G_gen - err_real, y_ones)
                            loss_G_rec = self.BCE_stable(err_G_random - err_real, y_ones)
                        else:
                            loss_G_rec = -torch.mean(torch.log(err_G_random + eps))
                            if cfg.WITH_GEN_LOSS:
                                loss_G_gen = -torch.mean(torch.log(err_G_gen + eps))
                        set_requires_grad(self.saae.D, True)


                        if cfg.WITH_GEN_LOSS:
                            loss_G = loss_G_rec*(1-w_gen) + loss_G_gen*(w_gen)
                            iter_stats.update({'loss_G_rec': loss_G_rec.item(), 'loss_G_gen': loss_G_gen.item()})
                        else:
                            loss_G = loss_G_rec

                        if train_autoencoder:
                            loss_G.backward(retain_graph=True)

                        iter_stats.update({'loss_G': loss_G.item(), 'err_fake': loss_G.mean().item()})

                # # Update auto-encoder
                if train_autoencoder and not eval:
                    if cfg.TRAIN_ENCODER:
                        self.optimizer_Q.step()
                    if cfg.TRAIN_DECODER:
                        self.optimizer_P.step()

        # statistics
        iter_stats.update({'epoch': self.epoch, 'timestamp': time.time(),
                           'iter_time': time.time() - self.iter_starttime,
                           'time_processing': time.time() - time_proc_start,
                           'iter': self.iter_in_epoch, 'total_iter': self.total_iter,
                           'batch_size': len(batch)})
        self.iter_starttime = time.time()
        self.epoch_stats.append(iter_stats)

        # print stats every N mini-batches
        if self._is_printout_iter():
            self._print_iter_stats(self.epoch_stats[-self.print_interval:])

        #
        # Batch visualization
        #
        if self._is_printout_iter() or eval:
            if self.args.show:
                self.visualize_batch(batch, X_recon, ssim_maps=cs_error_maps, ds=ds, wait=self.wait)


    #
    # Visualizations
    #

    def visualize_random_faces(self, wait=10):

        loc_err_gan = 'tr'
        nimgs = 8
        with torch.no_grad():
            z_random = self.enc_rand(nimgs, self.saae.z_dim).to(device)
            X_gen_vis = self.saae.P(z_random)[:, :3]
            disp_X_gen = to_numpy(ds_utils.denormalized(X_gen_vis).permute(0, 2, 3, 1))
            err_gan_gen = self.saae.D(X_gen_vis)
        disp_X_gen = vis.add_error_to_images(disp_X_gen, errors=1 - err_gan_gen, loc=loc_err_gan, format_string='{:.2f}', vmax=1.0)

        grid_img = vis.make_grid(disp_X_gen, nCols=nimgs)
        cv2.imshow("random face", cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(wait)


    def visualize_batch(self, batch, X_recon, ssim_maps, ds=None, wait=0):

        nimgs = min(8, len(batch))
        train_state_D = self.saae.D.training
        train_state_Q = self.saae.Q.training
        train_state_P = self.saae.P.training
        self.saae.D.eval()
        self.saae.Q.eval()
        self.saae.P.eval()

        loc_err_gan = 'tr'
        text_size_errors = 0.65

        input_images = vis._to_disp_images(batch.images[:nimgs], denorm=True)

        if batch.images_mod is not None:
            disp_images = vis._to_disp_images(batch.images_mod[:nimgs], denorm=True)
        else:
            disp_images = vis._to_disp_images(batch.images[:nimgs], denorm=True)


        # draw GAN score
        if cfg.WITH_GAN:
            with torch.no_grad():
                err_gan_inputs = self.saae.D(batch.images[:nimgs])
            disp_images = vis.add_error_to_images(disp_images, errors=1-err_gan_inputs, loc=loc_err_gan, format_string='{:>5.2f}', vmax=1.0)

        disp_images = vis.add_landmarks_to_images(disp_images, batch.landmarks[:nimgs], color=(0,1,0), radius=1,
                                                  draw_wireframe=False, landmarks_to_draw=lmcfg.LANDMARKS)
        rows = [vis.make_grid(disp_images, nCols=nimgs, normalize=False)]

        recon_images = vis._to_disp_images(X_recon[:nimgs], denorm=True)
        disp_X_recon = recon_images.copy()

        lm_ssim_errs = None
        # if batch.landmarks is not None:
        #     lm_recon_errs = lmutils.calc_landmark_recon_error(batch.images[:nimgs], X_recon[:nimgs], batch.landmarks[:nimgs], reduction='none')
        #     disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_recon_errs, size=text_size_errors, loc='bm',
        #                                            format_string='({:>3.1f})', vmin=0, vmax=10)
        #     lm_ssim_errs = lmutils.calc_landmark_ssim_error(batch.images[:nimgs], X_recon[:nimgs], batch.landmarks[:nimgs])
        #     disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_ssim_errs.mean(axis=1), size=text_size_errors, loc='bm-1',
        #                                            format_string='({:>3.2f})', vmin=0.2, vmax=0.8)

        X_recon_errs = 255.0 * torch.abs(batch.images - X_recon).reshape(len(batch.images), -1).mean(dim=1)
        disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, batch.landmarks[:nimgs], radius=1, color=None,
                                                   lm_errs=lm_ssim_errs, draw_wireframe=False,
                                                   landmarks_to_draw=lmcfg.ALL_LANDMARKS)
        disp_X_recon = vis.add_error_to_images(disp_X_recon[:nimgs], errors=X_recon_errs, size=text_size_errors, format_string='{:>4.1f}')
        if cfg.WITH_GAN:
            with torch.no_grad():
                err_gan = self.saae.D(X_recon[:nimgs])
            disp_X_recon = vis.add_error_to_images(disp_X_recon, errors=1 - err_gan, loc=loc_err_gan, format_string='{:>5.2f}', vmax=1.0)

        ssim = np.zeros(nimgs)
        for i in range(nimgs):
            data_range = 255.0 if input_images[0].dtype == np.uint8 else 1.0
            ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=data_range, multichannel=True)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', size=text_size_errors, format_string='{:>4.2f}', vmin=0.2, vmax=0.8)

        if ssim_maps is not None:
            disp_X_recon = vis.add_error_to_images(disp_X_recon, ssim_maps.reshape(len(ssim_maps), -1).mean(axis=1),
                                                   size=text_size_errors, loc='bl-2', format_string='{:>4.2f}', vmin=0.0, vmax=0.4)

        rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))


        if ssim_maps is not None:
            disp_ssim_maps = to_numpy(ds_utils.denormalized(ssim_maps)[:nimgs].transpose(0, 2, 3, 1))
            for i in range(len(disp_ssim_maps)):
                disp_ssim_maps[i] = vis.color_map(disp_ssim_maps[i].mean(axis=2), vmin=0.0, vmax=2.0)
            grid_ssim_maps = vis.make_grid(disp_ssim_maps, nCols=nimgs)
            cv2.imshow('ssim errors', cv2.cvtColor(grid_ssim_maps, cv2.COLOR_RGB2BGR))

        self.saae.D.train(train_state_D)
        self.saae.Q.train(train_state_Q)
        self.saae.P.train(train_state_P)

        f = 2 / cfg.INPUT_SCALE_FACTOR
        disp_rows = vis.make_grid(rows, nCols=1, normalize=False, fx=f, fy=f)
        wnd_title = 'recon errors '
        if ds is not None:
            wnd_title += ds.__class__.__name__
        cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
        cv2.waitKey(wait)

    def visualize_activations(self):
        out_dir = os.path.join(cfg.REPORT_DIR, 'activations')
        img_id = 0
        with torch.no_grad():
            # get activations from feature extraction network
            input = self.fixed_batch_val.images[img_id:img_id+1]
            recon = self.saae.P(self.saae.Q(input))
            feats_input = self.fe(input)
            feats_recon = self.fe(recon)
            loss_F = ((feats_input - feats_recon)**2).mean() * 500.0

            # plot results
            img_feats_input = vis.make_grid(to_numpy(feats_input[0, :25]), nCols=5)
            img_feats_recon = vis.make_grid(to_numpy(feats_recon[0, :25]), nCols=5)
            input = ds_utils.denormalized(input)
            recon = ds_utils.denormalized(recon)
            recon = recon.clamp(min=0, max=1)
            img_input = to_numpy(input[0].permute(1, 2, 0))
            img_recon = to_numpy(recon[0].permute(1, 2, 0))
            fig, axes = plt.subplots(2,2, figsize=(12,10))
            axes[0, 0].imshow(img_input)
            axes[0, 1].imshow(img_recon)
            axes[1, 0].imshow(img_feats_input)
            axes[1, 1].imshow(img_feats_recon)
            fig.suptitle("{:.4f}".format(loss_F))
            # plt.show()
            img_filepath =  os.path.join(out_dir, 'train', 'ft_train_{}.png'.format(self.epoch+1))
            utils.io.makedirs(img_filepath)
            try:
                plt.savefig(img_filepath, bbox_inches='tight', pad_inches=0)
            except SystemError:  # disable output when exiting program with Ctrl-C (hacky)
                pass

    def reconstruct_fixed_samples(self):
        out_dir = os.path.join(cfg.REPORT_DIR, 'reconstructions', self.session_name)

        f = 1.5 / cfg.INPUT_SCALE_FACTOR

        # reconstruct some training images
        b = self.fixed_batch_train
        img = vis_reconstruction(self.saae,
                                 b.images,
                                 landmarks=b.landmarks,
                                 ncols=10,
                                 skip_disentanglement=self.state_disentanglement==0,
                                 fx=f, fy=f)
        img_filepath =  os.path.join(out_dir, 'train', 'reconst_train-{}_{}.jpg'.format(self.session_name, self.epoch+1))
        utils.io.makedirs(img_filepath)
        cv2.imwrite(img_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


        # reconstruct some validation images
        b = self.fixed_batch_val
        img = vis_reconstruction(self.saae,
                                 b.images,
                                 clips=b.clips,
                                 landmarks=b.landmarks,
                                 ncols=10,
                                 skip_disentanglement=self.state_disentanglement==0,
                                 fx=f, fy=f)
        img_filepath = os.path.join(out_dir, 'val', 'reconst_val-{}_{}.jpg'.format(self.session_name, self.epoch+1))
        utils.io.makedirs(img_filepath)
        cv2.imwrite(img_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



def run(args):

    if args.seed is not None:
        init_random(args.seed)

    # log.info(json.dumps(vars(args), indent=4))

    phase_cfg = {
        TRAIN: {'dsnames': args.dataset_train,
                'count': args.train_count},
        VAL: {'dsnames': args.dataset_val,
              'count': args.val_count}
    }
    datasets = {}
    for phase in args.phases:
        dsnames = phase_cfg[phase]['dsnames']
        num_samples = phase_cfg[phase]['count']
        is_single_dataset = isinstance(dsnames, str) or len(dsnames) == 1
        train = phase == TRAIN
        datasets_for_phase = []
        for name in dsnames:
            ds = DATASETS[name](train=train, max_samples=num_samples, use_cache=args.use_cache,
                                start=args.st if train else None, test_split=args.test_split,
                                align_face_orientation=args.align,
                                crop_source=args.crop_source, daug=args.daug,
                                return_heatmaps=False,
                                return_modified_images=args.mod and train)
            datasets_for_phase.append(ds)
        if is_single_dataset:
            datasets[phase] = datasets_for_phase[0]
        else:
            datasets[phase] = multi.MultiFaceDataset(datasets_for_phase, max_samples=args.train_count_total)

        print(datasets[phase])

    fntr = AAETraining(datasets, args, session_name=args.sessionname, lr=args.lr, batch_size=args.batchsize,
                       snapshot_interval=args.save_freq, snapshot=args.resume,
                       workers=args.workers, train_autoencoder=args.train_ae,
                       state_disentanglement=args.disent, wait=args.wait)

    torch.backends.cudnn.benchmark = True
    if args.eval:
        fntr.eval_epoch()
    else:
        fntr.train(num_epochs=args.epochs)


if __name__ == '__main__':

    import sys
    # Disable traceback on Ctrl+c
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    bool_str = lambda x: (str(x).lower() in ['true', '1'])

    default_batchsize = 100 if cfg.INPUT_SCALE_FACTOR == 1 else 50
    default_batchsize_eval = 45 if cfg.INPUT_SCALE_FACTOR < 4 else 15

    parser = argparse.ArgumentParser()
    parser.add_argument('--sessionname',  default=None, type=str, help='output filename (without ext)')
    parser.add_argument('--train-count', default=None, type=int, help='number of training images per DB')
    parser.add_argument('--train-count-total', default=None, type=int, help='number of total training images')
    parser.add_argument('--st', default=None, type=int, help='skip first n training images')
    parser.add_argument('--val-count',  default=200, type=int, help='number of test images')
    parser.add_argument('-b', '--batchsize', default=default_batchsize, type=int, metavar='N', help='batch size (default: {})'.format(default_batchsize))
    parser.add_argument('--batchsize-eval', default=default_batchsize_eval, type=int, metavar='N', help='batch size for evaluation')
    parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print every N steps')
    parser.add_argument('--print-freq-eval', default=1, type=int, metavar='N', help='print every N steps')
    parser.add_argument('--save-freq', default=1, type=int, metavar='N', help='save snapshot every N epochs')
    parser.add_argument('--eval-freq', default=1, type=int, metavar='N', help='evaluate every N steps')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH', help='path to snapshot (default: None)')
    parser.add_argument('-e', '--epochs', default=10000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', help='number of data loading workers (default: 6)')
    parser.add_argument('--dataset-train', default=['vggface2', 'affectnet'], type=str, help='dataset(s) for training.', choices=DATASETS, nargs='+')
    parser.add_argument('--dataset-val', default=['300w'], type=str, help='dataset for training.', choices=DATASETS, nargs='+')
    parser.add_argument('--test-split', default='challenging', type=str, help='test set split for 300W/AFLW/WFLW',
                        choices=['challenging', 'common', '300w', 'full', 'frontal']+wflw.subsets)
    parser.add_argument('--tag', type=str, default=None, help='append string to tensorboard log name')
    parser.add_argument('--disent', type=int, default=0, help='state of disentanglement (0=disabled, 1=run, 2=train', choices=[0,1,2])
    parser.add_argument('--daug', type=int, default=0, help='state of data augmentation for training')
    parser.add_argument('--train-ae', type=bool_str, default=True, help='train auto-encoder')
    parser.add_argument('--lr', default=0.00002, type=float, help='learning rate for autoencoder')
    parser.add_argument('--eval', default=False, action='store_true',  help='run evaluation instead of training')
    parser.add_argument('--phases', default=[TRAIN, VAL], nargs='+')
    parser.add_argument('--use-cache', type=bool_str, default=True, help='use cached crops')
    parser.add_argument('--crop-source', type=str, default='bb_ground_truth')
    parser.add_argument('--align', type=bool_str, default=cfg.CROP_ALIGN_ROTATION, help='rotate crop so eyes are horizontal')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--reset', default=False, action='store_true', help='reset the discriminator')
    parser.add_argument('--mod', type=bool_str, default=False, help='create modified copies of input images')
    parser.add_argument('-z','--embedding-dims', default=cfg.EMBEDDING_DIMS, type=int, help='dimenionality of embedding ')

    # AE losses
    parser.add_argument('--no-ladv', default=False, action='store_true', help='without GAN loss')
    parser.add_argument('--no-lgen', default=True, action='store_true', help='without gen loss')
    parser.add_argument('--no-laecyc', default=True, action='store_true', help='without AE cycle loss')
    parser.add_argument('--no-lssim', default=False, action='store_true', help='without SSIM loss')

    # visualization
    parser.add_argument('--show-random-faces', default=False, action='store_true')
    parser.add_argument('--wait', default=10, type=int)
    parser.add_argument('--show', type=bool_str, default=True, help='visualize batches')

    args = parser.parse_args()

    lmcfg.config_landmarks('300w')

    args.color = True
    if args.eval:
        log.info('Switching to evaluation mode...')
        args.train_ae = False
        args.batchsize_eval = 10
        args.wait = 0
        args.workers = 4
        args.print_freq = 1
        if args.disent == 2:
            args.disent = 1
        args.epochs = 1
        args.phases = [VAL]
        args.val_count = 10000

    if args.sessionname is None:
        if args.resume:
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = 'debug'

    cfg.WITH_GAN = not args.no_ladv
    cfg.WITH_GEN_LOSS = not args.no_lgen
    cfg.WITH_SSIM_LOSS = not args.no_lssim

    cfg.WITH_LANDMARK_LOSS = False
    cfg.EMBEDDING_DIMS = args.embedding_dims

    run(args)

