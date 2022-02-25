#!/usr/bin/env python
from __future__ import print_function
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
import argparse
from collections import OrderedDict, defaultdict
from datetime import date, datetime

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
import torch.nn.functional as F
print(torch.__version__)
# from personality_recognition.evaluate import compute_metrics
# import apex

from utils import count_params, import_class


OCEAN_COLUMNS = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z',
             'NEGATIVEEMOTIONALITY_Z']

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def my_loss_func(output, target, weight):
    assert weight.shape[1] == target.shape[1]
    loss = torch.mean(((output - target) ** 2) * weight)
    return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def update_feeder_args(feeder_args, dataset_dir):
    if "data_path" in feeder_args:
        feeder_args['data_path'] = os.path.join(dataset_dir, feeder_args['data_path'])

    if "label_path" in feeder_args:
        feeder_args['label_path'] = os.path.join(dataset_dir, feeder_args['label_path'])

    if "laban_path" in feeder_args:
        feeder_args['laban_path'] = os.path.join(dataset_dir, feeder_args['laban_path'])

    if "info_path" in feeder_args:
        feeder_args['info_path'] = os.path.join(dataset_dir, feeder_args['info_path'])

def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
        '--dataset-dir',
        required=True,
        help='Path to dataset folder'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--assume-yes',
        action='store_true',
        help='Say yes to every prompt')

    parser.add_argument(
        '--personality_index',
        default='0',
        help='Index of the OCEAN trait')

    parser.add_argument(
        '--phase',
        default='train',
        help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    parser.add_argument(
        '--seed',
        type=int,
        default=random.randrange(200),
        help='random seed')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--eval-start',
        type=int,
        default=1,
        help='The epoch number to start evaluating models')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    parser.add_argument(
        '--feeder',
        default='feeder.feeder',
        help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=os.cpu_count(),
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--amp-opt-level',
        type=int,
        default=1,
        help='NVIDIA Apex AMP optimization level')

    parser.add_argument(
        '--base-lr',
        type=float,
        default=0.01,
        help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--optimizer',
        default='SGD',
        help='type of optimizer')
    parser.add_argument(
        '--nesterov',
        type=str2bool,
        default=False,
        help='use nesterov or not')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=256,
        help='test batch size')
    parser.add_argument(
        '--forward-batch-size',
        type=int,
        default=16,
        help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--optimizer-states',
        type=str,
        help='path of previously saved optimizer states')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='path of previously saved training checkpoint')
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Debug mode; default false')

    return parser


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == "test":
            self.ses_df = pd.read_csv(os.path.join(self.arg.dataset_dir, "metadata_test/sessions_test.csv"), dtype={'ID': str})
            self.ppl_df = pd.read_csv(os.path.join(self.arg.dataset_dir, "metadata_test/parts_test.csv"))
        else:
            train_ses_df = pd.read_csv(os.path.join(self.arg.dataset_dir, "metadata_train/sessions_train.csv"), dtype={'ID': str})
            train_ppl_df = pd.read_csv(os.path.join(self.arg.dataset_dir, "metadata_train/parts_train.csv"))
            val_ses_df = pd.read_csv(os.path.join(self.arg.dataset_dir, "metadata_val/sessions_val.csv"), dtype={'ID': str})
            val_ppl_df = pd.read_csv(os.path.join(self.arg.dataset_dir, "metadata_val/parts_val.csv"))

            self.ses_df = pd.concat((train_ses_df, val_ses_df), ignore_index=True)
            self.ppl_df = pd.concat((train_ppl_df, val_ppl_df), ignore_index=True)

            print('train number_of_people', len(train_ppl_df), 'val number_of_people', len(val_ppl_df))
            print('train number_of_sessions', len(train_ses_df), 'val number_of_sessions', len(val_ses_df))

        print('number_of_people', len(self.ppl_df))
        print('number_of_sessions', len(self.ses_df))
        if arg.phase == 'train':
            # Added control through the command line
            arg.train_feeder_args['debug'] = arg.train_feeder_args['debug'] or self.arg.debug
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.train_feeder_args['debug']:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'n'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)
                date_str = str(datetime.now())
                date_str = date_str.replace(" ", "_")
                date_str = date_str.replace(":", "_")
                self.train_writer = SummaryWriter(os.path.join(logdir, 'train_' + date_str), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val_'+ date_str), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')
        update_feeder_args(self.arg.train_feeder_args, self.arg.dataset_dir)
        update_feeder_args(self.arg.test_feeder_args, self.arg.dataset_dir)

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_loss = float("inf")
        self.best_loss_epoch = 0


        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            # self.model, self.optimizer = apex.amp.initialize(
            #     self.model,
            #     self.optimizer,
            #     opt_level=f'O{self.arg.amp_opt_level}'
            # )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)

        # Copy model file and main
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.MSELoss().cuda(output_device)
        self.print_log(f'Model total number of params: {count_params(self.model)}')

        if self.arg.weights:
            try:
                self.global_step = int(arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            # for w in self.arg.ignore_weights:
            #     if weights.pop(w, None) is not None:
            #         self.print_log(f'Sucessfully Remove Weights: {w}')
            #     else:
            #         self.print_log(f'Can Not Remove Weights: {w}')
            # filter weights
            for i in self.arg.ignore_weights:
                ignore_name = list()
                for w in weights:
                    if w.find(i) == 0:
                        ignore_name.append(w)
                for n in ignore_name:
                    weights.pop(n)
                    self.print_log('Filter [{}] remove weights [{}].'.format(i, n))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
            for param in self.model.named_parameters():
                if param[0] in weights:
                    param[1].requires_grad = False
                    print("Freezing", param[0])
    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(personality_index=self.arg.personality_index, **self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn,)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(personality_index=self.arg.personality_index, **self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
        self.save_states(epoch, state_dict, out_folder, checkpoint_name)

    def save_weights(self, epoch, out_folder='weights', is_best=False):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        weights_name = f'weights-{epoch}-{int(self.global_step)}-{"best" if is_best else ""}.pt'
        self.save_states(epoch, weights, out_folder, weights_name)

    def train(self, epoch, save_model=False):
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []
        # self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.4f}')

        process = tqdm(loader, dynamic_ncols=True)
        print(self.model.named_parameters())
        total_num_params = 0
        total_num_zeros = 0
        for batch_idx, (data, laban, label, weight) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                laban = laban.float().cuda(self.output_device)
                label = label.float().cuda(self.output_device)
                weight = weight.float().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, batch_laban, batch_label = data[left:right], laban[left:right], label[left:right]

                # forward
                output = self.model(batch_data, batch_laban)
                # (output == 0).sum(1).float() / output.size(1)

                if isinstance(output, tuple):
                    output, l1 = output
                    l1 = l1.mean()
                else:
                    l1 = 0

                loss = self.loss(output, batch_label) / splits

                if self.arg.half:
                    pass
                    # with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                else:
                    loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                # value, predict_label = torch.max(output, 1)
                # acc = torch.mean((predict_label == batch_label).float())
                #
                # self.train_writer.add_scalar('acc', acc, self.global_step)
                # self.train_writer.add_scalar('loss_l1', l1, self.global_step)

            #####################################

            for index, (name, param) in enumerate(self.model.named_parameters()):
                grads = param.grad

                if grads is None:
                    continue
                num_elements = torch.numel(grads)
                num_non_zeros = torch.sum(torch.abs(grads) > 0).item()


                total_num_params += num_elements
                total_num_zeros += num_elements - num_non_zeros



            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            # self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        print("=" * 50, "\nCALCULATING ZERO GRADS\n", "=" * 50)

        ratio = total_num_zeros / total_num_params
        print("Zero Ratio:", ratio, " " * 8, total_num_zeros, " / ", total_num_params)
        print("=" * 50)

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        self.train_writer.add_scalar('losses/per_frame_loss', mean_loss, epoch)

        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(f'\tMean training loss: {mean_loss:.4f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.4f}).')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        # PyTorch > 1.2.0: update LR scheduler here with `.step()`
        # and make sure to save the `lr_scheduler.state_dict()` as part of checkpoint
        self.lr_scheduler.step()

        if save_model:
            # save training checkpoint & weights
            self.save_weights(epoch + 1)
            self.save_checkpoint(epoch + 1)

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return

        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            mse_per_class = [-1 for _ in OCEAN_COLUMNS]
            for ln in loader_name:
                loss_values = []
                score_batches = []
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                pred = []
                for batch_idx, (data, laban, label, weight) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    laban = laban.float().cuda(self.output_device)

                    label = label.float().cuda(self.output_device)
                    output = self.model(data, laban)
                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_batches.append(output.data.cpu().numpy())
                    loss_values.append(loss.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    predict = output.cpu().numpy()
                    pred.append(predict)
                pred = np.concatenate(pred)
                personality_index = self.arg.personality_index
                people_ids = self.ppl_df['ID'].to_numpy()
                predictions = {people_id: np.zeros((0, self.arg.model_args['num_class'])) for people_id in people_ids}
                people_occurence_counts = {people_id: 0 for people_id in people_ids}
                print(pred.shape)
                infos = self.data_loader[ln].dataset.info
                new_info = np.zeros((0, 2))
                for info in infos:
                    info = np.tile(info[:2], (int(info[2]), 1))
                    new_info = np.vstack((new_info, info))
                print(pred.shape, new_info.shape)
                person_predictions = []
                for i, data in enumerate(pred):
                    session_id, person_id = new_info[i]
                    # print(y_test[i])
                    # print(pred[i])
                    person_id = int(person_id)
                    people_occurence_counts[person_id] += 1
                    predictions[person_id] = np.vstack((predictions[person_id], pred[i]))
                    # print('---')
                    person_predictions.append((person_id, *pred[i]))


                predictions_data = []
                gt_data = []

                for person_id, count in people_occurence_counts.items():
                    if count == 0: continue
                    assert len(predictions[person_id]) == count
                    prediction = predictions[person_id].mean(axis=0)

                    assert len(prediction) == predictions[person_id].shape[1]
                    gt = self.ppl_df[self.ppl_df['ID'] == person_id][OCEAN_COLUMNS].to_numpy()[0]

                    prediction = np.hstack(([person_id], prediction))
                    gt = np.hstack(([person_id], gt))
                    predictions_data.append(prediction)
                    gt_data.append(gt)
                gt_data = np.array(gt_data)
                predictions_data = np.array(predictions_data)
                predictions_df = pd.DataFrame(data=predictions_data, columns=['Participant ID', *OCEAN_COLUMNS])
                gt_df = pd.DataFrame(data=gt_data, columns=['Participant ID', *OCEAN_COLUMNS])
                predictions_df['Participant ID'] = predictions_df['Participant ID'].astype(int)

                person_predictions_df = pd.DataFrame(data=person_predictions,
                                                     columns=['Participant ID', *OCEAN_COLUMNS])
                # gt_df['Participant ID'] = gt_df['Participant ID'].astype(int)
                if self.arg.phase == "test":
                    pd.set_option('display.max_columns', None)
                    print(predictions_df)
                    print(gt_df)
                    predictions_df.to_csv("./predictions.csv", index=False)
                    gt_df.to_csv("./ground_truth.csv", index=False)
                    person_predictions_df.to_csv("./person_predictions.csv", index=False) #SERKAN


                mse_per_class = np.mean(np.square(np.subtract(gt_data[:, 1:], predictions_data[:, 1:])), axis=0)
                my_loss = np.mean(mse_per_class)
                print('My Mean Loss:', my_loss)
                print('My Losses:', mse_per_class)


            score = np.concatenate(score_batches)
            loss = my_loss#np.mean(loss_values)
            per_frame_loss = np.mean(loss_values)
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_loss_epoch = epoch + 1
                self.save_weights(epoch + 1, is_best=True)
                self.save_checkpoint(epoch + 1)

            if self.arg.phase == 'train' and not self.arg.debug:
                self.val_writer.add_scalar('losses/my_loss', loss, epoch)
                self.val_writer.add_scalar('losses/per_frame_loss', per_frame_loss, epoch)

                for trait, loss in zip(OCEAN_COLUMNS, mse_per_class):
                    self.val_writer.add_scalar(f'losses/{trait}_loss', loss, epoch)

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {per_frame_loss}.')
            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')
            wf = rf = None
            self.eval(
                epoch=0,
                save_score=self.arg.save_score,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()