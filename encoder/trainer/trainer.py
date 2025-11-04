import json
import os
import time
import random
import numpy as np
from numpy import random
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.optim as optim
from encoder.trainer.metrics import Metric
from encoder.models.bulid_model import build_model
from encoder.config.configurator import configs
from .utils import DisabledSummaryWriter, log_exceptions


writer = DisabledSummaryWriter()

def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if 'log_loss' in configs['train'] and configs['train']['log_loss']:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)

    '''# 专为llmrec的
    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()

            # Set retain_graph=True for the first backward pass
            loss.backward(retain_graph=True)  # Retain the computation graph for the next backward pass

            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if 'log_loss' in configs['train'] and configs['train']['log_loss']:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)'''

    @log_exceptions
    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_recall = -1e9
        self.create_optimizer(model)
        train_config = configs['train']
        for epoch_idx in range(train_config['epoch']):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.evaluate(model, epoch_idx)

                if eval_result['recall'][-1] > best_recall:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_recall = eval_result['recall'][-1]
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                # early stop
                if now_patience == configs['train']['patience']:
                    break

        # evaluation again
        # print(best_state_dict)
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict, strict=False)
        self.evaluate(model)

        # final test
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict, strict=False)
        test_result = self.test(model)

        # save result
        self.save_model(model)
        self.logger.log("Best Epoch {}. Final test result: {}.".format(best_epoch, test_result))

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        eval_result, _ = self.metric.eval_save(model, self.data_handler.valid_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        eval_result, _ = self.metric.eval_save(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        with open('candidate.txt', 'w') as file:
            json.dump(_, file, ensure_ascii=False, indent=4)
        return eval_result
    
    @log_exceptions
    def test_save(self, model):
        model.eval()
        eval_result, candidate_set = self.metric.eval_save(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        with open('candidate.txt', 'w') as file:
            json.dump(candidate_set, file, ensure_ascii=False, indent=4)
        return eval_result, candidate_set

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            if not configs['tune']['enable']:
                save_dir_path = './encoder/checkpoint/{}'.format(model_name)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                torch.save(model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed']))
                self.logger.log("Save model parameters to {}".format('{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed'])))
            else:
                save_dir_path = './encoder/checkpoint/{}/tune'.format(model_name)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))

class AutoCFTrainer(Trainer):
    def __init__(self, data_handler, logger):
        super(AutoCFTrainer, self).__init__(data_handler, logger)
        self.fix_steps = configs['model']['fix_steps']

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        model.train()
        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))

            if i % self.fix_steps == 0:
                sampScores, seeds = model.sample_subgraphs()
                encoderAdj, decoderAdj = model.mask_subgraphs(seeds)

            loss, loss_dict = model.cal_loss(batch_data, encoderAdj, decoderAdj)

            if i % self.fix_steps == 0:
                localGlobalLoss = -sampScores.mean()
                loss += localGlobalLoss
                loss_dict['infomax_loss'] = localGlobalLoss

            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        # writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)


class AdaGCLTrainer(Trainer):
    def __init__(self, data_handler, logger):
        from encoder.models.general_cf.adagcl import VGAE, DenoiseNet
        super(AdaGCLTrainer, self).__init__(data_handler, logger)
        self.generator_1 = VGAE().cuda()
        self.generator_2 = DenoiseNet().cuda()

    def create_optimizer(self, model):
        self.generator_1.set_adagcl(model)
        self.generator_2.set_adagcl(model)
        model.set_denoiseNet(self.generator_2)

        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'],
                                        weight_decay=optim_config['weight_decay'])
            self.optimizer_gen_1 = optim.Adam(self.generator_1.parameters(), lr=optim_config['lr'],
                                              weight_decay=optim_config['weight_decay'])
            self.optimizer_gen_2 = optim.Adam(filter(lambda p: p.requires_grad, self.generator_2.parameters()),
                                              lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def generator_generate(self, model):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.data_handler.torch_adj)
        idxs = adj._indices()

        with torch.no_grad():
            view = model.vgae_generate(self.data_handler.torch_adj, idxs, adj)

        return view

    def train_epoch(self, model, epoch_idx):
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        model.train()
        for _, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
            self.optimizer.zero_grad()
            self.optimizer_gen_1.zero_grad()
            self.optimizer_gen_2.zero_grad()

            temperature = max(0.05, configs['model']['init_temperature'] * pow(configs['model']['temperature_decay'],
                                                                               epoch_idx))

            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            data1 = self.generator_generate(self.generator_1)

            loss_cl, loss_dict_cl, out1, out2 = model.cal_loss_cl(batch_data, data1)
            ep_loss += loss_cl.item()
            loss_cl.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_ib, loss_dict_ib = model.cal_loss_ib(batch_data, data1, out1, out2)
            ep_loss += loss_ib.item()
            loss_ib.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_main, loss_dict_main = model.cal_loss(batch_data)
            ep_loss += loss_main.item()
            loss_main.backward()

            loss_vgae, loss_dict_vgae = self.generator_1.cal_loss_vgae(self.data_handler.torch_adj, batch_data)
            loss_denoise, loss_dict_denoise = self.generator_2.cal_loss_denoise(batch_data, temperature)
            loss_generator = loss_vgae + loss_denoise
            ep_loss += loss_generator.item()
            loss_generator.backward()

            self.optimizer.step()
            self.optimizer_gen_1.step()
            self.optimizer_gen_2.step()

            loss_dict = {**loss_dict_cl, **loss_dict_ib, **loss_dict_main, **loss_dict_vgae, **loss_dict_denoise}
            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)



