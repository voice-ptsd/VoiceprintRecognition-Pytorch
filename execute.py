import io
import json
import os
import platform
import shutil
import time
from datetime import timedelta

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from visualdl import LogWriter

from DictClass import DictClass
from mvector import SUPPORT_MODEL
from mvector.data_utils.collate_fn import collate_fn
from mvector.data_utils.featurizer import AudioFeaturizer
from mvector.data_utils.reader import CustomDataset
from mvector.models.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from mvector.models.loss import AAMLoss
from mvector.utils.logger import setup_logger
from mvector.utils.utils import dict_to_object, cal_accuracy_threshold, print_arguments


def save_checkpoint(configs, model, save_model_path, epoch_id, best_acc=0., best_model=False):
    state_dict = model.state_dict()
    if best_model:
        model_path = os.path.join(save_model_path,
                                  f'{configs.use_model}_{configs.preprocess_conf.feature_method}',
                                  'best_model')
    else:
        model_path = os.path.join(save_model_path,
                                  f'{configs.use_model}_{configs.preprocess_conf.feature_method}',
                                  'epoch_{}'.format(epoch_id))
    os.makedirs(model_path, exist_ok=True)
    torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
    torch.save(state_dict, os.path.join(model_path, 'model.pt'))
    with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
        f.write('{"last_epoch": %d, "accuracy": %f}' % (epoch_id, best_acc))
    if not best_model:
        last_model_path = os.path.join(save_model_path,
                                       f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                       'last_model')
        shutil.rmtree(last_model_path, ignore_errors=True)
        shutil.copytree(model_path, last_model_path)
        # 删除旧的模型
        old_model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'epoch_{}'.format(epoch_id - 3))
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)
    logger.info('已保存模型：{}'.format(model_path))


logger = setup_logger(__name__)
device = torch.device("cuda")

config_file = open('configs/ecapa_tdnn.yml', 'r', encoding='utf-8')
configs = yaml.load(config_file.read(), Loader=yaml.FullLoader)
print_arguments(configs=configs)
configs = DictClass(configs)
# 获取特征器
audio_featurizer = AudioFeaturizer()
audio_featurizer.to(device)

if platform.system().lower() == 'windows':
    configs.dataset_conf.num_workers = 0
    logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')

writer = LogWriter(logdir='log')
# 获取数据
train_dataset = CustomDataset(data_list_path='dataset/train_list.txt', mode='train')
train_sampler = None
train_loader = DataLoader(dataset=train_dataset,
                          collate_fn=collate_fn,
                          shuffle=(train_sampler is None),
                          batch_size=configs.dataset_conf.batch_size,
                          sampler=train_sampler,
                          num_workers=configs.dataset_conf.num_workers)
# 获取测试数据
test_dataset = CustomDataset(data_list_path='dataset/test_list.txt', mode='eval')
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=configs.dataset_conf.batch_size,
                         collate_fn=collate_fn,
                         num_workers=configs.dataset_conf.num_workers)

# 获取模型
ecapa_tdnn = EcapaTdnn(audio_featurizer.feature_dim)
model = SpeakerIdetification(backbone=ecapa_tdnn, num_class=configs.dataset_conf.num_speakers)
model.to(device)

summary(model, (1, 98, audio_featurizer.feature_dim))
# print(self.model)
# 获取损失函数
loss = AAMLoss()
# 获取优化方法
optimizer = torch.optim.AdamW(params=model.parameters(),
                              lr=float(configs.optimizer_conf.learning_rate),
                              weight_decay=float(configs.optimizer_conf.weight_decay))

# 学习率衰减函数
scheduler = CosineAnnealingLR(optimizer, T_max=int(configs.train_conf.max_epoch * 1.2))
pretrained_model = None

# 加载恢复模型
last_epoch = -1
best_acc = 0
last_model_dir = os.path.join('models/',
                              f'{configs.use_model}_{configs.preprocess_conf.feature_method}',
                              'last_model')
if os.path.exists(os.path.join(last_model_dir, 'model.pt')) and os.path.exists(
        os.path.join(last_model_dir, 'optimizer.pt')):
    # 自动获取最新保存的模型
    resume_model = last_model_dir
    assert os.path.exists(os.path.join(resume_model, 'model.pt')), "模型参数文件不存在！"
    assert os.path.exists(os.path.join(resume_model, 'optimizer.pt')), "优化方法参数文件不存在！"
    state_dict = torch.load(os.path.join(resume_model, 'model.pt'))
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pt')))
    with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    last_epoch = json_data['last_epoch'] - 1
    best_acc = json_data['accuracy']
    logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))

    if last_epoch > 0:
        optimizer.step()
        [scheduler.step()
         for _ in range(last_epoch)]

    test_step, train_step = 0, 0
    last_epoch += 1
    writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], last_epoch)

    # 开始训练
    for epoch_id in range(last_epoch, configs.train_conf.max_epoch):
        epoch_id += 1
        start_epoch = time.time()

        # 训练一个epoch

        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        sum_batch = len(train_loader) * configs.train_conf.max_epoch
        for batch_id, (audio, label, input_lens_ratio) in enumerate(train_loader):
            audio = audio.to(device)
            input_lens_ratio = input_lens_ratio.to(device)
            label = label.to(device).long()
            features, _ = audio_featurizer(audio, input_lens_ratio)
            output = model(features)
            # 计算损失值
            los = loss(output, label)
            optimizer.zero_grad()
            los.backward()
            optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            train_times.append((time.time() - start) * 1000)

            # 多卡训练只使用一个进程打印
            if batch_id % configs.train_conf.log_interval == 0:
                # 计算每秒训练数据量
                train_speed = configs.dataset_conf.batch_size / (sum(train_times) / len(train_times) / 1000)
                # 计算剩余时间
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch: [{epoch_id}/{configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                            f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                            f'learning rate: {scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), train_step)
                writer.add_scalar('Train/Accuracy', (sum(accuracies) / len(accuracies)), train_step)
                # 记录学习率
                writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], train_step)
                self.train_step += 1
                train_times = []
            # 固定步数也要保存一次模型
            if batch_id % 10000 == 0 and batch_id != 0:
                save_checkpoint(save_model_path='models/', epoch_id=epoch_id)
            start = time.time()
        self.scheduler.step()

        # 多卡训练只使用一个进程执行评估和保存模型
        logger.info('=' * 70)
        loss, acc, precision, recall, f1_score = self.evaluate(resume_model=None)
        logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}, precision: {:.5f}, '
                    'recall: {:.5f}, f1_score: {:.5f}'.format(epoch_id, str(timedelta(
            seconds=(time.time() - start_epoch))), loss, acc, precision, recall, f1_score))
        logger.info('=' * 70)
        writer.add_scalar('Test/Accuracy', acc, test_step)
        writer.add_scalar('Test/Loss', loss, test_step)
        test_step += 1
        model.train()
        # # 保存最优模型
        if acc >= best_acc:
            best_acc = acc
        save_checkpoint(save_model_path='models/', epoch_id=epoch_id, best_acc=acc,
                        best_model=True)
        # 保存模型
        save_checkpoint(save_model_path='models/', epoch_id=epoch_id, best_acc=acc)
