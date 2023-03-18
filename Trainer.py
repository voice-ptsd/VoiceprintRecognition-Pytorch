import json
import math
import os
import platform
import shutil
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram
from torchinfo import summary
from tqdm import tqdm

from mvector.data_utils.audio import AudioSegment
from mvector.data_utils.collate_fn import collate_fn
from mvector.data_utils.utils import make_pad_mask
from mvector.models.ecapa_tdnn import EcapaTdnn, SpeakerIdetification
from mvector.utils.logger import setup_logger

logger = setup_logger(__name__)


class MelSpectrogramFeaturizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=320, win_length=1024, f_min=0,
                                        f_max=16000, n_mels=64)

    def forward(self, waveforms, input_lens_ratio):
        """从AudioSegment中提取音频特征

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: list
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        feature = self.extractor(waveforms)
        feature = feature.transpose(2, 1)
        # 归一化
        mean = torch.mean(feature, 1, keepdim=True)
        std = torch.std(feature, 1, keepdim=True)
        feature = (feature - mean) / (std + 1e-5)
        input_lens = input_lens_ratio * feature.shape[1]
        input_lens = input_lens.int()
        masks = ~make_pad_mask(input_lens).int().unsqueeze(-1)
        feature = feature * masks
        return feature, input_lens

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        return 64


class VoiceDataset(Dataset):
    def __init__(self, mode='train'):
        super(VoiceDataset, self).__init__()
        self.SAMPLE_RATE = 16000
        self.CHUNK_DURATION = 3
        self.MIN_DURATION = 0.5
        self.NORMALIZE_DB = True
        self.TARGET_DB = -20
        self.DO_VAD = True

        self.min_samples = int(self.MIN_DURATION * self.SAMPLE_RATE)
        self.min_chunk_samples = int(self.CHUNK_DURATION * self.SAMPLE_RATE)
        self.mode = mode
        # self.epoch_id = epoch_id
        self.data_list: list[list[str, int]] = []

        data = json.load(open('dataset/data.json', 'r', encoding='utf-8'))

        # i1 = (epoch_id % 10) / 10
        # i2 = (epoch_id % 10 + 2) / 10
        # for i in data['sounds'].keys():
        #     c = data['sounds'][i]['sounds_count']
        #     c1 = math.ceil(c * i1)
        #     c2 = math.ceil(c * i2)
        #     train_list = [[sound, i] for sound in data['sounds'][i]['sounds'][0:c1]] + \
        #                  [[sound, i] for sound in data['sounds'][i]['sounds'][c2:c]]
        #     test_list = [[sound, i] for sound in data['sounds'][i]['sounds'][c1:c2]]
        #     assert len(train_list) + len(test_list) == c, \
        #         "!!数据集划分不完整 epoch: %d, speaker: %s, count: %d, train: %d, test: %d!!" % \
        #         (epoch_id, data['speakers'][i], c, len(train_list), len(test_list))
        #     if mode == 'train':
        #         self.data_list.extend(train_list)
        #     elif mode == 'test':
        #         self.data_list.extend(test_list)
        for i in data['sounds'].keys():
            c = data['sounds'][i]['sounds_count']
            s = math.ceil(c * 0.8)
            train_list = [[sound, i] for sound in data['sounds'][i]['sounds'][0:s]]
            test_list = [[sound, i] for sound in data['sounds'][i]['sounds'][s:c]]
            assert len(train_list) + len(test_list) == c, \
                "!!数据集划分不完整speaker: %s, count: %d, train: %d, test: %d!!" % \
                (data['speakers'][i], c, len(train_list), len(test_list))
            if mode == 'train':
                self.data_list.extend(train_list)
            elif mode == 'test':
                self.data_list.extend(test_list)

    def __getitem__(self, index):
        # 解析出音频路径和说话人标签
        [audio_path, speaker_label] = self.data_list[index]
        # 读取音频
        audio_segment = AudioSegment.from_file(audio_path)
        # 裁剪静音
        if self.DO_VAD:
            audio_segment.vad()
        if self.mode == 'train' and audio_segment.num_samples < self.min_samples:
            return self.__getitem__(index + 1 if index + 1 < len(self.data_list) else 0)
        # 重采样
        pass
        # 平衡响度
        if self.NORMALIZE_DB:
            audio_segment.normalize(target_db=self.TARGET_DB)
        # 数据增强
        # 随机修改响度
        # if random.random() < 0.5:
        #     # "min_gain_dBFS": -15,
        #     # "max_gain_dBFS": 15
        #     gain = random.uniform(-15, 15)
        #     audio_segment.gain_db(gain)
        # 不足长度不够的音频
        if audio_segment.num_samples < self.min_chunk_samples:
            shortage = self.min_chunk_samples - audio_segment.num_samples
            audio_segment.pad_silence(duration=float(shortage / audio_segment.sample_rate * 1.1), sides='end')
        # 裁剪需要的数据
        audio_segment.crop(duration=self.CHUNK_DURATION, mode=self.mode)
        return np.array(audio_segment.samples, dtype=np.float32), np.array(int(speaker_label), dtype=np.int64)

    def __len__(self):
        return len(self.data_list)


class AdditiveAngularMargin(torch.nn.Module):
    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        """The Implementation of Additive Angular Margin (AAM) proposed
       in the following paper: '''Margin Matters: Towards More Discriminative
       Deep Neural Network Embeddings for Speaker Recognition'''
       (https://arxiv.org/abs/1906.07317)

        Args:
            margin (float, optional): margin factor. Defaults to 0.0.
            scale (float, optional): scale factor. Defaults to 1.0.
            easy_margin (bool, optional): easy_margin flag. Defaults to False.
        """
        super(AdditiveAngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        cosine = outputs.float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class AAMLoss(torch.nn.Module):
    def __init__(self, margin=0.2, scale=30, easy_margin=False):
        super(AAMLoss, self).__init__()
        self.loss_fn = AdditiveAngularMargin(margin=margin, scale=scale, easy_margin=easy_margin)
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets):
        targets = F.one_hot(targets, outputs.shape[1]).float()
        predictions = self.loss_fn(outputs, targets)
        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss


class Trainer:
    def __init__(self):
        self.BATCH_SIZE = 100
        self.NUM_SPEAKERS = 600
        self.NUM_WORKERS = 16
        self.TRAIN_LIST = 'dataset/train_list.txt'
        self.TEST_LIST = 'dataset/test_list.txt'

        self.FEATURE_METHOD = 'MelSpectrogram'

        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 1e-6

        self.EMBD_DIM = 192
        self.CHANNELS = 512

        self.MAX_EPOCH = 30
        self.LOG_INTERVAL = 100

        self.USE_MODEL = 'ecapa_tdnn'

        self.SAVE_MODEL_PATH = 'models/'

        if platform.system().lower() == 'windows':
            self.NUM_WORKERS = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')

        assert torch.cuda.is_available(), "No CUDA device found."
        self.device = torch.device("cuda")

        # 获取特征器
        self.audio_featurizer = MelSpectrogramFeaturizer()
        self.audio_featurizer.to(self.device)

        # 获取模型
        self.ecapa_tdnn = EcapaTdnn(input_size=self.audio_featurizer.feature_dim, embd_dim=self.EMBD_DIM,
                                    channels=self.CHANNELS)
        self.model = SpeakerIdetification(backbone=self.ecapa_tdnn, num_class=self.NUM_SPEAKERS)
        self.model.to(self.device)
        summary(self.model, (1, 98, self.audio_featurizer.feature_dim))

        # 获取训练数据
        self.train_loader = DataLoader(VoiceDataset(mode='train'), collate_fn=collate_fn, shuffle=True,
                                       batch_size=self.BATCH_SIZE,
                                       sampler=None, num_workers=self.NUM_WORKERS)

        # 获取测试数据
        self.test_loader = DataLoader(VoiceDataset(mode='test'), collate_fn=collate_fn, batch_size=self.BATCH_SIZE,
                                      num_workers=self.NUM_WORKERS)

        # 获取损失函数
        self.loss = AAMLoss()

        # 获取优化方法(AdamW)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=float(self.LEARNING_RATE),
                                           weight_decay=float(self.WEIGHT_DECAY))
        # 学习率衰减函数
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=int(self.MAX_EPOCH * 1.2))

    def train(self, resume_model=None, pretrained_model=None):
        # 加载预训练模型
        if pretrained_model is not None:
            assert os.path.exists(os.path.join(pretrained_model, 'model.pt')), "模型参数文件不存在！"
            model_dict = self.model.state_dict()
            model_state_dict = torch.load(os.path.join(pretrained_model, 'model.pt'))
            # 过滤不存在的参数
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            self.model.load_state_dict(model_state_dict, strict=False)
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))

        last_epoch = -1
        best_accuracy = 0

        # 恢复训练
        if resume_model is not None:
            assert os.path.exists(os.path.join(resume_model, 'model.pt')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pt')), "优化方法参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'model.state')), "模型状态文件不存在！"
            self.model.load_state_dict(torch.load(os.path.join(resume_model, 'model.pt')))
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_accuracy = json_data['accuracy']
            if last_epoch > 0:
                self.optimizer.step()
                [self.scheduler.step() for _ in range(last_epoch)]
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))

        # 开始训练
        test_step, train_step = 0, 0
        last_epoch += 1
        for epoch_id in range(last_epoch, self.MAX_EPOCH):
            epoch_id += 1
            start_epoch = time.time()
            # 训练一个epoch
            train_times, accuracies, loss_sum = [], [], []
            start = time.time()
            sum_batch = len(self.train_loader) * self.MAX_EPOCH
            for batch_id, (audio, label, input_lens_ratio) in enumerate(self.train_loader):
                audio = audio.to(self.device)
                input_lens_ratio = input_lens_ratio.to(self.device)
                label = label.to(self.device).long()
                features, _ = self.audio_featurizer(audio, input_lens_ratio)
                output = self.model(features)
                # 计算损失值
                loss = self.loss(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 计算准确率
                output = torch.nn.functional.softmax(output, dim=-1)
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                accuracies.append(acc)
                loss_sum.append(loss)
                train_times.append((time.time() - start) * 1000)

                # 打印进度
                if batch_id % self.LOG_INTERVAL == 0:
                    # 计算每秒训练数据量
                    train_speed = self.BATCH_SIZE / (sum(train_times) / len(train_times) / 1000)
                    # 计算剩余时间
                    eta_sec = (sum(train_times) / len(train_times)) * (
                            sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                    eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                    logger.info(f'Train epoch: [{epoch_id}/{self.MAX_EPOCH}], '
                                f'batch: [{batch_id}/{len(self.train_loader)}], '
                                f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                                f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                                f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                                f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                    train_step += 1
                    train_times = []
                # 固定步数也要保存一次模型
                # if batch_id % 10000 == 0 and batch_id != 0 and local_rank == 0:
                #     self.save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id)
                start = time.time()
            self.scheduler.step()
            logger.info('=' * 70)
            loss, accuracy, precision, recall, f1_score = self.evaluate()
            logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}, precision: {:.5f}, '
                        'recall: {:.5f}, f1_score: {:.5f}'.format(epoch_id,
                                                                  str(timedelta(seconds=(time.time() - start_epoch))),
                                                                  loss, accuracy, precision, recall, f1_score))
            logger.info('=' * 70)
            test_step += 1
            self.model.train()

            # 保存模型
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                model_path = os.path.join(self.SAVE_MODEL_PATH, f'{self.USE_MODEL}_{self.FEATURE_METHOD}', 'best_model')
            else:
                model_path = os.path.join(self.SAVE_MODEL_PATH,
                                          f'{self.USE_MODEL}_{self.FEATURE_METHOD}', 'epoch_{}'.format(epoch_id))
            # 写入
            os.makedirs(model_path, exist_ok=True)
            torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
            torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pt'))
            with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
                f.write('{"last_epoch": %d, "accuracy": %f}' % (epoch_id, best_accuracy))
            # 复制一份最后一次模型
            last_model_path = os.path.join(self.SAVE_MODEL_PATH, f'{self.USE_MODEL}_{self.FEATURE_METHOD}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型，保留最后三份
            old_model_path = os.path.join(self.SAVE_MODEL_PATH, f'{self.USE_MODEL}_{self.FEATURE_METHOD}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
            logger.info('已保存模型：{}'.format(model_path))

    def evaluate(self):
        accuracies, losses, predictions, r_labels = [], [], [], []
        features, labels = None, None
        with torch.no_grad():
            for batch_id, (audio, label, input_lens_ratio) in enumerate(tqdm(self.test_loader)):
                audio = audio.to(self.device)
                input_lens_ratio = input_lens_ratio.to(self.device)
                label = label.to(self.device).long()
                audio_features, _ = self.audio_featurizer(audio, input_lens_ratio)
                output = self.model(audio_features)
                feature = self.model.backbone(audio_features).data.cpu().numpy()
                loss = self.loss(output, label)
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                # 模型预测标签
                prediction = np.argmax(output, axis=1)
                predictions.extend(prediction.tolist())
                # 真实标签
                r_labels.extend(label.tolist())
                # 计算准确率
                acc = np.mean((prediction == label).astype(int))
                accuracies.append(acc)
                losses.append(loss.data.cpu().numpy())
                # 存放特征
                features = np.concatenate((features, feature)) if features is not None else feature
                labels = np.concatenate((labels, label)) if labels is not None else label
        loss = float(sum(losses) / len(losses))
        acc = float(sum(accuracies) / len(accuracies))
        self.model.train()
        # 计算精确率、召回率、f1_score
        cm = confusion_matrix(labels, predictions)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        # 精确率
        precision = TP / (TP + FP + 1e-6)
        # 召回率
        recall = TP / (TP + FN + 1e-6)
        f1_score = (2 * precision * recall) / (precision + recall + 1e-12)
        return loss, acc, np.mean(precision), np.mean(recall), np.mean(f1_score)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    # trainer.train(resume_model="models/ecapa_tdnn_MelSpectrogram/last_model")
    # trainer.train(pretrained_model='models/ecapa_tdnn_MelSpectrogram_30epoch/last_model')
