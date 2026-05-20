#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch鸟类识别训练脚本
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import yaml
import shutil
import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.cuda.amp import autocast, GradScaler

import timm 
from timm.data import create_transform
from timm.utils import ModelEmaV2

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
from easydict import EasyDict as edict

class BirdDataset:
    """鸟类数据集类"""
    
    def __init__(self, config):
        self.config = config
        self.input_size = tuple(config.input_size)
        self.rgb_mean = config.rgb_mean
        self.rgb_std = config.rgb_std
        
    def get_transforms(self, trans_type="train"):
        """获取数据增强变换"""
        if trans_type == "train":
            transform = transforms.Compose([
                #修改为BICUBIC
                transforms.Resize([self.input_size[1], self.input_size[0]],interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                transforms.ToTensor(),
                #随机擦除需要放在totensor之后
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize([self.input_size[1], self.input_size[0]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std),
            ])
        return transform

    def get_transforms_timm(self, trans_type="train"):
        """获取数据增强变换"""
        if trans_type == "train":
            # 全图训练
            # transform = create_transform(
            #     input_size=self.input_size,          # 例如 (380, 380)
            #     is_training=True,
            #     color_jitter=0.2,                    # 颜色抖动，随机改变亮度、对比度、饱和度、色调（幅度 20%）
            #     auto_augment='rand-m6-mstd0.3-inc1', # 【核心】RandAugment随机组合 6 种几何/颜色变换（如旋转、平移、剪切、锐化等）
            #     re_prob=0.1,                         # 【核心】Random Erasing有 10% 的概率，在图中随机画一个矩形块并填充像素（遮挡）。
            #     re_mode='pixel',                     # 像素擦除模式
            #     re_count=1,                          # 擦除次数
            #     interpolation='bicubic',             # 【核心】双三次插值，保留鸟类纹理细节
            #     mean=self.rgb_mean,                  # 使用配置文件中的均值
            #     std=self.rgb_std,                    # 使用配置文件中的标准差
            #     crop_pct=0.95,                       # 取图片中心的 95% 区域，再缩放到网络大小。
            # )

            #cut图训练
            transform = create_transform(
                input_size=self.input_size,          # 建议改为 (224, 224) 或 (256, 256)，不要强行放大到 380
                is_training=True,
                color_jitter=0.4,                    # 【调高】0.2 -> 0.4。因为背景少了，需靠颜色变化防止过拟合纹理
                auto_augment='rand-m5-mstd0.3-inc1', # 【减弱】rand-m6 -> rand-m5。减少几何形变幅度，防止鸟头/尾巴被切歪
                re_prob=0.25,                        # 【调高】0.1 -> 0.25。强制模型学习鸟的全身特征，而不是只盯着某一块羽毛
                re_mode='pixel',                     # 像素擦除模式
                re_count=2,                          # 【调高】擦除次数1 -> 2。小图信息量少，多遮挡几次能极大提升鲁棒性
                interpolation='bicubic',             # 【核心】双三次插值，保留鸟类纹理细节
                mean=self.rgb_mean,                  # 使用配置文件中的均值
                std=self.rgb_std,                    # 使用配置文件中的标准差
                # crop_pct=1.0,                        # 【关键】紧贴框cut小图
                crop_pct=0.95,                       # 【关键】外扩30%图，可以适当裁剪
            )
        else:
            # 验证集变换：保持确定性，仅 Resize + CenterCrop + Normalize
            transform = create_transform(
                input_size=self.input_size,
                is_training=False,
                interpolation='bicubic',
                # crop_pct=0.95,
                crop_pct=1.0,                        # 针对小图        
                mean=self.rgb_mean,
                std=self.rgb_std,
            )
            
        return transform
    
    def get_datasets(self):
        """获取训练和测试数据集"""
        # train_transform = self.get_transforms("train")
        # test_transform = self.get_transforms("test")
        #使用强增强模式
        train_transform = self.get_transforms_timm("train")
        test_transform = self.get_transforms_timm("test")

        # 加载训练数据
        train_datasets = []
        print('loading train data...')
        for train_path in self.config.train_data:
            if os.path.exists(train_path):
                train_dataset = ImageFolder(train_path, transform=train_transform)
                train_datasets.append(train_dataset)
        if len(train_datasets) > 1:
            train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        else:
            train_dataset = train_datasets[0] if train_datasets else None
            
        # 加载测试数据
        test_dataset = None
        if os.path.exists(self.config.test_data):
            test_dataset = ImageFolder(self.config.test_data, transform=test_transform)
            
        return train_dataset, test_dataset

class BirdClassifier:
    """鸟类分类器"""
    
    def __init__(self, config):
        self.config = config
        # 自动检测可用 GPU，不再硬编码 gpu_id[0]，提高兼容性
        if torch.cuda.is_available():
            if hasattr(config, 'gpu_id') and config.gpu_id:
                self.device = torch.device(f"cuda:{config.gpu_id[0]}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.num_classes = self._get_num_classes()
        
    def _get_num_classes(self):
        """获取类别数量"""
        with open(self.config.class_name, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return len(classes)
    
    def build_model(self):
        """构建模型"""
        model = None
        net_type = self.config.net_type
        model_kwargs = {
            "num_classes": self.num_classes,
            "pretrained": self.config.pretrained,
            "drop_rate": self.config.drop_rate,
            "drop_path_rate": self.config.drop_path_rate,
        }

        try:
            # 尝试使用 timm 加载
            print(f"正在通过 timm 加载模型：{net_type} ...")
            model = timm.create_model(net_type, **model_kwargs)
            if model is not None:
                model = model.to(self.device)
        except Exception as e:
            print(f"timm 加载 {net_type} 失败：{e}")
            print(f"使用原生 torchvision 或自定义模型...")
            
            if net_type == "resnet18":
                model = models.resnet34(pretrained=self.config.pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif net_type == "resnet34":
                model = models.resnet34(pretrained=self.config.pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif net_type == "resnet50":
                model = models.resnet50(pretrained=self.config.pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif net_type == "resnet152":
                model = models.resnet152(pretrained=self.config.pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif net_type == "mobilenet_v2":
                model = models.mobilenet_v2(pretrained=self.config.pretrained)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            elif net_type == "googlenet":
                model = models.googlenet(pretrained=self.config.pretrained)
                model.fc = nn.Linear(model.fc.in_features, self.num_classes)
            elif net_type == "efficientnet_b4":
                model = models.efficientnet_b4(pretrained=self.config.pretrained)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
            else:
                raise ValueError(f"Unsupported model type: {net_type}")
            
            if model is not None:
                model = model.to(self.device)

        return model
    
    def build_optimizer(self, model):
        """构建优化器"""
        if self.config.optim_type == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optim_type == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optim_type == "AdamW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optim_type}")
            
        return optimizer
    
    def build_scheduler(self, optimizer):
        """构建学习率调度器"""
        warm_up_epochs = self.config.get("num_warm_up", 0)
        if self.config.scheduler == "multi-step":   #里程碑式衰减
            base_scheduler  = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.config.milestones,
                gamma=0.1
            )
        elif self.config.scheduler == "cosine":     #余弦退火
            # 续训时，T_max = 总epoch - 已训练epoch - warm_up
            if hasattr(self, 'start_epoch'):  # 断点续训时
                T_max = self.config.num_epochs - self.start_epoch - warm_up_epochs
            else:  # 首次训练
                T_max = self.config.num_epochs - warm_up_epochs
            # 确保 T_max 至少为 1
            T_max = max(1, T_max)
            base_scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=1e-6
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
        
        if warm_up_epochs > 0:
            print(f"启用Warm-up策略：前{warm_up_epochs}个epoch线性预热学习率")
            # 预热调度器（线性从初始lr*0.1升到初始lr）
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,  # 预热起始学习率 = 初始lr * 0.1
                total_iters=warm_up_epochs  # 预热轮数
            )
            # 组合调度器：先预热，后执行基础调度器
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, base_scheduler],
                milestones=[warm_up_epochs]  # 预热结束后切换到基础调度器
            )
        else:
            scheduler = base_scheduler  # 无预热，直接使用基础调度器
        return scheduler
    
    def build_criterion(self):
        """构建损失函数"""
        smoothing = getattr(self.config, 'label_smoothing', 0.1)
        if smoothing > 0:
            print(f"Using Label Smoothing Cross Entropy (smoothing={smoothing})")
            return nn.CrossEntropyLoss(label_smoothing=smoothing)
        else:
            return nn.CrossEntropyLoss()

class Trainer:
    """训练器"""

    def __init__(self, config, resume_path=None):
        self.config = config
        # 设备逻辑与 Classifier 保持一致
        if torch.cuda.is_available():
            if hasattr(config, 'gpu_id') and config.gpu_id:
                self.device = torch.device(f"cuda:{config.gpu_id[0]}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.work_dir = config.work_dir
        self.resume_path = resume_path

        # 初始化组件
        self.classifier = BirdClassifier(config)
        self.dataset = BirdDataset(config)
        
        # 创建模型保存目录
        self.model_dir = os.path.join(self.work_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 创建日志目录
        self.log_dir = os.path.join(self.work_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)

        # 彻底清理日志配置，防止重复打印
        self.logger = logging.getLogger("BirdTrainer")
        self.logger.setLevel(logging.INFO)
        
        # 清除所有已有 handler
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        # 关键：禁止向 root logger 冒泡，防止被 root 的 handler 再次打印
        self.logger.propagate = False 

        # 创建控制台 Handler (屏幕输出)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 创建文件 Handler (保存到 log/train.log)
        log_file_path = os.path.join(self.log_dir, "train.log")
        fh = logging.FileHandler(log_file_path, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        # 添加 Handler 到 Logger
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        
        self.logger.info(f"训练器初始化完成。工作目录：{self.work_dir}")

        # 检查是否禁用 AMP
        self.use_amp = getattr(self.config, 'use_amp', True)
        if self.use_amp and torch.cuda.is_available():
            self.scaler = GradScaler()
            self.logger.info("已启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            self.logger.info("混合精度训练 (AMP) 已禁用，使用全精度 FP32")
        
        # 从配置读取梯度裁剪阈值，默认 1.0
        self.grad_clip = getattr(self.config, 'grad_clip', 1.0)
        # 从配置读取早停耐心值，默认 20
        self.patience = getattr(self.config, 'patience', 20)
        
        # EMA 配置
        self.use_ema = getattr(self.config, 'use_ema', True)
        self.ema_decay = getattr(self.config, 'ema_decay', 0.9999)
        self.model_ema = None
        
        # 用于绘制曲线的历史记录
        self.history = {'train_loss': [], 'val_loss': [], 'train_top1': [], 'val_top1': []}
        
    def calculate_accuracy(self, output, target, topk=(1, 5)):
        """
        计算 Top-k 准确率
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            # 防止 k 大于类别数
            maxk = min(maxk, output.size(1))
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            res = []
            for k in topk:
                if k > output.size(1): continue
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.item())
            return res

    def train_epoch(self, model, dataloader, criterion, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        # 通用获取类别数，避免访问 model.fc 报错 (Transformer 没有 fc 属性)
        num_classes = self.classifier.num_classes
        k_top5 = min(5, num_classes)
        topk_tuple = (1, k_top5)
        
        # 禁用 tqdm 的频繁刷新以提升 I/O 性能
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", dynamic_ncols=True)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # NaN/Inf 检测
            if not torch.isfinite(loss).all():
                self.logger.warning(f"Batch {batch_idx}: 检测到 Loss 非法 ({loss.item()}), 跳过此批次")
                optimizer.zero_grad()
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip)
                optimizer.step()
            
            # 更新 EMA 权重
            if self.model_ema is not None:
                self.model_ema.update(model)
            
            running_loss += loss.item()
            batch_size = targets.size(0)
            total += batch_size
            
            acc1, acc5 = self.calculate_accuracy(outputs, targets, topk=topk_tuple)
            top1_correct += acc1
            top5_correct += acc5
            
            if batch_idx % self.config.log_freq == 0:
                avg_loss = running_loss / (batch_idx + 1)
                curr_top1_acc = 100.0 * top1_correct / total
                curr_top5_acc = 100.0 * top5_correct / total
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Top1': f'{curr_top1_acc:.2f}%',
                    'Top5': f'{curr_top5_acc:.2f}%'
                })
                
        final_loss = running_loss / len(dataloader)
        final_top1_acc = 100.0 * top1_correct / total
        final_top5_acc = 100.0 * top5_correct / total
                
        return final_loss, final_top1_acc, final_top5_acc
    
    def validate(self, model, dataloader, criterion):
        """验证模型 (优先使用 EMA 模型)"""
        # 如果启用了 EMA，则使用 EMA 模型进行验证，否则使用原始模型
        eval_model = self.model_ema.module if self.model_ema is not None else model
        eval_model.eval()
        
        running_loss = 0.0
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        num_classes = self.classifier.num_classes
        k_top5 = min(5, num_classes)
        topk_tuple = (1, k_top5)
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validating", dynamic_ncols=True):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = eval_model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                batch_size = targets.size(0)
                total += batch_size
                acc_top1, acc_top5 = self.calculate_accuracy(outputs, targets, topk=topk_tuple)
                top1_correct += acc_top1
                top5_correct += acc_top5
                
        avg_loss = running_loss / len(dataloader)
        top1_acc = 100.0 * top1_correct / total
        top5_acc = 100.0 * top5_correct / total
        
        return avg_loss, top1_acc, top5_acc
    
    def load_checkpoint(self, model, optimizer, scheduler):
        """加载断点续训的checkpoint"""
        if not os.path.exists(self.resume_path):
            self.logger.error(f"Checkpoint文件不存在：{self.resume_path}")
            raise FileNotFoundError(f"Checkpoint文件不存在：{self.resume_path}")
        
        self.logger.info(f"\n从断点加载模型：{self.resume_path}")
        checkpoint = torch.load(self.resume_path, map_location=self.device)
        
        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"成功恢复模型权重 (训练至第 {checkpoint['epoch'] + 1} 个epoch)")
        
        # 恢复 EMA 权重
        if self.model_ema is not None and 'state_dict_ema' in checkpoint:
            self.model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            self.logger.info("成功恢复 EMA 权重状态")
        elif self.model_ema is not None:
            self.logger.warning("未找到 EMA 权重状态，将重新初始化 EMA (可能导致初期验证波动)")
            # 如果 checkpoint 中没有 EMA，但当前启用了 EMA，则基于加载的模型重新初始化
            self.model_ema.update(model) 
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("成功恢复优化器状态")
        
        # 恢复调度器状态
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("成功恢复学习率调度器状态")
        
        # 恢复最佳准确率
        best_acc = checkpoint.get('best_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始训练
        
        if 'scaler_state_dict' in checkpoint and self.scaler is not None and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.logger.info("成功恢复梯度缩放器状态")
        
        # 恢复历史数据用于绘图（如果存在）
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            self.logger.info("已恢复训练历史记录用于绘图")
        
        self.logger.info(f"恢复信息 - 起始 epoch: {start_epoch}, 最佳 Top1 准确率：{best_acc:.2f}%")
        if 'val_top1' in checkpoint:
            self.logger.info(f"上次验证Top1准确率：{checkpoint['val_top1']:.2f}%, Top5准确率：{checkpoint['val_top5']:.2f}%")
        
        return model, optimizer, scheduler, start_epoch, best_acc
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_top1, val_top5, best_acc, is_best):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # 保存梯度缩放器状态
            'val_top1': val_top1,
            'val_top5': val_top5,
            'best_acc': best_acc,
            'config': self.config,
            'history': self.history # 保存历史用于绘图
        }
        
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存 EMA 权重
        if self.model_ema is not None:
            checkpoint['state_dict_ema'] = self.model_ema.module.state_dict()
        
        last_path = os.path.join(self.model_dir, 'last.pth')
        torch.save(checkpoint, last_path)
        
        # 2. 保存最佳模型 (best.pth) - 仅在精度提升时覆盖
        if is_best:
            best_path = os.path.join(self.model_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"✨ ep{epoch+1}发现新最佳模型 (Top1: {val_top1:.2f}%) -> 已保存至 best.pth")
        
        return best_acc
    
    def plot_curves(self):
        """绘制训练曲线"""
        if not self.history['train_loss']:
            return
            
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss 曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'loss_curve.png'))
        plt.close()
        
        # Accuracy 曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.history['train_top1'], 'b-', label='Training Top1 Acc')
        plt.plot(epochs, self.history['val_top1'], 'r-', label='Validation Top1 Acc')
        plt.title('Training and Validation Accuracy (Top1)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'acc_curve.png'))
        plt.close()
        
        self.logger.info(f"训练曲线已保存至：{self.log_dir}/loss_curve.png 和 acc_curve.png")

    def train(self):
        """主训练函数"""
        self.logger.info("开始训练鸟类识别模型...")
        self.logger.info(f"使用设备：{self.device}")
        self.logger.info(f"当前模型架构：{self.config.net_type}")
        self.logger.info(f"梯度裁剪阈值：{self.grad_clip}, 早停耐心值：{self.patience}")
        if self.use_ema:
            self.logger.info(f"EMA 已启用 (Decay: {self.ema_decay})")
        if not self.use_amp:
            self.logger.warning("警告：未启用混合精度训练，训练速度可能较慢")
        
        # 获取数据
        train_dataset, test_dataset = self.dataset.get_datasets()
        if train_dataset is None:
            self.logger.error("错误：未找到训练数据")
            return
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=2 if self.config.num_workers > 0 else None
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=2 if self.config.num_workers > 0 else None
        ) if test_dataset else None
        
        # 构建模型
        model = self.classifier.build_model()
        # 在模型建立后初始化 EMA，并传入模型
        if self.use_ema:
            self.model_ema = ModelEmaV2(model, decay=self.ema_decay, device=None)
            self.logger.info(f"已启用 EMA (decay={self.ema_decay}), 将在 CPU 上维护移动平均权重")
        criterion = self.classifier.build_criterion()
        optimizer = self.classifier.build_optimizer(model)
        scheduler = self.classifier.build_scheduler(optimizer)
        
        start_epoch = 0
        best_acc = 0.0
        patience_counter = 0
        # 如果指定了resume路径，加载checkpoint
        if self.resume_path:
            model, optimizer, scheduler, start_epoch, best_acc = self.load_checkpoint(
                model, optimizer, scheduler
            )

        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"模型总参数量：{total_params / 1e6:.2f} M")
        
        # 训练循环  
        start_time = time.time()

        for epoch in range(start_epoch, self.config.num_epochs):
            # 训练
            train_loss, train_top1, train_top5 = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_top1'].append(train_top1)
            
            if test_loader:
                val_loss, val_top1, val_top5 = self.validate(model, test_loader, criterion)
            else:
                val_loss, val_top1, val_top5 = train_loss, train_top1, train_top5
            
            # 记录历史
            self.history['val_loss'].append(val_loss)
            self.history['val_top1'].append(val_top1)
                
            scheduler.step()
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Top1: {train_top1:.2f}% , Train Top5: {train_top5:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Top1: {val_top1:.2f}% , Val Top5: {val_top5:.2f}%"
            )
            
            # 保存检查点,判断当前模型是否优于历史最佳
            # 注意：如果是第一个epoch (epoch==start_epoch)，建议强制视为最佳以便保存初始基准
            is_best = False
            if epoch == start_epoch:
                is_best = True
                best_acc = val_top1
                patience_counter = 0
            elif val_top1 > best_acc:
                is_best = True
                best_acc = val_top1
                patience_counter = 0
            else:
                is_best = False
                patience_counter += 1


            best_acc = self.save_checkpoint(model, optimizer, scheduler, epoch, val_top1, val_top5, best_acc, is_best)

            # 使用从 config 读取的 patience
            if patience_counter >= self.patience:
                self.logger.warning(f"早停：连续{self.patience}轮Val Top1无提升，最佳准确率：{best_acc:.2f}%")
                break

        # 保存最终模型
        final_path = os.path.join(self.model_dir, 'final_model.pth')
        final_checkpoint = {
            'epoch': self.config.num_epochs,
            'model_state_dict': model.state_dict(),
            'state_dict_ema': self.model_ema.module.state_dict() if self.model_ema else None,
            'best_acc': best_acc,
            'config': self.config,
            'history': self.history
        }
        if self.use_amp and self.scaler:
            final_checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(final_checkpoint, final_path)
        
        # 绘制曲线
        self.plot_curves()
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成！总耗时：{total_time:.2f}秒")
        self.logger.info(f"最佳验证准确率 (Top1): {best_acc:.2f}%")
        self.logger.info(f"模型文件保存在：{self.model_dir}")
        self.logger.info(f"日志文件保存在：{self.log_dir}")

def load_config(config_file):
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return edict(config)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch鸟类识别训练')
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='配置文件路径')
    # 新增resume参数
    parser.add_argument('--resume', type=str, default=None,
                        help='断点续训的checkpoint文件路径 (例如：./model/best_model_ep10_85.50_95.20.pth)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # --- 清空logger配置 ---
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # 清除所有已有的 Handler
    root_logger.setLevel(logging.INFO) # 重置级别
    # ---------------------------------------

    # 为 main 函数配置临时日志，以便记录启动信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Main")
    
    logger.info(f"工作目录为：{config.work_dir}")
    os.makedirs(config.work_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(config.work_dir, 'config.yaml'))
    logger.info(f"配置文件已复制到：{os.path.join(config.work_dir, 'config.yaml')}")

    # 开始训练
    trainer = Trainer(config, resume_path=args.resume)
    trainer.train()

if __name__ == "__main__":
    main()