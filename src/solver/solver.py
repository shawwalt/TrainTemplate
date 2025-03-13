import sys
source_root = "/mnt/disk2/Shawalt/Demos/ImageFusion/TrainTemplate/src"
sys.path.append(source_root)

import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms, datasets
from models.builder import *
from utils.transformation import *
from datasets.builder import *
from losses.builder import *
from metrics.builder import *
from utils.logger import Logger
from datetime import datetime
import logging
import statistics

class Solver:
    def __init__(self, config_path):
        """
        初始化Solver类，用于加载配置文件并设置模型、优化器等。
        参数:
        - config_path: 配置文件路径
        """
        # 加载配置文件
        with open(config_path, "r") as f:
            self.config = json.load(f)
            self.config_name = os.path.basename(config_path)

        # 初始化配置
        self.model_config = self.config["model"]
        self.training_config = self.config["training"]
        self.optimizer_config = self.config["optimizer"]
        self.scheduler_config = self.config["scheduler"]
        self.loss_config = self.config['loss']
        self.data_config = self.config["data"]
        self.logging_config = self.config["logging"]
        self.hardware_config = self.config["hardware"]
        self.metric_config = self.config["metric"]
        self.misc_config = self.config['misc']

        # 初始化日志以及训练结果保存路径
        if not self.training_config['dev']:
            self.start_time = datetime.now().strftime('%Y-%m-%d__%H:%M') # 获取程序运行时的时间
            self.output_path = os.path.join(self.logging_config['log_dir'], self.start_time)
            os.makedirs(self.output_path, exist_ok=True) # 创建日志目录
            self.logger = Logger(self.model_config['name']+" "+self.start_time, self.output_path, logging.DEBUG) 
            self.writer = SummaryWriter(self.output_path)
            with open(os.path.join(self.logging_config['log_dir'], self.start_time, self.config_name), "w") as json_file:
                json.dump(self.config, json_file, indent=4) # 保存当前配置
            

            # 初始化保存路径
            self.checkpoint_dir = os.path.join(self.logging_config["checkpoint_dir"], self.start_time)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 初始化硬件配置（GPU/CPU）
        self.device = torch.device(f"cuda:{self.hardware_config['gpu_id']}" if self.hardware_config["use_gpu"] and torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = self._build_model()
        self.model.to(self.device)

        # 初始化优化器
        self.optimizer = self._build_optimizer()

        # 加载预训练模型
        if self.training_config['continue']:
            if self.model_config['pretrained']:
                self.start_epoch = self.load_checkpoint(self.model_config['pretrained'])
        else:
            self.start_epoch = 0


        # 初始化损失函数
        self.loss_dict, self.loss_weights_dict = self._build_losses()

        # 初始化学习率调度器
        self.scheduler = self._build_scheduler()

        # 初始化指标计算函数
        self.metric_dict = self._build_metrics()

        # 初始化数据加载器
        self.train_loader, self.val_loader, self.test_loader = self._build_data_loaders()

        # 随机种子
        np.random.seed(self.misc_config["random_seed"])
        torch.manual_seed(self.misc_config["random_seed"])

    def _build_model(self):
        """根据配置构建模型"""
        model_name = self.model_config["name"]
        self.logger.debug('model:' + " " + model_name)
        self.model_builder = MODEL_REGISTRY[model_name]
        return self.model_builder()

    def _build_optimizer(self):
        """根据配置构建优化器"""
        optimizer_type = self.optimizer_config["type"]
        self.logger.debug('optimizer:' + " " + optimizer_type)
        if optimizer_type == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer_config["learning_rate"])
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer_config["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return optimizer
    
    def _build_losses(self):
        loss_dict = {}
        loss_weights_dict = {}
        losses_type = self.loss_config['losses']
        loss_weights = self.loss_config['loss_weights']
        self.logger.debug("loss init: " + str(losses_type))
        self.logger.debug("loss weights: " + str(loss_weights))
        for i in range(len(losses_type)):
            loss_dict[losses_type[i]] = LOSS_REGISTRY[losses_type[i]]
            loss_weights_dict[losses_type[i]] = loss_weights[i]
        return loss_dict, loss_weights_dict
    
    def _build_metrics(self):
        metric_dict = {}
        metric_type = self.metric_config['metrics']
        self.logger.debug("metrics init: " + str(metric_type))
        for i in range(len(metric_type)):
            metric_dict[metric_type[i]] = METRIC_REGISTRY[metric_type[i]]
        return metric_dict

    def _build_scheduler(self):
        """根据配置构建学习率调度器"""
        self.logger.debug("schedular: " + self.scheduler_config['type'])
        scheduler_type = self.scheduler_config["type"]
        if scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_config["step_size"], 
                                                   gamma=self.scheduler_config["gamma"])
        elif scheduler_type == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                             factor=self.scheduler_config["gamma"], 
                                                             patience=4)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return scheduler

    def _build_data_loaders(self):
        """根据配置构建数据加载器"""
        self.logger.debug("dataset: " + self.data_config['dataset'])
        transform = TRANSFORM_REGISTRY['totensor']
        data_root = self.data_config['dataset_root']
        train_root = os.path.join(data_root, self.data_config['train_data_dir'])
        val_root = os.path.join(data_root, self.data_config['val_data_dir'])
        test_root = os.path.join(data_root, self.data_config['test_data_dir'])
        # 数据加载
        train_set = DATASET_REGISTRY[self.data_config['dataset']](train_root, transform=transform)
        if train_root == val_root:
            train_set, val_set = data_split(train_set, self.training_config['train_split'], seed=self.misc_config['random_seed'])
        else:
            val_set = DATASET_REGISTRY[self.data_config['dataset']](val_root, transform=transform)
        test_set = DATASET_REGISTRY[self.data_config['dataset']](test_root, transform=transform)
        
        train_loader = DataLoader(train_set, batch_size=self.training_config["batch_size"], shuffle=self.training_config["shuffle"])
        val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def validate(self, epoch):
        NotImplemented

    def test(self):
        NotImplemented
        

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            # 恢复模型的状态字典
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 恢复优化器的状态字典
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复epoch信息
            epoch = checkpoint['epoch']
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch+1}")
            return epoch  # 返回加载的 epoch，可以用来继续训练

    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def train(self):
        NotImplemented

    def close(self):
        """关闭TensorBoard日志"""
        self.writer.close()

    def save_one_batch_to_tensorboard(self, vis_ycbcr, vis_rgb, ir, fuse_y, epoch):
        self.logger.info(f'saving batch of epoch {epoch+1}....')
        to_pil_image = transforms.ToPILImage()
        batch_size = ir.size(0)
        # 遍历每个批次中的图像
        for i in range(batch_size):
            # 可见光图像
            vis = vis_rgb[i, :, :, :].cpu()
            vis = to_pil_image(vis)  # 转移到 CPU
            self.writer.add_image(f'Visible Image/{i+1}', np.array(vis), epoch, dataformats='HWC')  # 0 是 global_step，可以是当前迭代步数

            # 红外图像
            ir_tmp = ir[i, :, :, :].cpu()
            ir_tmp = to_pil_image(ir_tmp)  # 转移到 CPU
            self.writer.add_image(f'Infrared Image/{i+1}', np.array(ir_tmp), epoch, dataformats='HW')

            # 融合图像
            fuse_out = self.put_fuse_y_to_vis(fuse_y[i, :, :, :].cpu(), [vis_ycbcr[j][i, :, :, :] for j in range(3)])
            self.writer.add_image(f'Fused Image/{i+1}', np.array(fuse_out), epoch, dataformats='HWC')

    def put_fuse_y_to_vis(self, fuse_y, vis_ycbcr):
        to_pil_image = transforms.ToPILImage()
        Y, Cb, Cr = to_pil_image(fuse_y).convert('L'), to_pil_image(vis_ycbcr[1]).convert('L'), to_pil_image(vis_ycbcr[2]).convert('L')
        img_fuse_ycbcr = Image.merge('YCbCr', (Y, Cb, Cr))
        img_fuse_rgb = img_fuse_ycbcr.convert('RGB')
        return img_fuse_rgb


class MMIFFusionGANSolver(Solver):
    def __init__(self, config_path):
        super(MMIFFusionGANSolver, self).__init__(config_path)
        assert self.model_config['name'] == "FusionGAN", "config error"
        self.iter_idx = 0

    def _build_model(self):
        self.logger.debug('model init!')
        self.generator, self.discriminator = MODEL_REGISTRY['FusionGAN']()
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        return self.generator

    def _build_optimizer(self):
        """根据配置构建优化器"""
        optimizer_type = self.optimizer_config["type"]
        self.logger.debug('optimizer:' + " " + optimizer_type)
        if optimizer_type == "Adam":
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.optimizer_config["learning_rate"])
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.optimizer_config["learning_rate"])
        elif optimizer_type == "SGD":
            self.g_optimizer = optim.SGD(self.generator.parameters(), lr=self.optimizer_config["learning_rate"])
            self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=self.optimizer_config["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return None # 不使用self.optimizer

    def _build_losses(self):
        loss_dict, loss_weights_dict =  super()._build_losses()
        self.loss_portion_dict = {}
        # 判别器正负样本损失分量
        losses_portion = self.loss_config['loss_portions']
        loss_weights = self.loss_config['portion_weights']
        for i in range(len(losses_portion)):
            self.loss_portion_dict[losses_portion[i]] = loss_weights[i]
        return loss_dict, loss_weights_dict
    
    def _build_scheduler(self):
        """根据配置构建学习率调度器"""
        self.logger.debug("schedular: " + self.scheduler_config['type'])
        scheduler_type = self.scheduler_config["type"]
        if scheduler_type == "StepLR":
            self.g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=self.scheduler_config["step_size"], 
                                                   gamma=self.scheduler_config["gamma"])
            self.d_scheduler = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=self.scheduler_config["step_size"], 
                                                   gamma=self.scheduler_config["gamma"])
        elif scheduler_type == "ReduceLROnPlateau":
            self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, mode='min', 
                                                             factor=self.scheduler_config["gamma"], 
                                                             patience=4)
            self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer, mode='min', 
                                                             factor=self.scheduler_config["gamma"], 
                                                             patience=4)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return None
    
    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'g_model_state_dict': self.generator.state_dict(),
            'd_model_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict()
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            # 恢复模型的状态字典
            self.generator.load_state_dict(checkpoint['g_model_state_dict'])
            self.discriminator.load_state_dict(checkpoint['d_model_state_dict'])
            
            # 恢复优化器的状态字典
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            
            # 恢复epoch信息
            epoch = checkpoint['epoch']
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch+1}")
            return epoch  # 返回加载的 epoch，可以用来继续训练

    def train_fusion_gan_one_epoch_on_MSRS(self, epoch):
        self.generator.train()
        self.discriminator.train()

        with open(os.path.join(self.logger.log_dir, self.logger.log_name+'.log'), 'a') as f:
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{self.training_config['epochs']}", 
                dynamic_ncols=True, 
                file=f
            )

            crop_trans = transforms.RandomCrop(224) # 裁剪图像到指定大小方便判别器判别
            for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(progress_bar):
                batch_size = ir_img.size(0)
                vis_img_y = vis_img_ycbcr[0]
                ir_img = ir_img
                batch_stack = torch.concat([vis_img_y, ir_img], dim=0)
                batch_stack = crop_trans(batch_stack)
                vis_patch_y, ir_patch = batch_stack[:batch_size].to(self.device), batch_stack[batch_size:].to(self.device)
                
                fuse_out = self.generator(vis_patch_y, ir_patch)

                loss_g = 0.
                loss_d = 0.
                loss_g_gan = 0.
                loss_fuse = 0.

                for j in range(self.training_config['d_train_iter']):
                    self.d_optimizer.zero_grad()
                    neg = self.discriminator(fuse_out)
                    pos = self.discriminator(vis_patch_y)
                    loss_neg = self.loss_dict['L2Loss'](neg, torch.rand_like(neg)*0.3)
                    loss_pos = self.loss_dict['L2Loss'](pos, torch.rand_like(pos)*0.5 + 0.7)
                    if epoch >= 0:
                        loss_d = self.loss_portion_dict['d_neg'] * loss_neg + self.loss_portion_dict['d_pos'] * loss_pos
                    else:
                        loss_d = 0.95 * loss_pos + 0.05 * loss_neg
                    loss_d.backward(retain_graph=True)
                    self.d_optimizer.step()
                
                for j in range(self.training_config['g_train_iter']):
                    self.g_optimizer.zero_grad()
                    fuse_out = self.generator(vis_patch_y, ir_patch)
                    fake_pos = self.discriminator(fuse_out)
                    loss_g_gan = self.loss_dict['L2Loss'](fake_pos, torch.rand_like(pos)*0.5 + 0.7)
                    loss_g_gan  = self.loss_portion_dict['loss_g_gan'] * loss_g_gan
                    loss_g_gan.backward(retain_graph=True)
                    self.g_optimizer.step()
                
                self.g_optimizer.zero_grad()
                fuse_out = self.generator(vis_patch_y, ir_patch)
                loss_fuse = (self.loss_dict['PixelWiseL2Loss'](fuse_out, ir_patch) + self.loss_dict['GradientL2Loss'](fuse_out, vis_patch_y)) * self.loss_portion_dict['loss_fuse']
                loss_fuse.backward(retain_graph=True)
                self.g_optimizer.step()
                
                self.logger.info(f'loss generator: {loss_fuse}')
                self.logger.info(f'loss gan generator: {loss_g_gan}')
                self.logger.info(f'loss discriminator: {loss_d}')

                self.writer.add_scalar('Loss/Generator', loss_fuse, self.iter_idx)
                self.writer.add_scalar('Loss/Generator_GAN', loss_g_gan, self.iter_idx)
                self.writer.add_scalar('Loss/Discriminator', loss_d, self.iter_idx)
                self.iter_idx += 1

    def train(self):
        if self.data_config['dataset'] == "MSRS":
            if self.model_config['name'] == "FusionGAN":
                for epoch in range(self.start_epoch, self.training_config['epochs']):
                    self.logger.info('epoch' + str(epoch + 1))
                    # 训练一个epoch
                    self.train_fusion_gan_one_epoch_on_MSRS(epoch)
                    if epoch == self.training_config['epochs'] - 1:
                        self.save_checkpoint(epoch)
                    # 验证一个epoch
                    self.validate(epoch)
                    # 测试集结果
                self.test()
        else:
            self.logger.error("model not implement!")
    
    def validate(self, epoch):
        if self.data_config['dataset'] == "MSRS":
            self.validate_fusion_gan_on_MSRS(epoch, valid=True)

    def test(self):
        if self.data_config['dataset'] == "MSRS":
            self.validate_fusion_gan_on_MSRS(self.training_config['epochs'], valid=False)


    def validate_fusion_gan_on_MSRS(self, epoch, valid=False):
        if valid:
            self.logger.info('validating ....')
        else:
            self.logger.info('testing....')
        self.model.eval()
        self.model.is_training = False

        # 用于存储指标
        ssim = []
        psnr = []
        ag = []
        en = []
        mi = []
        q_abf = []
        vif = []
        cc = []

        flag = False
        with open(os.path.join(self.logger.log_dir, self.logger.log_name+'.log'), 'a') as f:
            progress_bar = tqdm(
                self.val_loader if valid else self.test_loader,
                desc=f"Epoch {epoch+1}/{self.training_config['epochs']}", 
                dynamic_ncols=True, 
                file=f
            )
            for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(progress_bar):
                # 推理
                vis_img_y = vis_img_ycbcr[0].to(self.device)
                ir_img = ir_img.to(self.device)
                with torch.no_grad():
                    fuse_out = self.generator(vis_img_y, ir_img)
                    if not flag:
                        if valid:
                            self.save_one_batch_to_tensorboard(vis_img_ycbcr, vis_img, ir_img, fuse_out, epoch)
                        flag = True

                if not self.model.is_training:
                    ssim += ((np.array(self.metric_dict['SSIM'](vis_img_y, fuse_out)) + np.array(self.metric_dict['SSIM'](ir_img, fuse_out))) / 2).tolist()
                    psnr += self.metric_dict['PSNR'](vis_img_y, ir_img, fuse_out)
                    ag += self.metric_dict['AG'](fuse_out)
                    en += self.metric_dict['EN'](fuse_out)
                    mi += self.metric_dict['MI'](fuse_out, vis_img_y) + self.metric_dict['MI'](fuse_out, ir_img)
                    q_abf += self.metric_dict['Q_abf'](vis_img_y, ir_img, fuse_out)
                    vif += self.metric_dict['VIF'](vis_img_y, ir_img, fuse_out)
                    cc += self.metric_dict['CC'](vis_img_y, ir_img, fuse_out)
                else:
                    ssim += ((np.array(self.metric_dict['SSIM'](vis_img_y, fuse_out[0])) + np.array(self.metric_dict['SSIM'](ir_img, fuse_out[1]))) / 2).tolist()

            
            ssim = statistics.mean(ssim)
            psnr = statistics.mean(psnr)
            ag = statistics.mean(ag)
            en = statistics.mean(en)
            mi = statistics.mean(mi)
            q_abf = statistics.mean(q_abf)
            vif = statistics.mean(vif)
            cc = statistics.mean(cc)

            self.logger.info('SSIM on valid set: ' + str(ssim))
            self.logger.info('PSNR on valid set: ' + str(psnr))
            self.logger.info('AG on valid set: ' + str(ag))
            self.logger.info('EN on valid set: ' + str(en))
            self.logger.info('MI on valid set: ' + str(mi))
            self.logger.info('Qabf on valid set: ' + str(q_abf))
            self.logger.info('VIF on valid set: ' + str(vif))
            self.logger.info('CC on valid set: ' + str(cc))
            
            self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
            self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
            self.writer.add_scalar('Metrics/AG', ag, epoch)
            self.writer.add_scalar('Metrics/EN', en, epoch)
            self.writer.add_scalar('Metrics/MI', mi, epoch)
            self.writer.add_scalar('Metrics/Qabf', q_abf, epoch)
            self.writer.add_scalar('Metrics/VIF', vif, epoch)
            self.writer.add_scalar('Metrics/CC', cc, epoch)




# 用于图像融合的solover
class MMIFDenseFuseSolver(Solver):
    def __init__(self, config_path):
        super(MMIFDenseFuseSolver, self).__init__(config_path)
        self.loss_per_iter = []
        self.loss_per_epoch = []
        self.iter_idx = 0

    def train_densefuse_one_epoch_on_MSRS(self, epoch):
        """
        训练一个 epoch
        """
        self.model.train()  # 设置模型为训练模式
        self.model.is_training = True
        running_loss = 0.0
        iters = 0.0
        

        # 使用 tqdm 进度条显示训练进度
        with open(os.path.join(self.logger.log_dir, self.logger.log_name+'.log'), 'a') as f:
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{self.training_config['epochs']}", 
                dynamic_ncols=True, 
                file=f
            )

            for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(progress_bar):
                # 推理阶段
                self.optimizer.zero_grad() # 清空梯度，避免梯度累计
                vis_img_y = vis_img_ycbcr[0].to(self.device)
                ir_img = ir_img.to(self.device)
                vi_pred, ir_pred = self.model(vis_img_y, ir_img) # 只输入y通道和红外

                # 计算损失
                loss_ssim_vi = 1 - self.loss_dict['SSIM'](vi_pred, vis_img_y)
                loss_ssim_ir = 1 - self.loss_dict['SSIM'](ir_pred, ir_img)
                loss_pixelwise_l2_vi = self.loss_dict['PixelWiseL2Loss'](vi_pred, vis_img_y)
                loss_pixelwise_l2_ir = self.loss_dict['PixelWiseL2Loss'](ir_pred, ir_img)
                loss_vi = self.loss_weights_dict['SSIM'] * loss_ssim_vi + self.loss_weights_dict['PixelWiseL2Loss'] * loss_pixelwise_l2_vi
                loss_ir = self.loss_weights_dict['SSIM'] * loss_ssim_ir + self.loss_weights_dict['PixelWiseL2Loss'] * loss_pixelwise_l2_ir
                loss = loss_vi + loss_ir
                loss.backward()
                self.optimizer.step()
                running_loss += loss
                iters += 1.0
                self.iter_idx += 1

                # 结算
                self.logger.info('loss: '+str(loss.item()))
                self.logger.info('loss_ssim_vi: '+str(loss_ssim_vi.item()))
                self.logger.info('loss_ssim_ir: '+str(loss_ssim_ir.item()))
                self.logger.info('loss_pixel_l2_vi: '+str(loss_pixelwise_l2_vi.item()))
                self.logger.info('loss_pixel_l2_ir: '+str(loss_pixelwise_l2_ir.item()))

                # 记录数据到tensorboard
                self.writer.add_scalar('Loss/SSIM/Visible', loss_ssim_vi.item(), self.iter_idx)
                self.writer.add_scalar('Loss/SSIM/Infrared', loss_ssim_ir.item(), self.iter_idx)
                self.writer.add_scalar('Loss/PixelWiseL2/Visible', loss_pixelwise_l2_vi.item(), self.iter_idx)
                self.writer.add_scalar('Loss/PixelWiseL2/Infrared', loss_pixelwise_l2_ir.item(), self.iter_idx)
                self.writer.add_scalar('Loss/Visible', loss_vi.item(), self.iter_idx)
                self.writer.add_scalar('Loss/Infrared', loss_ir.item(), self.iter_idx)
                self.writer.add_scalar('Loss/Total', loss.item(), self.iter_idx)
                            
            self.logger.debug("loss of epoch "+ str(epoch) + ": "+ str(running_loss / iters))
            self.writer.add_scalar('Loss/Total_Epoch_Wise', (running_loss / iters).item(), epoch+1)


    def train(self):
        if self.data_config['dataset'] == "MSRS":
            if self.model_config['name'] == "DenseFuse":
                for epoch in range(self.start_epoch, self.training_config['epochs']):
                    self.logger.info('epoch' + str(epoch + 1))
                    # 训练一个epoch
                    self.train_densefuse_one_epoch_on_MSRS(epoch)
                    if epoch == self.training_config['epochs'] - 1:
                        self.save_checkpoint(epoch)
                    # 验证一个epoch
                    self.validate(epoch)
                    # 测试集结果
                self.test()
        else:
            self.logger.error("model not implement!")

    def validate(self, epoch):
        if self.data_config['dataset'] == "MSRS":
            if self.model_config['name'] == "DenseFuse":
                self.validate_densefuse_on_MSRS(epoch, valid=True)

    def test(self): # 最后一个epoch存测试结果
        if self.data_config['dataset'] == "MSRS":
            if self.model_config['name'] == "DenseFuse":
                self.validate_densefuse_on_MSRS(self.training_config['epochs'], valid=False)

    def validate_densefuse_on_MSRS(self, epoch, valid=False):
        self.logger.info('validating ....')
        self.model.eval()
        self.model.is_training = False

        # 用于存储指标
        ssim = []
        psnr = []
        ag = []
        en = []
        mi = []
        q_abf = []
        vif = []
        cc = []

        flag = False
        with open(os.path.join(self.logger.log_dir, self.logger.log_name+'.log'), 'a') as f:
            progress_bar = tqdm(
                self.val_loader if valid else self.test_loader,
                desc=f"Epoch {epoch+1}/{self.training_config['epochs']}", 
                dynamic_ncols=True, 
                file=f
            )
            for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(progress_bar):
                # 推理
                vis_img_y = vis_img_ycbcr[0].to(self.device)
                ir_img = ir_img.to(self.device)
                with torch.no_grad():
                    fuse_out = self.model(vis_img_y, ir_img)
                    if not flag:
                        if valid:
                            self.save_one_batch_to_tensorboard(vis_img_ycbcr, vis_img, ir_img, fuse_out, epoch)
                        flag = True

                if not self.model.is_training:
                    ssim += ((np.array(self.metric_dict['SSIM'](vis_img_y, fuse_out)) + np.array(self.metric_dict['SSIM'](ir_img, fuse_out))) / 2).tolist()
                    psnr += self.metric_dict['PSNR'](vis_img_y, ir_img, fuse_out)
                    ag += self.metric_dict['AG'](fuse_out)
                    en += self.metric_dict['EN'](fuse_out)
                    mi += self.metric_dict['MI'](fuse_out, vis_img_y) + self.metric_dict['MI'](fuse_out, ir_img)
                    q_abf += self.metric_dict['Q_abf'](vis_img_y, ir_img, fuse_out)
                    vif += self.metric_dict['VIF'](vis_img_y, ir_img, fuse_out)
                    cc += self.metric_dict['CC'](vis_img_y, ir_img, fuse_out)
                else:
                    ssim += ((np.array(self.metric_dict['SSIM'](vis_img_y, fuse_out[0])) + np.array(self.metric_dict['SSIM'](ir_img, fuse_out[1]))) / 2).tolist()

            
            ssim = statistics.mean(ssim)
            psnr = statistics.mean(psnr)
            ag = statistics.mean(ag)
            en = statistics.mean(en)
            mi = statistics.mean(mi)
            q_abf = statistics.mean(q_abf)
            vif = statistics.mean(vif)
            cc = statistics.mean(cc)

            self.logger.info('SSIM on valid set: ' + str(ssim))
            self.logger.info('PSNR on valid set: ' + str(psnr))
            self.logger.info('AG on valid set: ' + str(ag))
            self.logger.info('EN on valid set: ' + str(en))
            self.logger.info('MI on valid set: ' + str(mi))
            self.logger.info('Qabf on valid set: ' + str(q_abf))
            self.logger.info('VIF on valid set: ' + str(vif))
            self.logger.info('CC on valid set: ' + str(cc))
            
            self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
            self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
            self.writer.add_scalar('Metrics/AG', ag, epoch)
            self.writer.add_scalar('Metrics/EN', en, epoch)
            self.writer.add_scalar('Metrics/MI', mi, epoch)
            self.writer.add_scalar('Metrics/Qabf', q_abf, epoch)
            self.writer.add_scalar('Metrics/VIF', vif, epoch)
            self.writer.add_scalar('Metrics/CC', cc, epoch)


class MMIFDDcGANSolver(Solver):
    def __init__(self, config_path):
        super(MMIFDDcGANSolver, self).__init__(config_path)
        assert self.model_config['name'] == "DDcGAN", "config error"
        self.iter_idx = 0
        self.iter_d_v = 0.
        self.iter_d_i = 0.
        self.iter_g = 0.

    def _build_model(self):
        self.logger.debug('model init!')
        self.generator, self.discriminator_v, self.discriminator_i = MODEL_REGISTRY['DDcGAN']()
        self.generator = self.generator.to(self.device)
        self.discriminator_v = self.discriminator_v.to(self.device)
        self.discriminator_i = self.discriminator_i.to(self.device)
        return self.generator
    
    def _build_optimizer(self):
        """根据配置构建优化器"""
        optimizer_type = self.optimizer_config["type"]
        self.logger.debug('optimizer:' + " " + optimizer_type)
        if optimizer_type == "Adam":
            self.d_optimizer_v = optim.Adam(self.discriminator_v.parameters(), lr=self.optimizer_config["learning_rate"])
            self.d_optimizer_i = optim.Adam(self.discriminator_i.parameters(), lr=self.optimizer_config["learning_rate"])
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.optimizer_config["learning_rate"])
        elif optimizer_type == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer_config["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return None

    def _build_losses(self):
        loss_dict, loss_weights_dict =  super()._build_losses()
        self.loss_portion_dict = {}
        # 判别器正负样本损失分量
        losses_portion = self.loss_config['loss_portions']
        loss_weights = self.loss_config['portion_weights']
        for i in range(len(losses_portion)):
            self.loss_portion_dict[losses_portion[i]] = loss_weights[i]
        return loss_dict, loss_weights_dict
    
    def _build_scheduler(self):
        """根据配置构建学习率调度器"""
        self.logger.debug("schedular: " + self.scheduler_config['type'])
        scheduler_type = self.scheduler_config["type"]
        if scheduler_type == "StepLR":
            self.g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=self.scheduler_config["step_size"], 
                                                   gamma=self.scheduler_config["gamma"])
            self.d_scheduler_v = optim.lr_scheduler.StepLR(self.d_optimizer_v, step_size=self.scheduler_config["step_size"], 
                                                   gamma=self.scheduler_config["gamma"])
            self.d_scheduler_i = optim.lr_scheduler.StepLR(self.d_optimizer_i, step_size=self.scheduler_config["step_size"], 
                                                   gamma=self.scheduler_config["gamma"])
            
        elif scheduler_type == "ReduceLROnPlateau":
            self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, mode='min', 
                                                             factor=self.scheduler_config["gamma"], 
                                                             patience=4)
            self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer, mode='min', 
                                                             factor=self.scheduler_config["gamma"], 
                                                             patience=4)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        return None
    
    def train(self):
        if self.data_config['dataset'] == "MSRS":
            if self.model_config['name'] == "DDcGAN":
                for epoch in range(self.start_epoch, self.training_config['epochs']):
                    self.logger.info('epoch' + str(epoch + 1))
                    # 训练一个epoch
                    self.train_ddc_gan_one_epoch_on_MSRS(epoch)
                    if epoch == self.training_config['epochs'] - 1:
                        self.save_checkpoint(epoch)
                    # 验证一个epoch
                    self.validate(epoch)
                    # 测试集结果
                self.test()
        else:
            self.logger.error("model not implement!")

    def train_ddc_gan_one_epoch_on_MSRS(self, epoch):
        self.generator.train()
        self.discriminator_v.train()
        self.discriminator_i.train()

        with open(os.path.join(self.logger.log_dir, self.logger.log_name+'.log'), 'a') as f:
            progress_bar = tqdm(
                self.train_loader, 
                desc=f"Epoch {epoch+1}/{self.training_config['epochs']}", 
                dynamic_ncols=True, 
                file=f
            )

            crop_trans = transforms.RandomCrop(224) # 裁剪图像到指定大小方便判别器判别
            for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(progress_bar):
                batch_size = ir_img.size(0)
                vis_img_y = vis_img_ycbcr[0]
                ir_img = ir_img
                batch_stack = torch.concat([vis_img_y, ir_img], dim=0)
                batch_stack = crop_trans(batch_stack)
                vis_patch_y, ir_patch = batch_stack[:batch_size].to(self.device), batch_stack[batch_size:].to(self.device)
                
                fuse_out = self.generator(vis_patch_y, ir_patch)
                loss_d_v = 1000
                loss_d_i = 1000
                loss_gan_v = 0
                loss_gan_i = 0
                loss_gan = 0
                loss_fuse = 0
                cross_entropy = nn.BCELoss()
                # 训练判别器
                iter = 0.
                while loss_d_v > self.training_config['loss_d_v_thr'] and iter < self.training_config['d_train_iter']:
                    # 预测
                    self.d_optimizer_v.zero_grad()
                    pos_v = self.discriminator_v(vis_patch_y)
                    neg_v = self.discriminator_v(fuse_out)
                    loss_d_v = self.loss_portion_dict['d_pos_v'] * cross_entropy(pos_v, torch.ones_like(pos_v)) + self.loss_portion_dict['d_neg_v'] * cross_entropy(neg_v, torch.zeros_like(neg_v))
                    loss_d_v.backward(retain_graph=True)
                    self.d_optimizer_v.step()
                    iter += 1

                    self.writer.add_scalar('Loss/loss_d_v', loss_d_v, self.iter_d_v + iter)

                self.iter_d_v += iter

                iter = 0.
                while loss_d_i > self.training_config['loss_d_i_thr'] and iter < self.training_config['d_train_iter']:
                    # 预测
                    self.d_optimizer_i.zero_grad()
                    pos_i = self.discriminator_i(ir_patch)
                    neg_i = self.discriminator_v(fuse_out)
                    loss_d_i = self.loss_portion_dict['d_pos_i'] * cross_entropy(pos_i, torch.ones_like(pos_v)) + self.loss_portion_dict['d_neg_i'] * cross_entropy(neg_i, torch.zeros_like(neg_v))
                    loss_d_i.backward(retain_graph=True)
                    self.d_optimizer_i.step()
                    iter += 1

                    self.writer.add_scalar('Loss/loss_d_i', loss_d_i, self.iter_d_i + iter)

                self.iter_d_i += iter

                iter = 0.
                while (loss_gan_i < self.training_config['loss_g_i_thr'] or loss_gan_v < self.training_config['loss_g_v_thr']) and iter < 20:
                    self.g_optimizer.zero_grad()
                    img_fuse = self.generator(vis_patch_y, ir_patch)
                    fake_pos_v = self.discriminator_v(img_fuse)
                    fake_pos_i = self.discriminator_i(img_fuse)
                    loss_gan_v = cross_entropy(fake_pos_v, torch.ones_like(fake_pos_v))
                    loss_gan_i = cross_entropy(fake_pos_i, torch.ones_like(fake_pos_i))
                    loss_gan = self.loss_portion_dict['loss_gan_v'] * loss_gan_v + self.loss_portion_dict['loss_gan_i'] * loss_gan_i
                    loss_fuse = self.loss_dict['PixelWiseL2Loss'](img_fuse, ir_patch) * self.loss_weights_dict['PixelWiseL2Loss'] + self.loss_dict['GradientL1Loss'](img_fuse, vis_patch_y) * self.loss_weights_dict['GradientL1Loss']
                    loss_g = self.loss_portion_dict['loss_fuse'] * loss_fuse + loss_gan
                    loss_g.backward(retain_graph=True)
                    self.g_optimizer.step()
                    iter += 1
                    # 记录Loss
                    self.writer.add_scalar('Loss/loss_fuse', loss_fuse, self.iter_d_v + iter)
                    self.writer.add_scalar('Loss/loss_gan', loss_gan, self.iter_d_v + iter)
                    self.writer.add_scalar('Loss/loss_g_v', loss_gan_v, self.iter_d_v + iter)
                    self.writer.add_scalar('Loss/loss_g_i', loss_gan_i, self.iter_d_v + iter)

                self.iter_g += iter

                self.iter_idx += 1

                self.logger.info(f'loss_d_v: {loss_d_v}, loss_d_i: {loss_d_i}, loss_gan_v: {loss_gan_v}, loss_gan_i: {loss_gan_i}, loss_gan: {loss_gan}, loss_fuse: {loss_fuse}')

    def validate_ddc_gan_on_MSRS(self, epoch, valid=False):
        if valid:
            self.logger.info('validating ....')
        else:
            self.logger.info('testing....')
        self.model.eval()
        self.model.is_training = False

        # 用于存储指标
        ssim = []
        psnr = []
        ag = []
        en = []
        mi = []
        q_abf = []
        vif = []
        cc = []

        flag = False
        with open(os.path.join(self.logger.log_dir, self.logger.log_name+'.log'), 'a') as f:
            progress_bar = tqdm(
                self.val_loader if valid else self.test_loader,
                desc=f"Epoch {epoch+1}/{self.training_config['epochs']}", 
                dynamic_ncols=True, 
                file=f
            )
            for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(progress_bar):
                # 推理
                vis_img_y = vis_img_ycbcr[0].to(self.device)
                ir_img = ir_img.to(self.device)
                with torch.no_grad():
                    fuse_out = self.generator(vis_img_y, ir_img)
                    if not flag:
                        if valid:
                            self.save_one_batch_to_tensorboard(vis_img_ycbcr, vis_img, ir_img, fuse_out, epoch)
                        flag = True

                if not self.model.is_training:
                    ssim += ((np.array(self.metric_dict['SSIM'](vis_img_y, fuse_out)) + np.array(self.metric_dict['SSIM'](ir_img, fuse_out))) / 2).tolist()
                    psnr += self.metric_dict['PSNR'](vis_img_y, ir_img, fuse_out)
                    ag += self.metric_dict['AG'](fuse_out)
                    en += self.metric_dict['EN'](fuse_out)
                    mi += self.metric_dict['MI'](fuse_out, vis_img_y) + self.metric_dict['MI'](fuse_out, ir_img)
                    q_abf += self.metric_dict['Q_abf'](vis_img_y, ir_img, fuse_out)
                    vif += self.metric_dict['VIF'](vis_img_y, ir_img, fuse_out)
                    cc += self.metric_dict['CC'](vis_img_y, ir_img, fuse_out)
                else:
                    ssim += ((np.array(self.metric_dict['SSIM'](vis_img_y, fuse_out[0])) + np.array(self.metric_dict['SSIM'](ir_img, fuse_out[1]))) / 2).tolist()

            
            ssim = statistics.mean(ssim)
            psnr = statistics.mean(psnr)
            ag = statistics.mean(ag)
            en = statistics.mean(en)
            mi = statistics.mean(mi)
            q_abf = statistics.mean(q_abf)
            vif = statistics.mean(vif)
            cc = statistics.mean(cc)

            self.logger.info('SSIM on valid set: ' + str(ssim))
            self.logger.info('PSNR on valid set: ' + str(psnr))
            self.logger.info('AG on valid set: ' + str(ag))
            self.logger.info('EN on valid set: ' + str(en))
            self.logger.info('MI on valid set: ' + str(mi))
            self.logger.info('Qabf on valid set: ' + str(q_abf))
            self.logger.info('VIF on valid set: ' + str(vif))
            self.logger.info('CC on valid set: ' + str(cc))
            
            self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
            self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
            self.writer.add_scalar('Metrics/AG', ag, epoch)
            self.writer.add_scalar('Metrics/EN', en, epoch)
            self.writer.add_scalar('Metrics/MI', mi, epoch)
            self.writer.add_scalar('Metrics/Qabf', q_abf, epoch)
            self.writer.add_scalar('Metrics/VIF', vif, epoch)
            self.writer.add_scalar('Metrics/CC', cc, epoch)
    
    def validate(self, epoch):
        if self.data_config['dataset'] == "MSRS":
            self.validate_ddc_gan_on_MSRS(epoch, valid=True)


# 测试代码
if __name__ == "__main__":
    solver = MMIFDDcGANSolver('./meta/configs/config_DDcGAN.json')
    # solver = MMIFFusionGANSolver('./meta/configs/config_fusionGAN.json')
    solver.train()