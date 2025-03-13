import torch.nn as nn
from losses.loss import L2Loss, PixelWiseL2Loss, SSIM, GradientL2Loss, L1Loss, PixelWiseL1Loss, GradientL1Loss

# 1. 定义一个模型注册表，用于存储所有支持的模型和它们的构建函数
LOSS_REGISTRY = {}

def register_loss(name):
    """
    注册模型构建函数装饰器
    """
    def decorator(func):
        LOSS_REGISTRY[name] = func
        return func
    return decorator

# 2. 注册损失
@register_loss('PixelWiseL2Loss')
def build_PixelWiseL2Loss(output, gt):
    loss = PixelWiseL2Loss()
    return loss(output, gt)

@register_loss('L2Loss')
def build_L2Loss(output, gt):
    loss = L2Loss()
    return loss(output, gt)

@register_loss('GradientL2Loss')
def build_GradientL2Loss(output, gt):
    loss = GradientL2Loss()
    return loss(output, gt)

@register_loss('SSIM')
def build_SSIM(output, gt):
    loss = SSIM()
    return loss(output, gt)

@register_loss('L1Loss')
def build_L1Loss(output, gt):
    loss = L1Loss()
    return loss(output, gt)

@register_loss('PixelWiseL1Loss')
def build_PixelWiseL1Loss(output, gt):
    loss = PixelWiseL1Loss()
    return loss(output, gt)

@register_loss('GradientL1Loss')
def build_GradientL1Loss(output, gt):
    loss = GradientL1Loss()
    return loss(output, gt)

