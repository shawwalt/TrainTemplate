import torch
import torchvision.models as models
import torch.nn as nn
from models.densefuse import DenseFuse
from models.fusionGAN import FusionGANGenerator, FusionGANDiscriminator
from models.DDcGAN import DDcGANDiscriminator, DDcGANGenerator

# 1. 定义一个模型注册表，用于存储所有支持的模型和它们的构建函数
MODEL_REGISTRY = {}

def register_model(name):
    """
    注册模型构建函数装饰器
    """
    def decorator(func):
        MODEL_REGISTRY[name] = func
        return func
    return decorator

# 2. 注册模型架构
@register_model('DenseFuse')
def build_densefuse():
    model = DenseFuse()
    return model

@register_model('FusionGAN')
def build_fusionGAN():
    # 注意返回list
    g_model = FusionGANGenerator()
    d_model = FusionGANDiscriminator()
    return g_model, d_model

@register_model('DDcGAN')
def build_DDcGAN():
    g_model = DDcGANGenerator()
    d_model_v = DDcGANDiscriminator()
    d_model_i = DDcGANDiscriminator()
    return g_model, d_model_v, d_model_i