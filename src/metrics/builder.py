from metrics.metrics import *

METRIC_REGISTRY = {}

def register_metric(name):
    """
    注册模型构建函数装饰器
    """
    def decorator(func):
        METRIC_REGISTRY[name] = func
        return func
    return decorator

@register_metric('EN')
def build_entropy(batch):
    return entropy(batch)

@register_metric('MI')
def build_mutual_infomation(batch1, batch2):
    return mutual_infomation(batch1, batch2)

@register_metric('SSIM')
def build_mutual_infomation(batch1, batch2):
    return SSIM(batch1, batch2)

@register_metric('CC')
def build_CC(batch1, batch2, batch_fuse):
    return CC(batch1, batch2, batch_fuse)

@register_metric('SCD')
def build_SCD(batch1, batch2, batch_fuse):
    return SCD(batch1, batch2, batch_fuse)

@register_metric('NCC')
def build_normalized_correlation_coefficient(batch1, batch2, batch_fuse):
    return NCC(batch1, batch2, batch_fuse)

@register_metric('PSNR')
def build_PSNR(batch1, batch2, batch_fuse):
    return PSNR(batch1, batch2, batch_fuse)

@register_metric('AG')
def build_AG(batch):
    return AG(batch)

@register_metric('SF')
def build_SF(batch):
    return SF(batch)

@register_metric('Q_abf')
def build_Q_abf(batch1, batch2, batch_fuse):
    return Q_abf(batch1, batch2, batch_fuse)

# 慎用
@register_metric('N_abf')
def build_N_abf(batch1, batch2, batch_fuse):
    return N_abf(batch1, batch2, batch_fuse)

@register_metric('VIF')
def build_VIF(batch1, batch2, batch_fuse):
    return VIFF(batch1, batch2, batch_fuse)