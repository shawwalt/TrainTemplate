from datasets.dataset import *

DATASET_REGISTRY = {}

def register_dataset(name):
    """
    注册模型构建函数装饰器
    """
    def decorator(func):
        DATASET_REGISTRY[name] = func
        return func
    return decorator

@register_dataset('MSRS')
def build_MSRSDataset(data_root, transform):
    dataset = MSRSDataset(data_root, transform)
    return dataset