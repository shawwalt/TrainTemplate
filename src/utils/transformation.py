import torchvision.transforms.transforms as transforms

TRANSFORM_REGISTRY = {}

def register_transformation(name):
    """
    注册模型构建函数装饰器
    """
    def decorator(func):
        TRANSFORM_REGISTRY[name] = func
        return func
    return decorator

@register_transformation('totensor')
def totensor(image):
    return transforms.ToTensor()(image)