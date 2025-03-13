import json

# 创建一个字典，代表你的配置
config = {
    "model": {
        "architecture": "DenseFuse",              # 模型架构，支持的架构名如ResNet50, VGG16等
        "pretrained": True,                     # 是否使用预训练权重
        "batch_norm": True                      # 是否使用批量归一化
    },

    "training": {
        "epochs": 10,                           # 训练轮数
        "batch_size": 4,                       # 每批次的大小
        "validation_split": 0.2,                 # 从训练集划分多少作为验证集
        "shuffle": True,                        # 是否在每轮训练前打乱数据
        "losses": [
            "SSIM", "PixelWiseL2Loss"        # 损失函数
        ],
        "loss_weights":[
            0.6, 1                           # 损失函数的权重
        ]
    },

    "optimizer": {
        "type": "Adam",                         # 优化器类型，常用的有Adam, SGD, RMSprop等
        "learning_rate": 0.001,                 # 初始学习率
    },

    "scheduler": {
        "type": "StepLR",                       # 学习率调度器类型，如StepLR, ReduceLROnPlateau等
        "step_size": 10,                        # 每10个epoch降低一次学习率
        "gamma": 0.1                            # 学习率衰减系数
    },

    "data": {
        "dataset": "MSRS",
        "dataset_root": "/mnt/disk2/Shawalt/Demos/ImageFusion/DataSet/MSRS",
        "train_data_dir": "train",   # 训练数据目录路径
        "val_data_dir": "train",       # 验证数据目录路径
        "test_data_dir": "test",     # 测试数据目录路径
        "augmentation": {
            "enabled": True,                        # 是否启用数据增强
            "rotation_range": 20,                   # 旋转范围
            "width_shift_range": 0.2,               # 水平平移
            "height_shift_range": 0.2,              # 垂直平移
            "shear_range": 0.2,                     # 剪切变换
            "zoom_range": 0.2,                      # 缩放范围
            "horizontal_flip": True,                # 是否进行水平翻转
            "vertical_flip": False                  # 是否进行垂直翻转
        }
    },

    "logging": {
        "log_dir": "./meta/logs",                     # 日志文件保存的目录
        "save_freq": 5,                           # 保存模型的频率（每5轮保存一次）
        "tensorboard": True,                      # 是否启用 TensorBoard 可视化
        "checkpoint": True,                       # 是否启用模型检查点保存
        "checkpoint_dir": "./meta/checkpoints"         # 保存检查点的目录
    },

    "hardware": {
        "use_gpu": True,                         # 是否使用GPU进行训练
        "gpu_id": 0,                             # 使用的GPU ID（如果有多个GPU时）
        "multi_gpu": True                      # 是否使用多GPU训练
    },

    "misc": {
        "random_seed": 42,                       # 随机种子（确保可重复实验）
        "verbose": True                           # 是否输出详细信息
    }
}

# 将配置字典转换为JSON格式并保存到文件中
config_file_path = "./meta/configs/config_densefuse.json"
with open(config_file_path, "w") as json_file:
    json.dump(config, json_file, indent=4)

print(f"配置文件已保存至 {config_file_path}")
