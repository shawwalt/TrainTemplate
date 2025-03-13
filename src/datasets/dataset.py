import os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def data_split(dataset, train_portion, seed):
    train_size = int(train_portion * len(dataset))
    train_set, val_set = random_split(
        dataset=dataset, 
        lengths=[train_size, len(dataset)-train_size],  
        generator=torch.Generator().manual_seed(seed)
    )
    return train_set, val_set


# ************************************数据集定义**************************************************
class MSRSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录，包含train目录
            transform (callable, optional): 用于图像的预处理（例如，数据增强）
        """
        self.root_dir = root_dir
        self.transform = transform

        # 目录下的vis, ir, Segmentation_labels路径
        self.vis_dir = os.path.join(root_dir, 'vi')
        self.ir_dir = os.path.join(root_dir, 'ir')
        self.labels_dir = os.path.join(root_dir, 'Segmentation_labels')

        # 获取所有可见光图像的文件名（假设它们的文件名和红外图像、标签一致）
        self.vis_images = sorted(os.listdir(self.vis_dir))

    def __len__(self):
        """返回数据集的大小"""
        return len(self.vis_images)

    def __getitem__(self, idx):
        """
        获取指定索引的数据，包括可见光图像、红外图像和分割标签
        """
        # 获取文件路径
        vis_path = os.path.join(self.vis_dir, self.vis_images[idx])
        ir_path = os.path.join(self.ir_dir, self.vis_images[idx])  # 假设vis和ir图像的文件名相同
        label_path = os.path.join(self.labels_dir, self.vis_images[idx])  # 假设标签为png格式

        # 加载图像和标签
        vis_image = Image.open(vis_path).convert('RGB')
        vis_image_y, vis_image_cb, vis_image_cr = vis_image.convert('YCbCr').split()
        ir_image = Image.open(ir_path).convert('L')
        label_image = Image.open(label_path).convert('L')  # 转换为灰度图像作为标签

        if self.transform:
            vis_image = self.transform(vis_image)
            vis_image_y = self.transform(vis_image_y)
            vis_image_cb = self.transform(vis_image_cb)
            vis_image_cr = self.transform(vis_image_cr)
            ir_image = self.transform(ir_image)
            label_image = self.transform(label_image)  # 标签通常是单通道，所以直接转换为Tensor

        vis_image_ycbcr = [vis_image_y, vis_image_cb, vis_image_cr]

        return vis_image_ycbcr, vis_image, ir_image, label_image

# *****************************测试代码*****************************************************
def test_MSRSDataSet():
    # 创建MSRSDataset实例
    dataset = MSRSDataset(root_dir='/mnt/disk2/Shawalt/Demos/ImageFusion/DataSet/MSRS/train', transform=transform)

    # 创建DataLoader，设置批量大小、shuffle、并行加载等参数
    batch_size = 16  # 你可以根据GPU显存大小调整
    num_epochs = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 数据加载循环
    for epoch in range(num_epochs):
        for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, unit="batch")):
            # 确保你的模型和数据都在同一设备上（例如，GPU）
            vis_img_y = vis_img_ycbcr[0].to(device)
            ir_img = ir_img.to(device)
            label_img = label_img.to(device)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_MSRSDataSet()

