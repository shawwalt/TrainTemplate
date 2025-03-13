import sys
source_root = "/mnt/disk2/Shawalt/Demos/ImageFusion/TrainTemplate/src"
sys.path.append(source_root)

from torchvision.transforms import transforms
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from datasets.dataset import MSRSDataset
from metrics.vif import vifp_mscale

# 使用8bit量化图像
# 输入范围均为[0, 1] torch.tensor
# L = 255
# 只接受单通道图像
def entropy_for_single_tensor(img):
    to_pil_img = transforms.ToPILImage()
    pil_img = to_pil_img(img)
    img_np = np.array(pil_img)

    # 计算图像的直方图
    hist, _ = np.histogram(img_np.flatten(), bins=256, range=(0, 256))
    # 归一化直方图
    hist = hist / hist.sum()
    # 计算熵
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))  # 加上小值防止log(0)

    return entropy

# *******************************计算信息熵****************************************************************
# 信息熵越大，图像包含的信息越丰富，噪声图像信息熵一般较大
def entropy(batch):
    img_tuple = torch.split(batch, 1, dim=0)
    entropys = []
    for tensor in img_tuple:
        tmp = entropy_for_single_tensor(tensor.squeeze(0))
        entropys.append(tmp)
    return entropys

# *******************************计算互信息*****************************************************************
# 两张图像互信息小，则两张图像相似性越小
def mutual_infomation(batch1, batch2):
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    mutual_infs = []
    for tensor1, tensor2 in zip(img_1_tuple, img_2_tuple):
        tmp = mutual_infomation_for_single_pair(tensor1, tensor2)
        mutual_infs.append(tmp)
    return mutual_infs

def mutual_infomation_for_single_pair(tensor1, tensor2):
    entropy_A = entropy_for_single_tensor(tensor1.squeeze(0))
    entropy_B = entropy_for_single_tensor(tensor2.squeeze(0))
    entropy_AB = uni_entropy(tensor1.squeeze(0), tensor2.squeeze(0))
    return entropy_A + entropy_B - entropy_AB

# 计算联合信息熵
def uni_entropy(tensor1, tensor2):
    to_pil_image = transforms.ToPILImage()
    img1 = to_pil_image(tensor1)
    img2 = to_pil_image(tensor2)
    img1_np = np.array(img1)
    img2_np = np.array(img2)

    assert img1_np.shape == img2_np.shape, "两张图像需具有相同大小"

    # 计算联合直方图
    joint_histogram, _, _ = np.histogram2d(img1_np.flatten(), img2_np.flatten(), bins=256, range=[[0, 256], [0, 256]])
    
    # 归一化联合直方图，得到联合概率分布
    joint_histogram = joint_histogram / joint_histogram.sum()
    
    # 计算联合信息熵
    joint_entropy = -np.sum(joint_histogram * np.log2(joint_histogram + np.finfo(float).eps))  # 防止log(0)
    
    return joint_entropy

# *********************************计算SSIM******************************************************************************
def SSIM(batch1, batch2):
    SSIM_S = []
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    for tensor1, tensor2 in zip(img_1_tuple, img_2_tuple):
        tmp = SSIM_for_single_pair(tensor1.squeeze(0), tensor2.squeeze(0))
        SSIM_S.append(tmp)
    return SSIM_S

def SSIM_for_single_pair(tensor1, tensor2):
    to_pil_image = transforms.ToPILImage()
    img1 = to_pil_image(tensor1)
    img2 = to_pil_image(tensor2)
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    out = ssim(img1_np, img2_np, data_range=img1_np.max() - img1_np.min())
    return out

# **********************************SCD 差分相关总和**********************************************************
def SCD(batch1, batch2, batch_fuse):
    SCD_S = []
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    img_fuse_tuple = torch.split(batch_fuse, 1, dim=0)
    for tensor1, tensor2, tensor_fuse in zip(img_1_tuple, img_2_tuple, img_fuse_tuple):
        tmp = sum_of_correlation_difference_per_pair(tensor1.squeeze(0), tensor2.squeeze(0), tensor_fuse.squeeze(0))
        SCD_S.append(tmp)
    return SCD_S

def sum_of_correlation_difference_per_pair(tensor1, tensor2, tensor_fuse):
    difference_1 = tensor_fuse - tensor2 # 多出来的来自图像1的信息
    difference_2 = tensor_fuse - tensor1 # 多出来的来自图像2的信息
    c_1 = correlation(difference_1, tensor1)
    c_2 = correlation(difference_2, tensor2)
    return c_1 + c_2

# 计算图像相关系数
def correlation(tensor1, tensor2):
    h, w = tensor1.size(1), tensor1.size(2)
    mu_1 = torch.mean(tensor1)
    mu_2 = torch.mean(tensor2)
    std_1 = torch.std(tensor1) + torch.finfo(tensor1.dtype).eps # 防止除0错误
    std_2 = torch.std(tensor2) + torch.finfo(tensor1.dtype).eps
    covariance = torch.sum((tensor1 - mu_1) * (tensor2 - mu_2)) + torch.finfo(tensor1.dtype).eps
    covariance = covariance / (h * w)
    return (covariance / (std_1 * std_2)).item()

# **************************************计算平均相关系数**************************************
def CC(batch1, batch2, batch_fuse):
    CC_S = []
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    img_fuse_tuple = torch.split(batch_fuse, 1, dim=0)
    for tensor1, tensor2, tensor_fuse in zip(img_1_tuple, img_2_tuple, img_fuse_tuple):
        tmp = (correlation(tensor1.squeeze(0), tensor_fuse.squeeze(0)) + correlation(tensor2.squeeze(0), tensor_fuse.squeeze(0))) / 2.0
        CC_S.append(tmp)
    return CC_S

# **************************************计算归一化相关系数NCC***************************************
def NCC(batch1, batch2, batch_fuse):
    NCC_S = []
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    img_fuse_tuple = torch.split(batch_fuse, 1, dim=0)
    for tensor1, tensor2, tensor_fuse in zip(img_1_tuple, img_2_tuple, img_fuse_tuple):
        tmp = normalized_correlation_coefficient(tensor1.squeeze(0), tensor2.squeeze(0), tensor_fuse.squeeze(0))
        NCC_S.append(tmp)
    return NCC_S

def normalized_correlation_coefficient(tensor1, tensor2, tensor_fuse):
    pow_1 = torch.mean(tensor1 ** 2)
    pow_2 = torch.mean(tensor2 ** 2)
    mutual_corr_1 = torch.mean(tensor1 * tensor_fuse)
    mutual_corr_2 = torch.mean(tensor2 * tensor_fuse)
    return ((mutual_corr_1 / pow_1 + mutual_corr_2 / pow_2) / 2).item()

# **********************************计算峰值信噪比PSNR****************************************************
def PSNR(batch1, batch2, batch_fuse):
    PSNR_S = []
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    img_fuse_tuple = torch.split(batch_fuse, 1, dim=0)
    for tensor1, tensor2, tensor_fuse in zip(img_1_tuple, img_2_tuple, img_fuse_tuple):
        tmp = PSNR_for_single_pair(tensor1.squeeze(0), tensor2.squeeze(0), tensor_fuse.squeeze(0))
        PSNR_S.append(tmp)
    return PSNR_S

def PSNR_for_single_pair(tensor1, tensor2, tensor_fuse):
    L = 255
    tensor1 = (tensor1 * L).round()
    tensor2 = (tensor2 * L).round()
    tensor_fuse = (tensor_fuse * L).round()
    mse_AF = F.mse_loss(tensor1, tensor_fuse)
    mse_BF = F.mse_loss(tensor2, tensor_fuse)
    mse = (mse_AF + mse_BF) / 2.0
    r = 255
    psnr = 10 * torch.log10(r**2 / mse)
    return psnr.item()

# *************************************计算平均梯度AG**********************************************
def AG(batch):
    AG_S = []
    img_tuple = torch.split(batch, 1, dim=0)
    for tensor in img_tuple:
        tmp = AG_for_single_pair(tensor.squeeze(0))
        AG_S.append(tmp)
    return AG_S

def AG_for_single_pair(tensor):
    tensor = (tensor * 255).round()
    kernel_y = torch.tensor([[[[1.0], [-1.0]]]]).to(tensor.device)
    kernel_x = torch.tensor([[[[1.0, -1.0]]]]).to(tensor.device)
    g_map_x = F.conv2d(tensor.unsqueeze(0), kernel_x)
    g_map_y = F.conv2d(tensor.unsqueeze(0), kernel_y)
    g_map_x = g_map_x[:, :, :-1, :]
    g_map_y = g_map_y[:, :, :, :-1]
    average_grad = torch.mean(torch.sqrt((g_map_x**2 + g_map_y**2) / 2.0))
    return average_grad.item()

# *************************************计算空间频率SF***********************************************
def SF(batch):
    SF_S = []
    img_tuple = torch.split(batch, 1, dim=0)
    for tensor in img_tuple:
        tmp = SF_for_single_pair(tensor.squeeze(0))
        SF_S.append(tmp)
    return SF_S

def SF_for_single_pair(tensor):
    h, w = tensor.size(1), tensor.size(2)
    tensor = (tensor * 255).round()
    kernel_y = torch.tensor([[[[1.0], [-1.0]]]])
    kernel_x = torch.tensor([[[[1.0, -1.0]]]])
    g_map_x = F.conv2d(tensor.unsqueeze(0), kernel_x).to(tensor.device)
    g_map_y = F.conv2d(tensor.unsqueeze(0), kernel_y).to(tensor.device)
    g_map_x = g_map_x[:, :, :-1, :]
    g_map_y = g_map_y[:, :, :, :-1]
    pow_gx = torch.mean(g_map_x ** 2)
    pow_gy = torch.mean(g_map_y ** 2)
    return torch.sqrt(pow_gx + pow_gy).item()
    
# ******************************************基于边缘信息的指标Q_abf**********************************************
def Q_abf(batch1, batch2, batch_fuse):
    Q_abf_s = []
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    img_fuse_tuple = torch.split(batch_fuse, 1, dim=0)
    for tensor1, tensor2, tensor_fuse in zip(img_1_tuple, img_2_tuple, img_fuse_tuple):
        tmp = Q_abf_for_single_pair(tensor1.squeeze(0), tensor2.squeeze(0), tensor_fuse.squeeze(0))
        Q_abf_s.append(tmp)
    return Q_abf_s
    
def Q_abf_for_single_pair(tensor1, tensor2, tensor_fuse):
    # 计算梯度
    gA, aA = get_strength_and_orientation(tensor1)
    gB, aB = get_strength_and_orientation(tensor2)
    gf, af = get_strength_and_orientation(tensor_fuse)

    Q_AF = get_Qabf(gA, aA, gf, af)
    Q_BF = get_Qabf(gB, aB, gf, af)

    # 整合
    deno = torch.sum(gA + gB)
    nume = torch.sum(Q_AF * gA + Q_BF * gB)
    return (nume / (deno + torch.finfo(tensor1.dtype).eps)).item()


def get_Qabf(gA, aA, gf, af):
    # 模型参数
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    # 计算梯度变化
    G_AF = gf / (gA + torch.finfo(gA.dtype).eps) # 防止除零
    G_AF = torch.where(G_AF > 1, 1 / G_AF, G_AF)
    A_AF = torch.abs(1 - torch.abs(af - aA) / (torch.pi))
    Q_gAF = Tg/(1 + torch.exp(kg * (G_AF - Dg)))
    Q_aAF = Ta/(1 + torch.exp(ka * (A_AF - Da)))
    Q_AF = Q_gAF * Q_aAF
    return Q_AF


def get_strength_and_orientation(tensor):
    tensor = tensor.unsqueeze(0)
    # Sobel算子（计算水平方向梯度和垂直方向梯度）
    sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]]]], dtype=tensor.dtype).to(tensor.device)

    sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0],
                            [0.0, 0.0, 0.0],
                            [1.0, 2.0, 1.0]]]], dtype=tensor.dtype).to(tensor.device)
    
    # 应用卷积操作来计算梯度
    grad_x = F.conv2d(tensor, sobel_x, padding=1)  # 水平方向梯度
    grad_y = F.conv2d(tensor, sobel_y, padding=1)  # 垂直方向梯度

    # 计算梯度幅度和相位
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # 梯度强度
    gradient_direction = torch.atan2(grad_y, grad_x)  # 梯度方向    
    return gradient_magnitude, gradient_direction

# ************************************计算基于伪影的指标N_abf*************************************************
def N_abf(batch1, batch2, batch_fuse):
    N_abf_s = []
    img_1_tuple = torch.split(batch1, 1, dim=0)
    img_2_tuple = torch.split(batch2, 1, dim=0)
    img_fuse_tuple = torch.split(batch_fuse, 1, dim=0)
    for tensor1, tensor2, tensor_fuse in zip(img_1_tuple, img_2_tuple, img_fuse_tuple):
        tmp = N_abf_for_single_pair(tensor1.squeeze(0), tensor2.squeeze(0), tensor_fuse.squeeze(0))
        N_abf_s.append(tmp)
    return N_abf_s

def N_abf_for_single_pair(tensor1, tensor2, tensor_fuse):
    # 计算梯度
    gA, aA = get_strength_and_orientation(tensor1)
    gB, aB = get_strength_and_orientation(tensor2)
    gf, af = get_strength_and_orientation(tensor_fuse)

    Q_AF = get_Qabf(gA, aA, gf, af)
    Q_BF = get_Qabf(gB, aB, gf, af)

    AM = torch.zeros_like(gf)
    AM = torch.where((gf > gA) & (gf > gB), 1, AM)
    N_abf = torch.sum(AM * ((1 - Q_AF) * gA + (1 - Q_BF) * gB)) / torch.sum(gA + gB)
    return N_abf.item()

# ************************************计算视觉保真度VIF*************************************
def VIFF(batch1, batch2, batch_fuse):
    VIF1 = VIF(batch1, batch_fuse)
    VIF2 = VIF(batch2, batch_fuse)
    VIFF = list(map(lambda a, b: (a + b) / 2.0, VIF1, VIF2))
    return VIFF

def VIF(batch_source, batch_fuse):
    VIF_s = []
    source_tuple = torch.split(batch_source, 1, dim=0)
    fuse_tuple = torch.split(batch_fuse, 1, dim=0)
    for tensor_source, tensor_fuse in zip(source_tuple, fuse_tuple):
        tmp = VIF_for_single_pair(tensor_source.squeeze(0), tensor_fuse.squeeze(0))
        VIF_s.append(tmp)
    return VIF_s

def VIF_for_single_pair(tensor_source, tensor_fuse):
    return vifp_mscale(tensor_source, tensor_fuse).item()



if __name__ == "__main__":
    transform = transforms.ToTensor()
    # 创建MSRSDataset实例
    dataset = MSRSDataset(root_dir='/mnt/disk2/Shawalt/Demos/ImageFusion/DataSet/MSRS/train', transform=transform)

    # 创建DataLoader，设置批量大小、shuffle、并行加载等参数
    batch_size = 16  # 你可以根据GPU显存大小调整
    num_epochs = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 数据加载循环
    for epoch in range(num_epochs):
        for i, (vis_img_ycbcr, vis_img, ir_img, label_img) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, unit="batch")):
            val_ssim = (np.array(SSIM(ir_img, (vis_img_ycbcr[0] + ir_img)/2)) + np.array(SSIM(vis_img_ycbcr[0], (vis_img_ycbcr[0] + ir_img)/2))) / 2
            for i, val in enumerate(val_ssim):
                print("SSIM of img "+ str(i) +": " + str(val))
            # mutual_infs = mutual_infomation(ir_img, vis_img_ycbcr[0])
            # for i, mutual_infs in enumerate(mutual_infs):
            #     print('MI of img '+ str(i) + ": " + str(mutual_infs))
            noise = torch.normal(0, 0.15, size=ir_img.size())
            # val_scd = SCD(ir_img, vis_img_ycbcr[0], ir_img / 2 + vis_img_ycbcr[0] / 2)
            # for i, val in enumerate(val_scd):
            #     print("SCD of img "+ str(i) +": " + str(val))
            # val_ncc = NCC(ir_img, vis_img_ycbcr[0], ir_img + noise)
            # for i, val in enumerate(val_ncc):
            #     print("NCC of img "+ str(i) +": " + str(val))
            # val_psnr = PSNR(ir_img, vis_img_ycbcr[0], ir_img / 2 + vis_img_ycbcr[0] / 2)
            # for i, val in enumerate(val_psnr):
            #     print("PSNR of img "+ str(i) +": " + str(val))
            # val_ag = AG(ir_img / 2 + vis_img_ycbcr[0] / 2)
            # for i, val in enumerate(val_ag):
            #     print("AG of img "+ str(i) +": " + str(val))
            # val_sf = SF(ir_img / 2 + vis_img_ycbcr[0] / 2)
            # for i, val in enumerate(val_sf):
            #     print("SF of img "+ str(i) +": " + str(val))
            # val_N_abf = N_abf(ir_img, vis_img_ycbcr[0], vis_img_ycbcr[0] / 5 * 4 + ir_img / 5)
            # for i, val in enumerate(val_N_abf):
            #     print("N_abf of img "+ str(i) +": " + str(val))
            # val_VIF = VIFF(ir_img, vis_img_ycbcr[0], noise + ir_img / 2 + vis_img_ycbcr[0] / 5)
            # for i, val in enumerate(val_VIF):
            #     print("VIFF of img "+ str(i) +": " + str(val))
            
    
