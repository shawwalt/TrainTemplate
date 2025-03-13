import torch
import torch.nn as nn
import torch.nn.functional as F

def fusion_strategy(vi_feature, ir_feature):
    return (vi_feature + ir_feature) / 2

def fusion_strategy_l1(vi_feature, ir_feature):
    norm_vi = torch.norm(vi_feature, p=1, dim=1).unsqueeze(1)
    norm_ir = torch.norm(ir_feature, p=1, dim=1).unsqueeze(1)
    kernel = torch.tensor([[[[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]]], dtype=torch.float32).to(vi_feature.device)
    norm_ir = F.conv2d(norm_ir, kernel, padding=1)
    norm_vi = F.conv2d(norm_vi, kernel, padding=1)
    norm = torch.concat([norm_vi, norm_vi], dim=1)
    norm_val = torch.norm(norm, p=1, dim=1).unsqueeze(1)
    norm = norm / norm_val
    # print(norm[:, 0, :, :]  + norm[:, 1, :, :])
    fuse_feature = norm[:, 0, :, :].unsqueeze(1) * vi_feature + norm[:, 1, :, :].unsqueeze(1) * ir_feature
    return fuse_feature

class Encoder(nn.Module):
    def __init__(self, base_channel=16, depth=3):
        super(Encoder, self).__init__()

        # 定义网络层
        self.conv1_1 = self.conv_2d_act(1, base_channel, 3, nn.ReLU())
        self.dense_convs = nn.ModuleList()
        for i in range(depth):
            self.dense_convs.append(
                self.conv_2d_act((i+1) * base_channel, base_channel, 3, nn.ReLU())
            ) # 第i层输入是前i-1层输出与原始输入的叠加
        

    def conv_2d_act(self, in_channel, out_channel, kernel_size, activation):
        if not isinstance(activation, nn.Module):
            raise TypeError('activation should be nn.Module')
        
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=0),
            activation
        )


    def encode(self, image):
        image = F.pad(image, (1, 1, 1, 1), 'reflect')
        out = self.conv1_1(image)
        for i in range(len(self.dense_convs)):
            out_pad = F.pad(out, (1, 1, 1, 1), 'reflect')
            tmp = self.dense_convs[i](out_pad)
            out = torch.concat([out, tmp], dim=1)
        return out


    def forward(self, image):
        out = self.encode(image)
        return out
    

class Decoder(nn.Module):
    def __init__(self, original_in_channels=64, base_channel=16, depth=3):
        super(Decoder, self).__init__()

        # 定义网络层
        self.decode_layers = nn.ModuleList()
        in_channels = original_in_channels
        for i in range(depth):
            self.decode_layers.append(
                self.conv_2d_act(in_channels, in_channels // 2, 3, nn.ReLU())
            )
            in_channels = in_channels // 2
        self.final_conv = self.conv_2d_act(original_in_channels // (2 ** depth), 1, 3, nn.Sigmoid())


    def conv_2d_act(self, in_channel, out_channel, kernel_size, activation):
        if not isinstance(activation, nn.Module):
            raise TypeError('activation should be nn.Module')
        
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size),
            activation
        )


    def decode(self, image):
        out = image
        for i in range(len(self.decode_layers)):
            out = F.pad(out, (1,1,1,1), mode='reflect')
            out = self.decode_layers[i](out)
        out = F.pad(out, (1,1,1,1), mode='reflect')
        out = self.final_conv(out)
        return out


    def forward(self, image):
        out = self.decode(image)
        return out

class DenseFuse(nn.Module):
    def __init__(self, is_training=True):
        super(DenseFuse, self).__init__()
        self.is_training = is_training
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, vi, ir):
        vi_feature = self.encoder(vi)
        ir_feature = self.encoder(ir)
        if not self.is_training:
            feature_fuse = fusion_strategy_l1(vi_feature, ir_feature)
            output = self.decoder(feature_fuse)
        else:
            features = [vi_feature, ir_feature]
            output = [
                self.decoder(vi_feature),
                self.decoder(ir_feature)
            ]
        return output

# *****************模块测试代码******************************

if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()
    input_tensor = torch.randn(1, 1, 32, 32)
    for layer in encoder.children():
        print(layer)
    for layer in decoder.children():
        print(layer)

    # 创建一个测试输入 (batch_size=1, channels=1, height=32, width=32)
    input_tensor_1 = torch.randn(8, 1, 32, 32)
    input_tensor_2 = torch.randn(8, 1, 32, 32)

    # 进行前向传播
    output_1 = encoder(input_tensor_1)
    output_2 = encoder(input_tensor_2)
    print(output_1.size())
    output = fusion_strategy_l1(output_1, output_2)

    output = decoder(output)
    print(output.size())