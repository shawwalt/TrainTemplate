import torch
import torch.nn as nn
import torch.nn.functional as F

class DDcGANGenerator(nn.Module):
    def __init__(self):
        super(DDcGANGenerator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, vis, ir):
        fused_feature = self.encoder(torch.concat([vis, ir], dim=1))
        recon_img = self.decoder(fused_feature)
        return recon_img
    
class DDcGANDiscriminator(nn.Module):
    def __init__(self, base_channel=16):
        super(DDcGANDiscriminator, self).__init__()
        # input 224*224
        self.conv1 = nn.Conv2d(1, base_channel, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channel, 2*base_channel, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2*base_channel, 4*base_channel, kernel_size=3, stride=2, padding=1)

        self.feature_extract = nn.Sequential(
            self.conv1, nn.BatchNorm2d(base_channel), nn.ReLU(),
            self.conv2, nn.BatchNorm2d(2*base_channel), nn.ReLU(),
            self.conv3, nn.BatchNorm2d(4*base_channel), nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feature = self.feature_extract(x)
        return self.fc(torch.flatten(feature, start_dim=1, end_dim=-1))
        

class Encoder(nn.Module):
    def __init__(self, base_channel=48, depth=4):
        super(Encoder, self).__init__()

        # 定义网络层
        self.conv1_1 = self.conv_2d_act(2, base_channel, 3, nn.ReLU())
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
            activation,
            nn.BatchNorm2d(out_channel)
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
    def __init__(self, original_in_channels=240, base_channel=32, depth=2):
        super(Decoder, self).__init__()

        # 定义网络层
        self.in_conv = nn.Sequential(
            nn.Conv2d(original_in_channels, original_in_channels, 3, padding=1),
            nn.Conv2d(original_in_channels, base_channel * (2**depth), 3, padding=1)
        )
        
        self.decode_layers = nn.ModuleList()
        in_channels = base_channel * (2**depth)
        for i in range(depth):
            self.decode_layers.append(
                self.conv_2d_act(in_channels, in_channels // 2, 3, nn.ReLU())
            )
            in_channels = in_channels // 2
        self.final_conv = self.conv_2d_act(base_channel, 1, 3, nn.Sigmoid())


    def conv_2d_act(self, in_channel, out_channel, kernel_size, activation):
        if not isinstance(activation, nn.Module):
            raise TypeError('activation should be nn.Module')
        
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size),
            activation,
            nn.BatchNorm2d(out_channel),
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
        trans_feature = self.in_conv(image)
        out = self.decode(trans_feature)
        return out
    

# *****************模块测试代码******************************

if __name__ == "__main__":
    encoder = Encoder()
    decoder = Decoder()
    model = DDcGANGenerator()
    d_model = DDcGANDiscriminator()
    input_tensor = torch.randn(1, 1, 32, 32)
    for layer in encoder.children():
        print(layer)
    for layer in decoder.children():
        print(layer)

    # 创建一个测试输入 (batch_size=1, channels=1, height=32, width=32)
    input_tensor_1 = torch.randn(8, 1, 224,224)
    input_tensor_2 = torch.randn(8, 1, 224, 224)
    input_tensor = torch.randn(8, 2, 32, 32)

    # 进行前向传播
    output_1 = encoder(input_tensor)
    output_2 = encoder(input_tensor)
    output = model(input_tensor_1, input_tensor_2)
    d_output = d_model(input_tensor_1)
    print(output_1.size())

    print(output.size())
    print(d_output.size())