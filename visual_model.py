import torch
import torch.nn as nn


class GeometricEncoder(nn.Module):
    def __init__(self,input_nc=3,z_dim=64,bottom=False):
        super().__init__()
        self.bottom = bottom

        if self.bottom:
            self.enc_down_0 = nn.Sequential([
                nn.Conv2d(input_nc + 4,z_dim,3,stride=1,padding=1),
                nn.ReLU(True)])
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc+4, z_dim, 3, stride=2 if bottom else 1,  padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True))

    def forward(self,x):
        """
        input:
            x: input image, [B,3,H,W]
        output:
            feature_map: [B,C,H,W]
        """
        W,H = x.shape[3], x.shape[2]
        X = torch.linspace(-1,1,W)
        Y = torch.linspace(-1,1,H)
        y1_m,x1_m = torch.meshgrid([Y,X])
        x2_m,y2_m = 2 - x1_m,2 - y1_m # Normalized distance in the four direction
        pixel_emb = torch.stack([x1_m,x2_m,y1_m,y2_m]).to(x.device).unsqueeze(0) # [1,4,H,W]
        pixel_emb = pixel_emb.repeat([x.size(0),1,1,1])
        inputs = torch.cat([x,pixel_emb],dim=1)

        if self.bottom:
            x_down_0 = self.enc_down_0(inputs)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(inputs)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        return feature_map

class FeatureDecoder(nn.Module):
    def __init__(self, inchannel,input_channel,object_dim = 100):
        super(FeatureDecoder, self).__init__()
        self.im_size = 128
        self.conv1 = nn.Conv2d(inchannel + 2, 32, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, bias=False)
        # self.bn4 = nn.BatchNorm2d(32)
        self.celu = nn.CELU()
        self.inchannel = inchannel
        self.conv5_img = nn.Conv2d(32, input_channel, 1)
        self.conv5_mask = nn.Conv2d(32, 1, 1)

        x = torch.linspace(-1, 1, self.im_size + 8)
        y = torch.linspace(-1, 1, self.im_size + 8)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
        self.bias = 0

    def forward(self, z):
        # z (bs, 32)
        bs,_ = z.shape

        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size + 8, self.im_size + 8)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
                       self.y_grid.expand(bs, -1, -1, -1), z), dim=1)
        # x (bs, 32, image_h, image_w)
        x = self.conv1(x);x = self.celu(x)
        # x = self.bn1(x)
        x = self.conv2(x);x = self.celu(x)
        # x = self.bn2(x)
        x = self.conv3(x);x = self.celu(x)
        # x = self.bn3(x)
        x = self.conv4(x);x = self.celu(x)
        # x = self.bn4(x)

        img = self.conv5_img(x)
        img = .5 + 0.5 * torch.tanh(img + self.bias)
        logitmask = self.conv5_mask(x)


        return img, logitmask
