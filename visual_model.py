import torch
import torch.nn as nn

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

        self.object_score_marker   = nn.Linear(128 * 128 * 32,1)
        #self.object_score_marker   = FCBlock(256,2,64 * 64 * 16,1)
        #self.object_feature_marker = FCBlock(256,3,64 * 64 * 16,object_dim)
        self.object_feature_marker = nn.Linear(inchannel,object_dim)
        self.conv_features         = nn.Conv2d(32,16,3,2,1)


    def forward(self, z):
        # z (bs, 32)
        bs,_ = z.shape
        object_features = self.object_feature_marker(z)
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

        conv_features = x.flatten(start_dim=1)
        
        object_scores = torch.sigmoid( 0.0001 *  self.object_score_marker(conv_features)) 

        return img, logitmask, object_features,object_scores
