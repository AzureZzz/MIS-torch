import torch
import torch.nn.functional as F
from torchvision import models

from utils.model_components import *

from channel_unet import myChannelUnet


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        filters = [32, 64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input = DoubleConv(in_channels, filters[0])

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(filters[3], filters[4])
        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.conv5 = DoubleConv(filters[4], filters[3])
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.conv6 = DoubleConv(filters[3], filters[2])
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv7 = DoubleConv(filters[2], filters[1])
        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv8 = DoubleConv(filters[1], filters[0])

        self.output = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        input = self.input(x)
        p1 = self.pool1(input)
        c1 = self.conv1(p1)
        p2 = self.pool2(c1)
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.conv3(p3)
        p4 = self.pool4(c3)
        c4 = self.conv4(p4)
        up1 = self.up1(c4)
        c5 = self.conv5(torch.cat([up1, c3], dim=1))
        up2 = self.up2(c5)
        c6 = self.conv6(torch.cat([up2, c2], dim=1))
        up3 = self.up3(c6)
        c7 = self.conv7(torch.cat([up3, c1], dim=1))
        up4 = self.up4(c7)
        c8 = self.conv8(torch.cat([up4, input], dim=1))
        out = self.output(c8)
        return nn.Sigmoid()(out)


class XceptionUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XceptionUnet, self).__init__()
        filters = [32, 64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        #input: 640x640x3
        self.input = DoubleConv(in_channels, filters[0]) #640x640x32
        self.pool = nn.MaxPool2d(2) #320x320x32
        self.down1 = XceptionDownBlock(filters[0], filters[1], 2, 2, start_with_relu=False, grow_first=True)    #160x160x64
        self.down2 = XceptionDownBlock(filters[1], filters[2], 2, 2, start_with_relu=False, grow_first=True)    #80x80x128
        self.down3 = XceptionDownBlock(filters[2], filters[3], 2, 2, start_with_relu=False, grow_first=True)    #40x40x256
        self.down4 = XceptionDownBlock(filters[3], filters[4], 2, 2, start_with_relu=False, grow_first=True)    #20x20x512

        self.up1 = XceptionUpBlock(filters[4], filters[3], 2, 2, start_with_relu=False, grow_first=True)    #40x40x256
        self.up2 = XceptionUpBlock(filters[3], filters[2], 2, 2, start_with_relu=False, grow_first=True)    #80x80x128
        self.up3 = XceptionUpBlock(filters[2], filters[1], 2, 2, start_with_relu=False, grow_first=True)    #160x160x64
        self.up4 = XceptionUpBlock(filters[1], filters[0], 2, 2, start_with_relu=False, grow_first=True)    #320x320x32
        self.up5 = XceptionUpBlock(filters[0], self.out_channels, 2, 2, start_with_relu=False, grow_first=True)  #640x640x3

    def forward(self, x):
        input = self.input(x)
        pool = self.pool(input)
        down1 = self.down1(pool)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        up1 = self.up1(down4)
        up2 = self.up2(down3+up1)
        up3 = self.up3(down2+up2)
        up4 = self.up4(down1+up3)
        out = self.up5(pool+up4)
        return nn.Sigmoid()(out)


class MyDenseUNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MyDenseUNet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input = DoubleConv(in_channels, filters[0])

        self.pool1 = nn.MaxPool2d(2)
        self.pool1_4 = nn.MaxPool2d(4)
        self.pool1_8 = nn.MaxPool2d(8)

        self.conv1 = DoubleConv(filters[0], filters[1])
        self.up2_1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool2_4 = nn.MaxPool2d(4)

        self.conv2 = DoubleConv(filters[1], filters[2])
        self.up3_1 = nn.ConvTranspose2d(filters[2], filters[0], 4, stride=4)
        self.up3_2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.pool3 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(filters[2], filters[3])

        self.up4_1 = nn.ConvTranspose2d(filters[3], filters[0], 8, stride=8)
        self.up4_2 = nn.ConvTranspose2d(filters[3], filters[1], 4, stride=4)
        self.up4_3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(filters[3], filters[4])

        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.conv5 = DoubleConv(filters[3]*2 + filters[2] + filters[1] + filters[0], filters[3])
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.conv6 = DoubleConv(filters[2]*3 + filters[1] + filters[0], filters[2])
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.conv7 = DoubleConv(filters[1]*4 + filters[0], filters[1])
        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.conv8 = DoubleConv(filters[0]*5, filters[0])

        self.output = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        input = self.input(x)
        p1 = self.pool1(input)
        p1_4 = self.pool1_4(input)
        p1_8 = self.pool1_8(input)

        c1 = self.conv1(p1)
        up2_1 = self.up2_1(c1)
        p2 = self.pool2(c1)
        p2_4 = self.pool2_4(c1)

        c2 = self.conv2(p2)
        up3_1 = self.up3_1(c2)
        up3_2 = self.up3_2(c2)
        p3 = self.pool3(c2)

        c3 = self.conv3(p3)
        up4_1 = self.up4_1(c3)
        up4_2 = self.up4_2(c3)
        up4_3 = self.up4_3(c3)
        p4 = self.pool4(c3)

        c4 = self.conv4(p4)
        up1 = self.up1(c4)
        c5 = self.conv5(torch.cat([up1, c3, p3, p2_4, p1_8], dim=1))
        up2 = self.up2(c5)
        c6 = self.conv6(torch.cat([up2, up4_3, c2, p2, p1_4], dim=1))
        up3 = self.up3(c6)
        c7 = self.conv7(torch.cat([up3, up4_2, up3_2, c1, p1], dim=1))
        up4 = self.up4(c7)
        c8 = self.conv8(torch.cat([up4, up4_1, up3_1, up2_1, input], dim=1))
        out = self.output(c8)
        return nn.Sigmoid()(out)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        output = self.sigmoid(output)
        return output


class UNetResNet34(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(UNetResNet34, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.input = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final_deconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.input(x)
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_maxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        return nn.Sigmoid()(out)


class FCN8sVGG(nn.Module):

    def __init__(self, in_channels, n_class, pretrained_net):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = n_class
        self.n_class = n_class
        self.pretrained_net = pretrained_net

        self.input = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x = self.input(x)
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)
        score = nn.Sigmoid()(score)
        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN16sVGG(nn.Module):

    def __init__(self, in_channels, n_class, pretrained_net):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = n_class
        self.n_class = n_class
        self.pretrained_net = pretrained_net

        self.input = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        x = self.input(x)
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)
        score = nn.Sigmoid()(score)
        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN32sVGG(nn.Module):

    def __init__(self, in_channels, n_class, pretrained_net):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = n_class
        self.n_class = n_class
        self.pretrained_net = pretrained_net

        self.input = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        output = self.pretrained_net(x)
        x4 = output['x4']  # size=(N, 512, x.H/32, x.W/32)
        score = self.bn1(self.relu(self.deconv1(x4)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        # score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)
        score = self.sigmoid(score)
        return score  # size=(N, n_class, x.H/1, x.W/1)


class SegNet(nn.Module):
    def __init__(self, input_nbr, label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=1, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=1, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=1, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=1, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=1, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=1)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=1)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=1)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=1)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=1)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)
        x11d = self.sigmoid(x11d)
        return x11d


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(ch_in=img_ch, ch_out=64)
        self.Conv2 = DoubleConv(ch_in=64, ch_out=128)
        self.Conv3 = DoubleConv(ch_in=128, ch_out=256)
        self.Conv4 = DoubleConv(ch_in=256, ch_out=512)
        self.Conv5 = DoubleConv(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = DoubleConv(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = DoubleConv(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = DoubleConv(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = DoubleConv(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        return d1


# R2U_Net start
class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class R2U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.sigmoid(d1)
        return d1


def get_model(model, device, in_channels, out_channels):
    if model == 'UNet':
        model = UNet(in_channels, out_channels).to(device)
    if model == 'MyDenseUNet':
        model = MyDenseUNet(in_channels, out_channels).to(device)
    if model == 'ResNet34_UNet':
        model = UNetResNet34(in_channels, out_channels, pretrained=True).to(device)
    if model == 'UNet++':
        model = UNetPlusPlus(in_channels, out_channels).to(device)
    if model == 'Attention_UNet':
        model = AttU_Net(in_channels, out_channels).to(device)
    if model == 'SegNet':
        model = SegNet(in_channels, out_channels).to(device)
    if model == 'R2UNet':
        model = R2U_Net(in_channels, out_channels).to(device)
    if model == 'MyChannelUNet':
        model = myChannelUnet(in_channels, out_channels).to(device)
    if model == 'FCN32s':
        pretrained_net = VGGNet()
        model = FCN32sVGG(in_channels, out_channels, pretrained_net).to(device)
    if model == 'FCN16s':
        pretrained_net = VGGNet()
        model = FCN16sVGG(in_channels, out_channels, pretrained_net).to(device)
    if model == 'FCN8s':
        # assert dataset != 'esophagus', \
        #     "fcn8s模型不能用于数据集esophagus，因为esophagus数据集为80x80，经过5次的2倍降采样后剩下2.5x2.5，分辨率不能为小数" \
        #     "建议把数据集resize成更高的分辨率再用于fcn"
        pretrained_net = VGGNet()
        model = FCN8sVGG(in_channels, out_channels, pretrained_net).to(device)
    if model == 'CENet':
        from cenet import CE_Net_
        model = CE_Net_().to(device)
    if model == 'XceptionUnet':
        model = XceptionUnet(in_channels, out_channels).to(device)
    return model
