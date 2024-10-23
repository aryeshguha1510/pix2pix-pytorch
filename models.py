import torch
import torch.nn as nn

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, input_nc, target_nc, ngf):
        super(_netG, self).__init__()
        # Encoder (Downsampling)
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1, bias=False),
        )
        self.encoder_2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
        )
        self.encoder_3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
        )
        self.encoder_4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.encoder_5 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.encoder_6 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.encoder_7 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.encoder_8 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )

        # Decoder (Upsampling)
        self.decoder_8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.decoder_7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.decoder_6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.decoder_5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
        )
        self.decoder_4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
        )
        self.decoder_3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
        )
        self.decoder_2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
        )
        self.decoder_1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, target_nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        # Encoding layers (downsampling)
        output_e1 = self.encoder_1(input)
        output_e2 = self.encoder_2(output_e1)
        output_e3 = self.encoder_3(output_e2)
        output_e4 = self.encoder_4(output_e3)
        output_e5 = self.encoder_5(output_e4)
        output_e6 = self.encoder_6(output_e5)
        output_e7 = self.encoder_7(output_e6)
        output_e8 = self.encoder_8(output_e7)

        # Decoding layers (upsampling)
        output_d8 = self.decoder_8(output_e8)
        output_d7 = self.decoder_7(torch.cat((output_d8, output_e7), 1))
        output_d6 = self.decoder_6(torch.cat((output_d7, output_e6), 1))
        output_d5 = self.decoder_5(torch.cat((output_d6, output_e5), 1))
        output_d4 = self.decoder_4(torch.cat((output_d5, output_e4), 1))
        output_d3 = self.decoder_3(torch.cat((output_d4, output_e3), 1))
        output_d2 = self.decoder_2(torch.cat((output_d3, output_e2), 1))
        output_d1 = self.decoder_1(torch.cat((output_d2, output_e1), 1))

        return output_d1

class _netD(nn.Module):
    def __init__(self, input_nc, target_nc, ndf):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_nc + target_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, target):
        return self.main(torch.cat((input, target), 1))
