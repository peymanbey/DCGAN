from losses import combined_G_loss
from losses import D_loss
import torch.nn as nn
import torch
from gan import GAN, MinibatchDiscrimination


class DCGAN(GAN):
    def __init__(self, generator, discriminator, device,
                 real_label, fake_label, pFlip,
                 label_smoothing, double_layer=False):
        super(DCGAN, self).__init__(generator, discriminator, device)
        self.real_label = real_label
        self.fake_label = fake_label
        self.pFlip = pFlip
        self.label_smoothing = label_smoothing
        self.double_layer = double_layer

    def discriminate(self, batch_img):
        return self.discriminator(batch_img)

    def generate(self, latent_vec):
        return self.generator(latent_vec)

    def fake_batch(self, batchSize):
        # Generate a batch of random latent vectors
        noise = torch.randn(batchSize, self.generator.nz, 1, 1,
                            device=self.device)
        # Generate fake image batch with network
        return self.generate(noise)

    def discriminator_loss(self, real_batch, fake_batch):
        output_real, f1real, f2real = self.discriminate(real_batch)
        output_fake, _, _ = self.discriminate(fake_batch.detach())
        return D_loss(output_real, output_fake,
                      self.real_label, self.fake_label, self.pFlip,
                      self.label_smoothing, self.device)

    def generator_loss(self, real_batch, fake_batch):
        _, f1real, f2real = self.discriminate(real_batch)
        output, f1fake, f2fake = self.discriminate(fake_batch)
        return combined_G_loss(output, self.real_label, self.label_smoothing,
                               self.device,
                               f1fake, f2fake,
                               f1real, f2real,
                               self.double_layer)


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        # input random vector size
        self.nz = nz
        # output channel size
        self.nc = nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.features1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
        )

        self.features2 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        self.classifier = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        f1 = self.features1(input)
        f2 = self.features2(f1)
        cls = self.classifier(f2)
        return cls, f1, f2


def weights_init(m):
    """
    Custom weight initializer from the DCGAN papar
    Different from the usual way of initializing the weigths

    Params:
    -----------------------------------
    m:: network object

    -----------------------------------
    DCGAN paper arXiv:1511.06434
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DiscriminatorMiniBatchDiscrimination(Discriminator):
    def __init__(self, ngpu, nc, ndf):
        super(DiscriminatorMiniBatchDiscrimination,
              self).__init__(ngpu, nc, ndf)

        self.classifier = nn.Sequential(
            # Minibatch discrimination
            # state size. (ndf*8) x 4 x 4
            MinibatchDiscrimination(ndf * 8 * 4 * 4, ndf, ndf, mean=False),
            nn.Linear(ndf * 8 * 4 * 4 + ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        f1 = self.features1(input)
        f2 = self.features2(f1)
        # minibatch discrimination
        cls = self.classifier(f2.view(f2.shape[0], -1))
        return cls, f1, f2
