import numpy as np
import torch
from torch import nn
import time


class GAN():
    """GAN model class encapsulating generator and discriminator
        along with the common operations needed for training and testing
        a GAN model
    """

    def __init__(self, generator, discriminator, device):
        """Initialize a GAN by passing the initialized discriminator a
        nd generator network

        Parameters
        ----------
        generator : instance of nn.Module
            it should already initialized
        discriminator : instance of nn.Module
            it should already initialized
        device : [type]
            the device to compute on
        """
        # initialized generator
        self.generator = generator
        # initialized discriminator
        self.discriminator = discriminator
        self.device = device

    def discriminate(self, batch_img):
        """Forward pass of the discriminator

        Parameters
        ----------
        batch_img : torch tensor
            Containing batch of images

        Returns
        ------
        torch tensor
            discriminator output for the batch
        """
        raise NotImplementedError

    def generate(self, generator_input):
        """Forward pass of the generator

        Parameters
        ----------
        generator_input : torch tensor
            contains the generator input

        Returns
        ------
        torch tensor
            generated images
        """
        raise NotImplementedError

    def fake_batch(self, batch_size):
        """generate a batch of fake images

        Parameters
        ----------
        batch_size : int
            batch size of the fake images

        Returns
        -------
        tensor
            batch of fake images
        """
        raise NotImplementedError

    def discriminator_loss(self, real_batch, fake_batch):
        """calculate loss function of the discriminator

        Parameters
        ----------
        real_batch : torch tensor
            batch of the real images
        fake_batch : torch tensor
            batch of fake samples

        Returns
        ------
        scalar
            scalar loss function of the discriminator
        """
        raise NotImplementedError

    def generator_loss(self, real_batch, fake_batch):
        """calculate loss function of the discriminator

        Parameters
        ----------
        real_batch : torch tensor
            batch of the real images
        fake_batch : torch tensor
            batch of fake samples

        Returns
        -------
        scalar
            scalar loss function of the generator
        """
        raise NotImplementedError


class MinibatchDiscrimination(nn.Module):
    """Minibatch discriminator class used for stabilizing GAN
    """

    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(
            in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x


class trainer_GAN():
    """Contains common operations for training a GAN
    """

    def __init__(self,
                 gan, train_loader,
                 disc_optim, gen_optim,
                 D_loss_ceiling=.8, epochs=5
                 ):
        self.gan = gan
        self.train_loader = train_loader
        self.disc_optim = disc_optim
        self.gen_optim = gen_optim
        self.real_batch = None
        self.batch_size = None
        self.D_loss_ceiling = D_loss_ceiling
        self.epochs = epochs
        self.errD = np.inf
        self.errG = np.inf

    def train(self):
        for i in range(self.epochs):

            epoch_start = time.time()

            self.train_epoch()

            print("epoch {}/{} time {}".format(
                i, self.epochs, time.time()-epoch_start))

    def train_epoch(self):

        batch_start = time.time()
        for ix, data in enumerate(self.train_loader, 0):

            self.real_batch = data[0].to(self.gan.device)
            self.batch_size = self.real_batch.size(0)
            self.train_batch()

            if ix % 50 == 0:
                print('[{:0>4}/{:0>4}]  Loss_D: {:.3f}  Loss_G: {:.3f} t={:6.3f}'.format(
                    ix, len(self.train_loader), self.errD, self.errG, time.time()-batch_start, prec=3))
                batch_start = time.time()

    def train_batch(self):
        errs = []
        self.train_discriminator()
        errs.append(self.errD)
        # if discriminator loss is low enough, trian generator
        if self.errD < self.D_loss_ceiling:
            self.train_generator()
        errs.append(self.errG)

    def train_discriminator(self):
        fake_batch = self.fake_batch_gen()

        # Detach in fake_batch.detach() serves the purpose of avoiding
        # gradient computation w.r.t params of G. Note that even if do not
        # detach, the training of D is normal as the disc_optim should have
        # been defined only on parameters of D.
        errD = self.gan.discriminator_loss(self.real_batch,
                                           fake_batch.detach())

        self.update_params(self.disc_optim, errD)
        self.errD = errD.item()

    def train_generator(self):
        fake_batch = self.fake_batch_gen()
        # Detach is not needed for fake_batch because you are training
        # the G network
        errG = self.gan.generator_loss(self.real_batch,
                                       fake_batch)
        self.update_params(self.gen_optim, errG)
        self.errG = errG.item()

    def fake_batch_gen(self):
        """return a batch of fake images
        it calls the fake batch function of the gan model

        Returns
        ------
        torch tensor
            batch of fake samples
        """
        return self.gan.fake_batch(self.batch_size)

    def update_params(self, optimizer, loss):
        """update parameters of a network
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
