import torch
from torch.nn import BCELoss, MSELoss

# BCE loss function
criterion = BCELoss()
feature_matching_criterion = MSELoss()


def halfBCE(output, label, device):

    b_size = output.view(-1).size(0)
    target_label = torch.full((b_size,), label, device=device)
    loss = criterion(output.view(-1), target_label)

    return loss


def D_loss(real_batch, fake_batch, disc, gen, real_label, fake_label, device, featMatch=False):
    # all-real batch
    output_real, f1, f2 = disc(real_batch)
    errD_real = halfBCE(output_real,
                        real_label,
                        device)

    # all-fake batch
    output_fake, _, _ = disc(fake_batch.detach())
    errD_fake = halfBCE(output_fake,
                        fake_label,
                        device)

    if featMatch:
        return errD_real + errD_fake, errD_real, output_fake, f1.detach(), f2.detach()
    else:
        return errD_real + errD_fake, errD_real, output_fake


def G_loss(fake_batch, disc, real_label, device, featMatch=False):
    output, _, _ = disc(fake_batch)
    errG = halfBCE(output,
                   real_label,  # to train G we use real labels
                   device)

    return errG, output


def G_featMatch_loss(fake_batch, disc, real_label, device, f1real, f2real):
    output, f1, f2 = disc(fake_batch)
    fm_loss1 = feature_matching_criterion(f1, f1real)
    fm_loss2 = feature_matching_criterion(f2, f2real)

    return fm_loss1 + fm_loss1, output
