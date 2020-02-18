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


def D_loss(output_real, output_fake, real_label, fake_label, device):
    # all-real batch
    errD_real = halfBCE(output_real,
                        real_label,
                        device)

    # all-fake batch
    errD_fake = halfBCE(output_fake,
                        fake_label,
                        device)

    return errD_real + errD_fake


def G_loss(output, real_label, device):

    errG = halfBCE(output,
                   real_label,  # to train G we use real labels
                   device)

    return errG


def G_featMatch_loss(f1fake, f2fake, f1real, f2real):

    fm_loss1 = feature_matching_criterion(f1fake, f1real)
    fm_loss2 = feature_matching_criterion(f2fake, f2real)

    return fm_loss1 + fm_loss2
