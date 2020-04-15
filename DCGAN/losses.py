import torch
from torch.nn import BCELoss, MSELoss

# BCE loss function
criterion = BCELoss()
feature_matching_criterion = MSELoss()


def scale(x, low, high):
    return (high-low) * x + low


def flip_label(label, pFlip):
    if torch.rand(1).item() < pFlip:
        label = 1 - label
        # print('flipped label')
    return label


def smooth_label(n, label, device):

    x = torch.rand(n, device=device)

    if label > .5:
        out = scale(x, .8, 1)
    else:
        out = scale(x, 0, .2)

    return out


def make_label(b_size, label, device, pFlip, label_smoothing):
    # flip label
    label = flip_label(label, pFlip)
    # generate lables
    if label_smoothing:
        target_label = smooth_label((b_size,), label, device)
    else:
        target_label = torch.full((b_size,), label, device=device)
    return target_label


def halfBCE(output, label, device, label_smoothing, pFlip):

    b_size = output.view(-1).size(0)
    target_label = make_label(b_size, label, device, pFlip, label_smoothing)
    loss = criterion(output.view(-1), target_label)

    return loss


def D_loss(output_real, output_fake, real_label, fake_label, pFlip,
           label_smoothing, device):
    # all-real batch
    errD_real = halfBCE(output_real,
                        real_label,
                        device, label_smoothing, pFlip)

    # all-fake batch
    errD_fake = halfBCE(output_fake,
                        fake_label,
                        device, label_smoothing, pFlip)

    return errD_real + errD_fake


def G_loss(output, real_label, label_smoothing, device):

    errG = halfBCE(output,
                   real_label,  # to train G we use real labels
                   device, label_smoothing, 0)

    return errG


def G_featMatch_loss(f1fake, f2fake, f1real, f2real, double_layer=False):

    fm_loss2 = feature_matching_criterion(f2fake, f2real)
    loss = fm_loss2

    if double_layer:
        fm_loss1 = feature_matching_criterion(f1fake, f1real)
        loss += fm_loss1

    return loss


def combined_G_loss(
        output, real_label, label_smoothing, device,
        f1fake, f2fake, f1real, f2real, double_layer=False):
    loss = .1*G_loss(output, real_label, label_smoothing, device)
    loss += G_featMatch_loss(
        f1fake, f2fake, f1real, f2real, double_layer
    )

    return loss
