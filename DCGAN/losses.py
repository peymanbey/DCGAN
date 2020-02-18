import torch


def halfBCE(output, label, criterion, device):

    b_size = output.view(-1).size(0)
    target_label = torch.full((b_size,), label, device=device)
    loss = criterion(output.view(-1), target_label)

    return loss


def D_loss(real_batch, fake_batch, disc, gen, real_label, fake_label, criterion, device):
    # all-real batch
    output_real, _, _ = disc(real_batch)
    errD_real = halfBCE(output_real,
                        real_label,
                        criterion,
                        device)

    # all-fake batch
    output_fake, _, _ = disc(fake_batch.detach())
    errD_fake = halfBCE(output_fake,
                        fake_label,
                        criterion,
                        device)

    return errD_real + errD_fake, errD_real, output_fake


def G_loss(fake_batch, disc, real_label, criterion, device):
    output, _, _ = disc(fake_batch)
    errG = halfBCE(output,
                   real_label,  # to train G we use real labels
                   criterion,
                   device)

    return errG, output
