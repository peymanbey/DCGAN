import torch


def update_params(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def batch_fake_samples(batchSize, network, nz, device):
    # Generate batch of latent vectors
    noise = torch.randn(batchSize, nz, 1, 1, device=device)
    # Generate fake image batch with network
    return network(noise)
