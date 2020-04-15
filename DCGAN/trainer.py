def update_params(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
