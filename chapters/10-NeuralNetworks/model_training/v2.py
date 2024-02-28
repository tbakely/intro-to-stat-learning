
# Define epochs
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    # mini batch loop
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        # Send each mini batch to device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        mini_batch_loss = train_step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    
    # average loss over all mini batches
    loss = np.mean(mini_batch_losses)
    losses.append(loss)
