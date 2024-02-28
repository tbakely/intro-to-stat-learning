
# This time we also include the validatiion step

# Define number of epochs
n_epochs = 200

losses = []
val_losses = []

for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    # VALIDATION - no gradients because its in validation!
    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)
