
# Define the number of epochs
n_epochs = 200

# Define the learning rate
lr = 0.001

losses = []
val_losses = []

for epoch in range(n_epochs):
    loss = mini_batch(device, train_loader, train_step_fn)
    losses.append(loss)

    with torch.no_grad():
        val_loss = mini_batch(device, val_loader, val_step_fn)
        val_losses.append(val_loss)

    # Records both losses for each epoch under tag "loss"
    writer.add_scalars(
        main_tag="loss",
        tag_scalar_dict={
            "training": loss,
            "validation": val_loss
        },
        global_step=epoch
    )

# Close the writer
writer.close()
