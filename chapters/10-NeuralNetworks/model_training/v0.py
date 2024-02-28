
# Define number of epochs
n_epochs = 1000

for epoch in range(n_epochs):
    model.train()

    # Step 1 - Compute the output
    yhat = model(X_train_tensor)

    # Step 2 - Compute the loss
    loss = loss_fn(yhat, y_train_tensor)

    # Step 3 - Compute the gradients for b and w
    loss.backward()

    # Step 4 - Update parameters
    optimizer.step()
    optimizer.zero_grad()
