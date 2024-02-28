
# Define number of epochs
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    loss = train_step_fn(X_train_tensor, y_train_tensor)
    losses.append(loss)
