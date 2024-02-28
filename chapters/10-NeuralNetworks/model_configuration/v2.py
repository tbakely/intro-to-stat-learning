
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the learning rate
lr = 0.001

torch.manual_seed(0)

# Now we create a model and send it at once to the device
model = nn.Sequential(nn.Linear(6,1)).to(device)

# Define an SGD optimizer to update the parameters
optimizer = optim.SGD(model.parameters(), lr=lr)

# Define the MSE loss function
loss_fn = nn.MSELoss(reduction="mean")

# Create the train_step function for our model, loss, and optimizer
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

# Create the val_step function for our model, loss
val_step_fn = make_val_step_fn(model, loss_fn)
