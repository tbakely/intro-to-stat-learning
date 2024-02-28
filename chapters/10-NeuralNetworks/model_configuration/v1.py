
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Set learning rate
lr = 0.001

torch.manual_seed(0)

# Define model
model = nn.Sequential(nn.Linear(6,1)).to(device)

# Define SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# Define the MSE loss function
loss_fn = nn.MSELoss(reduction="mean")

# Create the train_step function for our model, loss, and optimizer
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)
