
# Set device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Set learning rate
lr = 0.001

# Create model and send to device
model = nn.Sequential(nn.Linear(6,1)).to(device)

# Define an SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

# Define the MSE loss function
loss_fn = nn.MSELoss(reduction="mean")
