
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set learning rate
lr = 0.001

torch.manual_seed(0)

model = nn.Sequential(nn.Linear(6,1)).to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction="mean")

# Create train_step
train_step_fn = make_train_step_fn(model, loss_fn, optimizer)

# Create val step
val_step_fn = make_val_step_fn(model, loss_fn)

# Create a Summary Writer to interface with TensorBoard
writer = SummaryWriter("runs/simple_linear_regression")

# Fetches a single mini-batch so we can use add_graph
x_dummy, y_dummy = next(iter(train_loader))
writer.add_graph(model, x_dummy.to(device))
