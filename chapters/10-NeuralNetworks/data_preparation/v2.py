
torch.manual_seed(0)

# Build tensors BEFORE split
X_tensor = torch.as_tensor(X).float()
y_tensor = torch.as_tensor(y).float()

# Build dataset with ALL points
dataset = TensorDataset(X_tensor, y_tensor)

# Perform the split
ratio = 0.8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train
train_data, val_data = random_split(dataset, [n_train, n_val])

# Builds a loader of each set
train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=16,
    shuffle=True,
)
