import sys

sys.path.append("../python")
from needle.data import MNISTDataset, DataLoader
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), 
                              nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
        #hidden_dim = hidden_dim // 2
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:    
        model.eval()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error/sample_nums, np.mean(tot_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    )
    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for _ in range(epochs):
        train_loss, train_acc = epoch(train_dataloader, model, opt)
    
    test_loss, test_acc = epoch(test_dataloader, model)
    return train_loss, train_acc, test_loss, test_acc
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
