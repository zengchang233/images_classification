import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import dataset
import simple_net

def train_one_epoch(network, criterion, trainloader, optimizer):
    network.train()
    losses = []
    correct = 0
    total = 0
    for idx, (feature, label) in enumerate(trainloader):
        optimizer.zero_grad()
        output = network(feature)
        _, ind = torch.max(output, dim = 1)
        correct += (ind == label).sum().item()
        total += len(label)
        loss = criterion(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        message = '\r[{:5d}/{:5d}({:3.0%})] train loss: {:.2f}\ttrain acc: {:.2%}'.format(len(label) * idx, 40000, len(label) * idx / 40000, loss, correct / total)
        print(message, end = '')
    print()
    message = 'Train Avg loss: {:.2f}\tTrain Acc: {:.2%}'.format(sum(losses) / len(losses), correct / total)
    print(message)

def valid(network, validloader):
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (feature, label) in validloader:
            output = network(feature)
            _, idx = torch.max(output, dim = 1)
            correct += (idx == label).sum().item()
            total += len(label)
        message = 'Valid Acc: {:.2%}'.format(correct / total)
        print(message)

def train(network, criterion, trainloader, validloader, optimizer, scheduler, start_epoch = 0, n_epochs = 20):
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        train_one_epoch(network, criterion, trainloader, optimizer)
        scheduler.step()

        if (epoch + 1) % 3 == 0:
            valid(network, validloader)
    torch.save({'state_dict': network,
                'optimizer': optimizer.state_dict()},
                'checkpoint.pth')

def main():
    trainset = dataset.Trainset()
    validset = dataset.Trainset(training = False)
    trainloader = DataLoader(trainset, batch_size = 64, shuffle = True, num_workers = 4)
    validloader = DataLoader(validset, batch_size = 16, shuffle = True, num_workers = 4)
    network = simple_net.SimpleNet()
    optimizer = optim.SGD(network.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5, last_epoch = -1)
    train(network, criterion, trainloader, validloader, optimizer, scheduler)

if __name__ == "__main__":
    main()