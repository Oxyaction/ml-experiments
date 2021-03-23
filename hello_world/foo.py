import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

device = 'cuda'

training_data = datasets.FashionMNIST(
  root='data',
  train=True,
  download=True,
  transform=ToTensor()
)   

test_data = datasets.FashionMNIST(
  root='data',
  train=False,
  download=True,
  transform=ToTensor()
)

batch_size = 100


train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super(NeuralNetwork, self).__init__()

    self.stack = nn.Sequential(
      nn.Conv2d(1, 24, 3, 2, 1),
      nn.ReLU(),
      nn.Conv2d(24, 32, 3, 2, 1),
      nn.ReLU(),
      nn.Conv2d(32, 48, 3), # -> 48, 5, 5
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(1200, 512),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(512, 10),
      nn.Softmax(dim=1),
    )

  def forward(self, X):
    return self.stack(X)

def train(model, dataloader, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)

    pred = model.forward(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * len(x)
      print(f'loss: {loss:>7f}, [{current:>5d}/{size:>5d}]')


def test(model, dataloader):
  size = len(dataloader.dataset)
  accuracy = 0
  model.eval()
  with torch.no_grad():
    for x, y in dataloader:
      x, y = x.to(device), y.to(device)

      pred = model.forward(x)
      accuracy += (pred.argmax(1) == y).type(torch.uint8).sum().item()

  print(f'Accuracy: {(accuracy / size):>7f}')




model = NeuralNetwork().to(device)

# print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 5

for i in range(epochs):
  print(f'epoch {i}')
  train(model, train_dataloader, loss_fn, optimizer)

test(model, test_dataloader)


