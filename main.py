import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

train_dataset = pd.read_csv('train.csv')

train_x = train_dataset.drop(['label'], axis=1)
train_y = train_dataset['label']

train_x = torch.tensor(train_x.values, dtype=torch.float32)
train_y = torch.tensor(train_y.values)

train_x /= 255

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)



df_train_dataset = TensorDataset(x_train, y_train)
df_test_dataset = TensorDataset(x_test, y_test)

batch_size = 32
df_train = DataLoader(df_train_dataset, batch_size=batch_size, shuffle=True)
df_test = DataLoader(df_test_dataset, batch_size=batch_size, shuffle=True)

class Mnist_model(nn.Module):
  def __init__(self):
    super(Mnist_model, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1),
        nn.BatchNorm2d(5),
        nn.ReLU(),
        nn.Conv2d(5, 10, 3, 1),
        nn.BatchNorm2d(10),
        nn.ReLU(),
        nn.Conv2d(10, 15, 3, 1),
        nn.BatchNorm2d(15),
        nn.ReLU(),
        nn.Conv2d(15, 10, 3, 1),
        nn.BatchNorm2d(10),
        nn.ReLU(),
        nn.Conv2d(10, 3, 5, 1),
        nn.BatchNorm2d(3),
        nn.ReLU()
    )
    self.linear = nn.Sequential(
        nn.Linear(768, 450),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(450, 100),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(100, 10),
    )

  def forward(self, x):
    x = x.view(-1, 1, 28,28)
    y = self.conv(x)
    y = y.view(-1, 3*16*16)
    y = self.linear(y)
    return y

model = Mnist_model()
cross = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(5):
  model.train()
  i = 0
  for image, label in df_train:
    pred = model(image)
    loss_model = cross(pred, label)
    if (i % 200) == 0:
      print(loss_model)
    optimizer.zero_grad()
    loss_model.backward()
    optimizer.step()
    i+=1
  print(epoch)
submission_date = pd.read_csv('test.csv')
tensor_date = torch.tensor(submission_date.values, dtype=torch.float32)
tensor_date /= 255

datasets = TensorDataset(tensor_date)
loader_test = DataLoader(datasets, batch_size=batch_size)

new_tensor = torch.Tensor()
for image in loader_test:
  pred = model(image[0])
  max_index = torch.argmax(pred, dim=1)
  new_tensor = torch.cat((new_tensor, max_index))
print(new_tensor.shape)
print(tensor_date.shape)
print(new_tensor[-1])

sample = pd.read_csv("sample_submission.csv")
sample['Label'] = new_tensor.numpy()
sample['Label'] = sample['Label'].astype(int)
sample.to_csv('submission.csv', index=False)

from google.colab import files
files.download('submission.csv')
