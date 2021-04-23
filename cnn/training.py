
import pytorch_lightning as pl
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch

class Data(pl.LightningDataModule):
    def prepare_data(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
      
        self.train_data = datasets.MNIST('', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST('', train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size= 32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size= 1024, shuffle=True)


def get_conv_net(inplanes):
    hidden_size = 128
    kernel_size = 3
    pool_size = 2
    regressor = nn.Sequential(
        nn.Conv2d(inplanes, hidden_size, kernel_size, stride=1, padding=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
        nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=1, padding=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
        nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=1, padding=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
        nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=1, padding=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(),
        nn.MaxPool2d(pool_size),
        nn.Flatten(),
        nn.Linear(hidden_size,hidden_size),
        nn.Linear(hidden_size,10),
        # nn.Softmax()
        # nn.ReLU(),
    )
    return regressor


class model(pl.LightningModule):
    def __init__(self):
        super(model,self).__init__()
        self.model = get_conv_net(1)
        # self.conv = nn.Conv2d(1, 128, 3, stride=1, padding=1)
        # self.fc1 = nn.Linear(28*28,256)
        # self.fc2 = nn.Linear(256,128)
        # self.out = nn.Linear(128,10)
        self.lr = 0.0001
        self.loss = nn.CrossEntropyLoss()
        self.register_buffer('one_hot', torch.eye(10, dtype=torch.long) )
    
    def forward(self,x):
        batch_size, _, _, _ = x.size()
        # print(x.size())
        # x = x.view(batch_size, 1, 28, 28)
        # print("Input ", x.size())
        # output = self.conv(x)
        # print("Output ", output.size())
        return self.model(x)

    
    def configure_optimizers(self):
        return Adam(self.parameters(),lr = self.lr)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        # y = self.one_hot[y]
        logits = self.forward(x)
        # print("Output ", y, logits)
        loss = self.loss(logits,y)
        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)

		# accuracy on training set
        accuracy_score(train_y, predictions)
        return loss
    
    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch
        logits = self.forward(x)
        loss = self.loss(logits,y)
        return loss

# Create Model Object
clf = model()
# Create Data Module Object
mnist = Data()
# Create Trainer Object
trainer = pl.Trainer(gpus=1, accelerator='dp', max_epochs=100)
trainer.fit(clf,mnist)



