import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as v2
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import HMDB51

#return names
def get_approach_names():
    return ["verybasiccnn", "advancedcnn"]

#return description based on names
def get_approach_description(approach_name):
    descriptions = {
        "verybasiccnn": "a basic neural network with three convolutional layers and relu activations.",
        "advancedcnn": "a hopefully more advanced cnn with three convolutional layers, batch normalization, and a simplified classifier."
    }
    return descriptions.get(approach_name)
#data transform
def get_data_transform(approach_name, training):

    if approach_name == "verybasiccnn":
        if training:
            return v2.Compose([
                v2.Resize((128, 128)),
                v2.RandomHorizontalFlip(),
                v2.ConvertImageDtype(torch.float32)
            ])
        else:
            return v2.Compose([
                v2.Resize((128, 128)),
                v2.ConvertImageDtype(torch.float32)
            ])
    elif approach_name == "advancedcnn":
        if training:
            return v2.Compose([
                v2.Resize((224, 224)),
                v2.RandomRotation(15),
                v2.RandomHorizontalFlip(),
                v2.ConvertImageDtype(torch.float32)
            ])
        else:
            return v2.Compose([
                v2.Resize((224, 224)),
                v2.ConvertImageDtype(torch.float32)
            ])
#toTensor breaks everything dont use


#given name, get batch size
def get_batch_size(approach_name):
    batch_sizes = {
        "verybasiccnn": 32,
        "advancedcnn": 8  
    }
    return batch_sizes.get(approach_name) 

#given approacch and output class build and return network
def create_model(approach_name, class_cnt):
    if approach_name == "verybasiccnn":
        model = veryBasicCNN(class_cnt)
    elif approach_name == "advancedcnn":
        model = AdvancedCNN(class_cnt)
    return model


#given the model, train it
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    """
    given the provided model, the device it is located, and the relevant dataloaders, train this model and return it.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 15  #15 seems to work decently?

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):
            if isinstance(data, (list, tuple)):
                if len(data) == 2:
                    inputs, labels = data
                elif len(data) == 3:
                    inputs, _, labels = data 

            if inputs.dim() == 5 and inputs.shape[1] != 3:
                inputs = inputs.permute(0, 2, 1, 3, 4)
            elif inputs.dim() == 5 and inputs.shape[1] == 3:
                pass


        
            if labels.dim() == 3 and labels.shape[1] == 1 and labels.shape[2] == 0:
                labels = labels.squeeze().long()
            elif labels.dim() == 1:
                labels = labels.long()

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"approach: {approach_name} | epoch [{epoch+1}/{epochs}]")

        #evaluate on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                if isinstance(data, (list, tuple)) and len(data) >= 2:
                    if len(data) == 2:
                        inputs, labels = data
                    elif len(data) == 3:
                        inputs, _, labels = data  

                if inputs.dim() == 5 and inputs.shape[1] != 3:
                    inputs = inputs.permute(0, 2, 1, 3, 4)
                elif inputs.dim() == 5 and inputs.shape[1] == 3:
                    pass

                if labels.dim() == 3 and labels.shape[1] == 1 and labels.shape[2] == 0:
                    labels = labels.squeeze().long()
                elif labels.dim() == 1:
                    labels = labels.long()

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"approach: {approach_name} | epoch [{epoch+1}/{epochs}]")

    return model

#define verybasiccnn 
class veryBasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(veryBasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),  
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)) 
        )
        self.classifier = nn.Linear(128, num_classes) 

    def forward(self, x):
        if x.dim() == 5 and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        elif x.dim() == 5 and x.shape[1] == 3:
            pass
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#define advancedcnn
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),  
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),  
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),  
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),  
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))  
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes) 
        )
      

    def forward(self, x):
        if x.dim() == 5 and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        elif x.dim() == 5 and x.shape[1] == 3:
            pass
        
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)
        return x
