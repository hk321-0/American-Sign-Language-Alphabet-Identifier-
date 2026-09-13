"""
Defining the following models: 
1. MLP - baseline model whose's architecture is flatten -> hidden -> logits.
2. CNN- exposes penultimate-layer embeddings via extract_features()
3. CNN + XGBoost: CNN is an input into XGBoost - peformed to improve model performance. 
                    The CNN is used to extract features from the images, which are then 
                    fed into an XGBoost classifier for final predictions.
"""
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, num_classes=24, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    #CNN's architecture: Two convolutional blocks + one FC hidden layer + output layer.
    #Embedding is the output of the FC hidden layer, which is inputted to XGBoost to create the 3rd model CNN + XGBoost.
    def __init__(self, num_classes=24, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   #28x28 --> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                                #14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   #14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                                #7x7
        )
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, embed_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.embed(x)
        return self.classifier(x)

    @torch.no_grad()
    def extract_features(self, x):
        #Returns the embedding layer so that it can be inputted to XGBoost.
        x = self.conv(x)
        x = self.embed(x)
        return x
