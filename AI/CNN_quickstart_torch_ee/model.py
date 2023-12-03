from torch import nn 

class ConvolutionNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # (64,32,28,28)
            nn.MaxPool2d(kernel_size=2), # (64,32,14,14)
            nn.Dropout(p=0.3),

            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),   # (64,64,14,14)
            nn.MaxPool2d(kernel_size=2),  # (64,64,7,7)
            nn.Dropout(p=0.3),

            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # (64,128,7,7)
            nn.MaxPool2d(kernel_size=2),   # (64,128,3,3)
            nn.Dropout(p=0.3),
            
        )

        self.fc = nn.Sequential(
            
            nn.Linear(1152,512, bias=True),
            nn.ReLU(),
            nn.Linear(512,10, bias=True)

            )   

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        logits = self.flatten(logits)
        logits = self.fc(logits)
        return logits