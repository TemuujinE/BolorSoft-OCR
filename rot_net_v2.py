import torch
import torch.nn as nn

class RotNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, in_channels=1, has_dropout=True, dropout_rate=0.25):
        super(RotNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.has_dropout = has_dropout

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.max_pool = nn.MaxPool2d((2, 2))

        # Fully Connected Layers (will be initialized dynamically)
        self.fc_1 = None
        self.fc_2 = nn.Linear(128, self.num_classes)

        # Activation and Dropout Layers
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Flag to indicate initialization of fully connected layer
        self.fc_initialized = False

    def forward(self, x):
        # Convolutional and Pooling layers
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.max_pool(x)
        
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.max_pool(x)
        
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.max_pool(x)
        
        if self.has_dropout:
            x = self.dropout(x)

        # Dynamically compute the input size for the first fully connected layer
        if not self.fc_initialized:
            self._initialize_fc_layer(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.shape[0], -1)
        
        # Fully Connected layers
        x = self.fc_1(x)
        x = self.ReLU(x)
        
        if self.has_dropout:
            x = self.dropout(x)
        
        x = self.fc_2(x)
        x = self.softmax(x)
        return x

    def _initialize_fc_layer(self, x):
        # Calculate flattened size
        flattened_size = x.view(x.shape[0], -1).shape[1]
        # Initialize the first fully connected layer
        self.fc_1 = nn.Linear(flattened_size, 128)
        self.fc_initialized = True

    def reset_fc_layer(self):
        """Reset fully connected layer initialization status."""
        self.fc_initialized = False

# Example usage
model = RotNet(num_classes=2, in_channels=1)

# Example with dynamic input sizes
input_tensor_1 = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
output_1 = model(input_tensor_1)
print("Output shape for 28x28 input:", output_1.shape)

# Reset model for a new input size
model.reset_fc_layer()

input_tensor_2 = torch.randn(1, 1, 64, 64)  # Batch size of 1, 1 channel, 64x64 image
output_2 = model(input_tensor_2)
print("Output shape for 64x64 input:", output_2.shape)


import torch
import torch.nn as nn

class RotNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1, in_channels=1, has_dropout=True, dropout_rate=0.25):
        super(RotNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Residual Blocks
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.max_pool = nn.MaxPool2d((2, 2))
        
        # MLP
        self.fc_1 = nn.Linear(64, 128)  # Initialize with dummy value
        self.fc_2 = nn.Linear(128, self.num_classes)

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = torch.nn.Softmax(dim=1)
        
        # Flag to initialize MLP layers dynamically based on input size
        self.init_done = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        
        # Dynamically compute the input size for the first fully connected layer
        if not self.init_done:
            self._initialize_fc_layers(x)
        
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.fc_1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        
        x = self.fc_2(x)
        x = self.softmax(x)
        return x

    def _initialize_fc_layers(self, x):
        # Calculate flattened size
        flattened_size = x.view(x.shape[0], -1).shape[1]
        self.fc_1 = nn.Linear(flattened_size, 128)
        self.init_done = True

# Example usage
model = RotNet(num_classes=2, in_channels=1)
print(model)
