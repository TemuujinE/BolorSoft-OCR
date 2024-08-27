import torch
import torch.nn as nn

class RotNet(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, has_dropout=True, dropout_rate=0.25):
        super(RotNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.has_dropout = has_dropout

        # Convolutional Layers with Batch Normalization
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Fully Connected Layers (will be initialized dynamically)
        self.fc_1 = None  # Initialize this dynamically based on input size
        self.fc_2 = nn.Linear(128, self.num_classes)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout_rate)

        # Flag to indicate initialization of fully connected layer
        self.fc_initialized = False

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Convolutional and Pooling layers
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.has_dropout:
            x = self.dropout(x)

        # Dynamically compute the input size for the first fully connected layer
        if not self.fc_initialized:
            self._initialize_fc_layer(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(x.shape[0], -1)
        
        # Fully Connected layers
        x = self.fc_1(x)
        x = nn.ReLU(inplace=True)(x)
        
        if self.has_dropout:
            x = self.dropout(x)
        
        x = self.fc_2(x)
        # No softmax here; use CrossEntropyLoss which applies LogSoftmax internally
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

# Example usage
model = RotNet(num_classes=4, in_channels=1)

# Example with dynamic input sizes
input_tensor_1 = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
output_1 = model(input_tensor_1)
print("Output shape for 28x28 input:", output_1.shape)

# Reset model for a new input size
model.reset_fc_layer()

input_tensor_2 = torch.randn(1, 1, 64, 64)  # Batch size of 1, 1 channel, 64x64 image
output_2 = model(input_tensor_2)
print("Output shape for 64x64 input:", output_2.shape)
