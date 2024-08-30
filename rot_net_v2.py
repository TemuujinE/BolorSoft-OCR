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


import numpy as np

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.

    Args:
    - box1, box2: Lists or arrays of format [x_min, y_min, x_max, y_max]

    Returns:
    - IoU value (float)
    """
    # Determine the coordinates of the intersection rectangle
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)

    # Compute the area of both the prediction and ground truth rectangles
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def are_boxes_horizontally_aligned(boxes):
    """
    Check if multiple bounding boxes are horizontally aligned (on the same line).

    Args:
    - boxes: List of bounding boxes [x_min, y_min, x_max, y_max]

    Returns:
    - True if boxes are horizontally aligned, False otherwise
    """
    y_mins = [box[1] for box in boxes]
    y_maxs = [box[3] for box in boxes]

    # Check if the vertical overlap exists by comparing the min and max y-values
    min_y_min = min(y_mins)
    max_y_max = max(y_maxs)
    
    # All boxes should have overlapping y range if horizontally aligned
    for box in boxes:
        if box[1] > max_y_max or box[3] < min_y_min:
            return False
    
    return True

def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Evaluate predicted bounding boxes against ground truth boxes.

    Args:
    - pred_boxes: List of predicted bounding boxes [x_min, y_min, x_max, y_max]
    - gt_boxes: List of ground truth bounding boxes [x_min, y_min, x_max, y_max]
    - iou_threshold: IoU threshold to consider a prediction correct

    Returns:
    - results: Dictionary with 'correct' and 'incorrect' predictions
    """
    results = {'correct': 0, 'incorrect': 0}
    
    for pred_box in pred_boxes:
        overlapping_gt_boxes = []
        for gt_box in gt_boxes:
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                overlapping_gt_boxes.append(gt_box)
        
        # Check alignment of overlapping ground truth boxes
        if len(overlapping_gt_boxes) > 0:
            if are_boxes_horizontally_aligned(overlapping_gt_boxes):
                results['correct'] += 1
            else:
                results['incorrect'] += 1
        else:
            results['incorrect'] += 1
    
    return results

# Example usage
predicted_boxes = [
    [100, 100, 200, 150],
    [250, 200, 350, 250],
    [150, 300, 300, 350]
]

ground_truth_boxes = [
    [110, 110, 190, 140],  # Horizontally aligned with the first predicted box
    [160, 120, 180, 145],  # Horizontally aligned with the first predicted box
    [255, 205, 345, 245],  # Correctly predicted
    [160, 310, 290, 340]   # Correctly predicted
]

results = evaluate_predictions(predicted_boxes, ground_truth_boxes)
print("Correct Predictions:", results['correct'])
print("Incorrect Predictions:", results['incorrect'])

