## pre-information
#### all the Geography are in ['France', 'Spain', 'Germany']， already changed to 0, 1, 2
#### all the Gender are in ['Male', 'Female'], already changed to 0, 1

#### after the dataload, the train data shape is torch.Size([1, 1, 10, 1]), the label data shape is torch.Size([1, 1, 1, 1])

# model
## ResNet
I modify a little, structure below:
 ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
#### ================================================================
            Conv2d-1               [1, 1, 5, 1]               6
       BatchNorm2d-2               [1, 1, 5, 1]               2
              ReLU-3               [1, 1, 5, 1]               0
            Conv2d-4               [1, 1, 3, 1]               3
       BatchNorm2d-5               [1, 1, 3, 1]               2
              ReLU-6               [1, 1, 3, 1]               0
            Conv2d-7               [1, 1, 1, 1]               3
       BatchNorm2d-8               [1, 1, 1, 1]               2
              ReLU-9               [1, 1, 1, 1]               0
           Conv2d-10               [1, 1, 1, 1]               1
      BatchNorm2d-11               [1, 1, 1, 1]               2
             ReLU-12               [1, 1, 1, 1]               0
#### ================================================================
#### Total params: 21
#### Trainable params: 21
#### Non-trainable params: 0
#### ----------------------------------------------------------------
#### Input size (MB): 0.00
#### Forward/backward pass size (MB): 0.00
#### Params size (MB): 0.00
#### Estimated Total Size (MB): 0.00
