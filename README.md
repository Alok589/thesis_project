**Introduction**:
This network has been implimentd by following the basic idea of UNet.
A Residual block from ResNet and Squeeze and Excitation block from SENet has been used.
Residual block allows training deeper networks and prevent from vanishing gradient problem
Whereas, in Squeeze block global information of the feature map is extracted through its spatial dimension
then provided as input to the Excitation block. This block extract the channel wise information and corelation
between them. Therefore emphasizing on most relevant features and suppress
less important one.
A hybrid deep convolution neural network (RSE-UNet) is shown below.
![](testset%evaluation/RES-UNet.png)


# Camera system
![](testset%evaluation/camera_overview.png)
