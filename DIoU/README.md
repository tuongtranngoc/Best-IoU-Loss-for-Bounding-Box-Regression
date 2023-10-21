# Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression (DIoU)

## Comparision between GIoU and DIoU

In the paper, the author provide three typical casese in simulation experiments:

First, the anchor box is set at diagonal orientation. GIoU loss tends to increase the size of predicted box to overlap with target box, while DIoU loss directly minizes normalized distance of center points

<p align='center'>
    <img src='images/comparision_diag.jpg'>
</p>

Second, the anchor box is set at the horizontal orientation. GIoU loss broadens the right edge of predicted box to overlap with target box, while the central point only move slightly towards target box. On th other hand, from the result at $T=400$, GIoU loss has totally degraded to IoU loss, while DIoU loss only at $T=120$

<p align='center'>
    <img src='images/comparision_horizontal.jpg'>
</p>

Thrid, the anchor box is set at the vertical orientation. Similarly, GIoU loss broadens the bottom edge of predicted box to overlap with target box and these two boxes do not match in the final iteration.

<p align='center'>
    <img src='images/comparision_vertical.jpg'>
</p>

To minimize the normalized distance between central points of two bounding boxes, and the penalty term can be defined as 

$$R_{DIoU}=\frac{\rho^2 (b, b^{gt})}{c^2}$$
where $b$ and $b^{gt}$ denote the central points of $B$ and $B^{gt}$, $\rho(.)$ is the Euclidean distance, and $c$ is the diagonal lenght of the smallest enclosing box covering the two boxes. And then the DIoU loss function can be defined as

$$L_{DIoU}=1 - IoU + \frac{\rho^2 (b, b^{gt})}{c^2}$$


## Experiments

|Backbone|mAP|mAP50|mAP75|
|--|--|--|--|
|ResNet34||||
|ResNet34+GIoU||||
|ResNet34+DIoU||||

## References
+ https://github.com/Zzh-tju/DIoU
+ https://arxiv.org/pdf/1911.08287.pdf