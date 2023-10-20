# Generalized Intersect over Union (GIoU)

Object Detection consists of two sub-tasks: Object localization and object classification. The common goal of object localization is to determize coordinates of bounding boxes where are in the picture.

Intersection over Union (IoU) is the metric measure to evaluate the accuracy of object detection model.

$$IoU = \frac{\mid A \cap B\mid}{\mid A \cup B\mid} = \frac{\mid I \mid}{\mid U \mid}$$

The IoU value is computed based on ground-truth box and predicted box. During training model, the cost function is usually optimized to return the best of predicted box and compute IoU to evaluate model. The common cost functions are l1-norm distance or l2-norm distance.

The below example show that if we calculate l1-norm distance and l2-norm distance for the bounding boxes in cases, the $l_n$-norm values are exactly the same, but their IoU and GIoU values are very difference. Therefore, there is no strong relation between loss optimization and improving IoU values.

<p align='center'>
    <img src='images/relation.png'>
</p>

Generalized version of IoU can be directly used as the objective function to optimize in the object detection problem. When IoU does not reflect if two shapes are in vicinity of each other or very far from each other.

$$GIoU = IoU - \frac{\mid C\setminus {(A\cup B)}\mid}{\mid C \mid}$$

Cost function for bounding box regression:

$$L_{GIoU} = 1-GIoU$$

## Experiments
I applied GIoU in loss function of YOLOv1 algorithm on PASCAL VOC 2007 testing dataset.

| Backbone | mAP| mAP50 | mAP75 | 
|---|---|---|---|
| ResNet34 |  | |
| ResNet34 + GIoU| 41.2| 64.5 | 44.0 | 

## References
> + https://giou.stanford.edu/
> + https://giou.stanford.edu/GIoU.pdf
