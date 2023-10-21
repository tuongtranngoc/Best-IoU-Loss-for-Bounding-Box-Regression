# Complete IoU Loss (CIoU)
DIoU aswerd the question: Is it feasible to directly minimize the normalized distance between predicted box and target box for achieving faster convergence?, CIoU will answer the question: How to make the regression more accurate and faster when having overlap even inclusion with target box?

CIoU is a good loss for bounding box regression consider three important geometric factors (overlap area, central point distance and aspect ratio).

There fore, based on DIoU loss, the CIoU loss is proposed by imposing the consistency of aspect ratio.

$$R_{CIoU} = \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

where $\alpha$ is a positive trade-off parameter, and $v$ measures the consistency of aspect ratio.

$$v=\frac{4}{\pi^2}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac{w}{h})^2$$

$$\alpha = \frac{v}{(1-IoU)+v}$$

The loss function can be defined as

$$L_{CIoU}=1-IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

## Experiments

|Backbone|mAP|mAP50|mAP75|
|--|--|--|--|
|ResNet34||||
|ResNet34+GIoU||||
|ResNet34+DIoU||||
|ResNet34+CIoU||||

## References

+ https://arxiv.org/pdf/1911.08287.pdf