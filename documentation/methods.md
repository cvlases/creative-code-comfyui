# Methods Examples
Keeping everything the same, but changing the model to demonstrate visual differences.



## What is it? 

Model_name: vgg16
method: THIS IS WHAT CHANGES
composition: pure_white
colormap: magma
intensity: 0.75
contrast: 1.5
threshold: 0.25
class_index: -1
input_size: 224
uses_smooth_grad: true
snoothgrad_samples: 25


## What is it?


Vanilla Gradient
![Vanilla](method_demo/vanilla_grad.png)

Smooth Gradient
![Smooth](method_demo/smooth_grad.png)

Integrated Gradient
![Integrated](method_demo/integrated_grad.png)

Blur Integrated Gradients
![Blur](method_demo/blur_ig.png)

Guided Integrated Gradients
![Guided](method_demo/guided_ig.png)

XRAI
![XRAI](method_demo/XRAI.png)