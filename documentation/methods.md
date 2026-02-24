# Methods Examples
Keeping everything the same, but changing the model to demonstrate visual differences.

## Understanding methods
These are the mathematical techniques used to ask "what pixels mattered to this model's decision?" Each has a different visual character.
Vanilla Gradients — The simplest approach: take the derivative of the output with respect to each input pixel. Fast and raw. The result is noisy and looks almost like static or grain — shows every pixel that had any influence at all, including spurious ones. Visually the most aggressive and chaotic.
SmoothGrad — Takes Vanilla Gradients but runs it many times with small random noise added to the image each time, then averages the results. This cancels out the noise and reveals more stable patterns. Visually smoother and more "reliable" looking than vanilla, but still texture-rich. The smoothgrad_samples parameter controls how many noisy runs are averaged — more = smoother but slower.
Integrated Gradients — Instead of just looking at gradients at the input image, this traces a path from a blank black image to your actual image and accumulates gradients along that path. This gives it a strong theoretical foundation (it satisfies mathematical axioms about attribution). Produces clean, smooth maps that highlight meaningful regions rather than just any active pixels. The ig_steps parameter controls how many steps along that path — more = more accurate but slower. This is usually the best starting method.
Blur IG — A variant of Integrated Gradients where instead of interpolating from black to the image, it interpolates from a heavily blurred version to the sharp image. This tends to produce maps focused on edges and structural details rather than color/texture. Produces a different aesthetic — more architectural-looking.
Guided IG — Another IG variant that uses gradient information to choose a smarter integration path, concentrating steps where the gradients are largest. Produces sharper, higher-contrast attribution maps with more defined edges. Often the most visually striking for portraits.
XRAI — Completely different approach. Instead of pixel-level gradients, it segments the image into regions and attributes importance to each region as a whole. Produces a patchwork/segmented look — almost like stained glass or a heat map made of irregular shapes. Very different aesthetic from the gradient methods. Uses algorithm="fast" by default; changing to "full" in the code gives higher quality but is much slower.


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