# Model Examples
Keeping everything the same, but changing the model to demonstrate visual differences.

## Understanding Models
Models
The model is the "brain" being interrogated — it's a pretrained ImageNet classifier whose internal decision-making we're visualizing. Different models have different "personalities" in how they look at images.

need to explain why these models, where they came from. explain reasoning behind choices

VGG16 / VGG19 — Old, deep, sequential architecture from 2014. For portraits, these are the most interesting because they're blunt and spatially crude. The attention patterns feel almost aggressive — big, blocky regions. Good for the "surveillance" feeling you're going for. VGG16 is the default for a reason.
ResNet50 / ResNet101 — More modern, uses skip connections. Produces cleaner, more distributed attention. Less visually dramatic than VGG but more "honest" about what it's actually classifying.
<!-- EfficientNet-B0 — Very modern, highly optimized. Produces fine-grained, sparse attention maps. Interesting for showing how efficiently a model can locate a subject with minimal "gaze." -->
DenseNet121 — Dense connectivity means attention is highly distributed across the image. Produces diffuse, almost painterly saliency maps.
MobileNetV3 — Designed for mobile devices, so it's aggressively compressed. Produces the sparsest, most stripped-down attention — shows only the absolute minimum the model needs to make a decision.
InceptionV3 — Uses parallel convolutions at multiple scales. Produces multi-scale attention that can look almost fractal. Requires input_size 299 instead of 224.

## What is it? 

Model_name: THIS IS WHAT CHANGES
method: guided IG
composition: pure_white
colormap: magma
intensity: 0.75
contrast: 1.5
threshold: 0.25
class_index: -1
input_size: 224
uses_smooth_grad: true
snoothgrad_samples: 25


## listed out


cool
![cool](method_demo/cool.png)
