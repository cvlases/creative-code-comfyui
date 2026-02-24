# Colormap Examples
Keeping everything the same, but changing the color map to demonstrate visual differences.


Other Parameters
intensity — How strongly the saliency visualization is blended onto the original. 0 = invisible, 1 = full. Most compositions look best between 0.6–0.85.
contrast (gamma) — Applies a power function to the saliency values before visualization. Values above 1.0 compress the low end and expand the high end — making the most salient regions pop dramatically while less important regions fade further. Cranking this to 2.5–3.0 produces extreme, high-drama results where only the very top attention regions survive.
threshold — Used primarily by isolation and cutout modes. Sets the minimum saliency value to be considered "attended to." Higher values make the visualization sparser and more selective — only the regions the model is most confident about survive.
class_index — ImageNet has 1000 classes. By default (-1) the node auto-detects what the model predicted and visualizes attention for that class. You can override this to force the visualization to show what the model would attend to if it were trying to classify the image as a specific thing — for example, forcing class 207 (golden retriever) on a human portrait produces attention patterns for a question the model was never meant to answer, which has interesting artistic implications.
input_size — All these models were trained on fixed-size inputs. 224×224 is standard for most; InceptionV3 requires 299×299. The image is resized to this before processing, which is why the output appears at that resolution.
use_smoothgrad — When enabled on top of any method (except SmoothGrad which already does this), applies the SmoothGrad averaging process to smooth the result. Generally produces more aesthetically pleasing outputs at the cost of being slower.
smoothgrad_samples — How many noisy samples to average when SmoothGrad is active. 25 is a good balance; 50+ produces very smooth results but takes proportionally longer.


## What is it? 

Model_name: vgg16
method: guided IG
composition: inverse
colormap: magma
[the following IS WHAT CHANGES]
default values:

intensity: 0.75
contrast: 1.5
threshold: 0.25
class_index: -1
input_size: 224
uses_smooth_grad: true
snoothgrad_samples: 25


## listed out
should explain why i picked the default values i did. 

### Intensity 

intensity = 0.5
(lower)
![low_intensity](method_demo/intensity_0.5.png)

intensity = 1.0
(higher)
![high_intensity](method_demo/intensity_1.0.png)

### Contrast

contrast = 1.0
(lower)
![low_contrast](method_demo/contrast_1.0.png)

contrast = 2.0
(higher)
![high_contrast](method_demo/contrast_2.0.png)

### threshold
threshold = 0.0
(lower)
![low_threshold](method_demo/threshold_0.png)

threshold = 0.5
(higher)
![high_threshold](method_demo/threshold_0.5.png)

 ### class index
class index = 1
(a little change)
![index-1](method_demo/class_index_1.png)

class index = 10
(a larger change)
![index-10](method_demo/class_index_10.png)

 ### input size
 Didn't adjust this because most models require this size (224) to run properly. 

 ### uses smooth grad
Without (uses smooth grad = false)
![false](method_demo/smooth_grad_false.png.png)



 ### smooth grad samples
samples = 50
 ![samples_50](method_demo/samples_50.png)