# Algorithmic Attention — Saliency Visualization Tool Documentation

This tool interrogates pretrained image classifiers using gradient-based saliency methods from Google's [PAIR saliency library](https://github.com/PAIR-code/saliency), exposing the hidden attentional logic of neural networks as visual artifacts. Built as a custom ComfyUI node set.

The core question it answers: *when a model classifies an image, which pixels drove that decision?*

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Models](#models)
3. [Methods](#methods)
4. [Composition Modes](#composition-modes)
5. [Colormaps](#colormaps)
6. [Other Parameters](#other-parameters)

---

## How It Works

All eight supported models are **convolutional neural networks (CNNs)** trained on ImageNet — a dataset of 1.2 million images across 1,000 categories. The training process adjusts millions of numerical weights until the network can reliably distinguish between categories like "Labrador retriever," "espresso maker," and "kimono."

What makes them useful for saliency visualization is that they're all **differentiable** — built from mathematical operations (matrix multiplications, convolutions, activation functions), so you can calculate the derivative of the classification output with respect to any input pixel. That derivative is the gradient: *if I changed this pixel slightly, how much would the model's confidence change?* High gradient = the model is paying attention to that pixel.

The models differ in architecture, which changes not just their accuracy but the *character* of their attention — where they look, how diffuse or concentrated it is, how coarse or fine-grained.

---

## Models

The model is the "brain" being interrogated. Different architectures have different visual personalities.

> **Demo settings:** method: Guided IG · composition: pure_white · colormap: magma · intensity: 0.75 · contrast: 1.5 · threshold: 0.25 · class_index: -1 · input_size: 224 · use_smoothgrad: true · smoothgrad_samples: 25

**TLDR:** Coarse and blunt (VGG) → distributed and semantic (ResNet) → diffuse and layered (DenseNet) → multi-scale (Inception) → sparse and shortcut-prone (MobileNet/EfficientNet)

---

### VGG16 and VGG19

Among the oldest and simplest networks still in common use, developed at Oxford's Visual Geometry Group in 2014. The design is deliberately minimal: stack many small 3×3 convolutional filters in sequence, going deeper and deeper, with max-pooling layers periodically halving the spatial resolution. VGG16 has 16 layers with learned weights; VGG19 has 19. Both end with three large fully-connected layers that make the final classification decision.

Each convolutional layer detects increasingly abstract features — early layers find edges and color gradients, middle layers find textures and simple shapes, later layers find object parts. The fully-connected layers at the end combine everything into a single verdict. Because the architecture is purely sequential with no shortcuts, everything must compress through a bottleneck.

VGG models have the **largest, most spatially coarse attention patterns** of any model here. Because the architecture is old and inefficient, it has to "look hard" at large regions rather than pinpointing specific features. The saliency maps feel blunt, aggressive, and somewhat unpredictable — they often grab large swaths of the face, background, or contextual objects. This reads more like surveillance than recognition. VGG16 and VGG19 produce slightly different patterns because the extra three layers in VGG19 allow slightly more abstraction, but both share this coarseness.

*Parameter count: VGG16 — 138M · VGG19 — 144M. Famously inefficient for their accuracy.*

**VGG16**
![vgg16](model_demo/vgg16.png)

**VGG19**
![vgg19](model_demo/vgg19.png)

---

### ResNet50 and ResNet101

Developed at Microsoft Research in 2015. ResNets introduced the **residual connection** (skip connection): instead of forcing every layer to learn a full transformation, each block of layers learns only the *residual* — the difference between its input and desired output. The input is added back to the output before passing to the next block. This allows networks to be made much deeper without gradients vanishing during training.

The residual connections create a scaffold — information can skip over groups of layers and travel directly toward the output, meaning the network doesn't have to compress everything through every layer. ResNet50 has 50 layers organized into four stages of residual blocks; ResNet101 doubles the middle stages.

ResNets produce **more distributed, semantically coherent attention** than VGG. Because residual connections allow gradients to flow more cleanly, the saliency maps tend to reflect what the model actually uses to make its decision more faithfully. Attention looks more "considered" — multiple relevant regions rather than one dominant blob. The difference between ResNet50 and ResNet101 is subtle: the deeper model tends toward slightly more refined, higher-level attention.

*Parameter count: ResNet50 — 25M · ResNet101 — 44M. Far more efficient than VGG.*

**ResNet50**
![resnet50](model_demo/resnet50.png)

**ResNet101**
![resnet101](model_demo/resnet101.png)

---

### DenseNet121

Developed at Cornell in 2016. DenseNet takes the residual connection idea from ResNet and pushes it to an extreme: while ResNet connects each block only to the next, in DenseNet **every layer connects directly to every subsequent layer** in its dense block. The 121 refers to the total number of layers.

Because every layer receives feature maps from all previous layers, the network never discards information. Early features — edges, simple textures, color gradients — remain directly accessible all the way through the network. Each layer has access to a "collective memory" of everything seen at every previous level of abstraction. This is the opposite of VGG's progressive compression.

DenseNet produces the **most distributed, multi-scale attention patterns**. Because every layer draws on both low-level and high-level features simultaneously, saliency maps highlight many overlapping regions at different scales — fine texture details and large structural regions at the same time. The results are diffuse and layered, almost painterly.

*Parameter count: 8M.*

**DenseNet121**
![densenet121](model_demo/densenet121.png)

---

### InceptionV3

Developed at Google in 2015. Inception introduced **parallel convolutions at multiple scales within a single layer**. Each "inception module" simultaneously applies 1×1, 3×3, and 5×5 convolutions plus a max-pooling operation, then concatenates all the outputs. The network learns which scale is useful for which features rather than committing to one.

Because features are detected at multiple scales simultaneously throughout the network, Inception can respond to both fine-grained texture (small filters) and large structural shapes (large filters) at every level. The 1×1 convolutions act as dimensionality reduction to keep computation manageable. V3 adds factorized convolutions (replacing 5×5 with two sequential 3×3 filters for efficiency) and batch normalization throughout.

Inception produces **multi-scale attention that can look almost fractal** — fine and coarse detail highlighted simultaneously, with a layered quality: broad regions of moderate attention containing sharp peaks at specific features within them.

> **Note:** InceptionV3 requires `input_size: 299` instead of 224, because it was trained at that resolution. This produces slightly more spatial detail in the output.

*Parameter count: 23.8M.*

**InceptionV3**
![inception_v3](model_demo/inception_v3.png)

---

### MobileNetV3 Large

Developed at Google in 2017–2019, explicitly designed to run on smartphones and embedded devices with limited compute. It uses **depthwise separable convolutions** — splitting a standard convolution into two steps (a depthwise step that filters each channel separately, then a pointwise step that combines channels) — reducing computational cost by roughly 8–9×. V3 added neural architecture search (NAS) and squeeze-and-excitation blocks that explicitly learn which feature channels matter most.

The aggressive computational compression means MobileNet achieves classification using far fewer operations than any other model here. The network is forced to extract the most discriminative features with minimal computation, pruning everything that doesn't directly help.

MobileNet produces the **sparsest, most minimal attention maps**. It often fixates on a single small region — sometimes not the face itself, but a piece of clothing, a background element, or a texture patch — because that's the cheapest path to a confident classification. This makes it conceptually interesting as commentary on algorithmic shortcuts: it's not trying to "understand" the image, just find the easiest distinguishing feature. The attention can feel almost arbitrary, which is itself a statement.

*Parameter count: 5.5M.*

**MobileNetV3**
![mobilenet_v3_large](model_demo/mobilenet_v3_large.png)

---

### EfficientNet-B0

Developed at Google Brain in 2019. EfficientNet rethought how to scale neural networks — instead of independently making networks wider, deeper, or taking higher-resolution inputs, it scales all three dimensions together in a mathematically derived fixed ratio. B0 is the smallest model in the family (scaling up to B7).

EfficientNet uses MBConv blocks (mobile inverted bottleneck convolution, borrowed from MobileNet): each block expands channel count, applies depthwise convolution, then compresses back down. It also includes squeeze-and-excitation blocks that rescale feature channels by learned importance weights — a form of built-in attention to its own internal representations.

EfficientNet produces the most **surgically precise saliency maps**. Designed to maximize classification information per computation, its attention patterns are extremely sparse — it finds the minimum signal needed to make a decision and ignores everything else. For portraits, this often means isolating very specific facial features (an eye, a jawline segment) while essentially ignoring the rest. This is conceptually striking: it shows how little the machine needs to see to classify confidently.

*Parameter count: 5.3M — an order of magnitude smaller than VGG for comparable accuracy.*

---

### What's Absent and Why

Transformer-based vision models (ViT, CLIP) were excluded because they use explicit attention heads rather than convolutions, making gradient-based saliency methods less directly applicable — their attention is already readable through other means. This project is specifically about making the *implicit* computational gaze of convolutional networks visible, which is a different conceptual claim than reading explicit attention weights from a transformer.

---

## Methods

These are the mathematical techniques used to ask "what pixels mattered to this model's decision?" Each produces a fundamentally different visual character.

> **Demo settings:** model: vgg16 · composition: pure_white · colormap: magma · intensity: 0.75 · contrast: 1.5 · threshold: 0.25 · class_index: -1 · input_size: 224 · use_smoothgrad: true · smoothgrad_samples: 25

---

### Vanilla Gradients

The simplest approach: take the partial derivative of the classification output with respect to each input pixel. Fast and raw. The result is noisy — almost like static or grain — because it shows every pixel that had *any* influence at all, including spurious ones. Produces the most aggressive, chaotic, high-frequency visual results.

![Vanilla Gradients](method_demo/vanilla_grad.png)

---

### SmoothGrad

Takes Vanilla Gradients but runs it many times with small random noise added to the image each time, then averages the results. The noise cancels out spurious gradients while stable signals reinforce, revealing more structurally reliable patterns. Visually smoother and more coherent than vanilla, but still texture-rich.

The `smoothgrad_samples` parameter controls how many noisy runs are averaged — more = smoother but proportionally slower.

![SmoothGrad](method_demo/smooth_grad.png)

---

### Integrated Gradients

Instead of computing gradients at only the input image, this traces a path from a blank black image to the actual image and accumulates gradients at every step along that path. This gives it strong theoretical grounding — it satisfies formal mathematical axioms about attribution (completeness, sensitivity, linearity). Produces clean, smooth maps that highlight semantically meaningful regions rather than just any active pixels.

The `ig_steps` parameter controls how many interpolation steps are taken — more steps = more accurate attribution but slower computation. This is usually the best starting method for portrait work.

![Integrated Gradients](method_demo/integrated_grad.png)

---

### Blur IG

A variant of Integrated Gradients where instead of interpolating from a black image to the input, it interpolates from a *heavily blurred version* of the image to the sharp version. The difference in what's being integrated changes what gets attributed — this tends to produce maps concentrated on edges and structural transitions rather than color or texture. The results look more architectural — sharper outlines, less fill.

![Blur IG](method_demo/blur_ig.png)

---

### Guided IG

Another IG variant that uses gradient information to choose a smarter integration path, concentrating steps where the gradients are largest rather than taking uniform steps. This produces sharper, higher-contrast attribution maps with more defined edges — it's spending its computational budget on the most informative parts of the integration. Often the most visually striking method for portraits.

![Guided IG](method_demo/guided_ig.png)

---

### XRAI

A completely different approach. Instead of pixel-level gradient computation, XRAI segments the image into regions using an oversegmentation algorithm, then attributes importance to each region as a whole by measuring how much removing it changes the model's output. Produces a patchwork, segmented appearance — almost like stained glass or an irregular heat map. Very distinct aesthetic from gradient methods.

> **Note:** XRAI returns a 2D region map rather than a 3D gradient volume, so it is handled differently in the rendering pipeline. It is also slower than the gradient-based methods.

![XRAI](method_demo/XRAI.png)

---

## Composition Modes

These control how the saliency map is composited back onto the original image. This is where the artistic and conceptual intent lives.

> **Demo settings:** model: vgg16 · method: Guided IG · colormap: magma · intensity: 0.75 · contrast: 1.5 · threshold: 0.25 · class_index: -1 · input_size: 224 · use_smoothgrad: true · smoothgrad_samples: 25

---

### isolation

The default and most conceptually pointed mode. Salient regions appear in full color using the chosen colormap; everything else converts to greyscale. The machine's gaze burns in color while what it ignores fades to grey — a direct visual statement about algorithmic attention and selective visibility.

![isolation](composition_demo/isolation.png)

---

### overlay

Classic heatmap blend. The colormap is alpha-composited over the original at the `intensity` level. Familiar and readable, but less subversive — the most conventional visualization approach.

![overlay](composition_demo/overlay.png)

---

### spotlight

Salient regions are bright; everything else darkens dramatically. Feels like a searchlight or surveillance camera — forensic, clinical, authoritarian.

![spotlight](composition_demo/spotlight.png)

---

### ghost

The original image fades to 25% opacity and the saliency map glows over it. The subject becomes spectral, haunted by its own classification. Good for uncanny or dissociative effects.

![ghost](composition_demo/ghost.png)

---

### invert

Shows what the algorithm *ignores* instead of what it sees. The saliency map is inverted before compositing. Conceptually the most subversive mode — it visualizes the blind spots, the parts of a face that are invisible to the machine.

![invert](composition_demo/invert.png)

---

### cutout

Hard binary mask. Pixels above the `threshold` value show the original image in full; everything below shows a dim version of the heatmap. Creates a stark graphic effect — the subject literally cut from the image by the algorithm's attention boundary.

![cutout](composition_demo/cutout.png)

---

### multiply

Photoshop-style multiply blend. Attention modulates brightness multiplicatively, dimming non-salient areas. Dark and moody — ignored regions sink into shadow.

![multiply](composition_demo/multiply.png)

---

### screen

Photoshop-style screen blend. Attention brightens salient regions. Lighter and more luminous than multiply — glowing rather than shadowed.

![screen](composition_demo/screen.png)

---

### triptych

Outputs three panels side by side: original image | saliency map alone | composite. Shows the full chain of the visualization in one frame. Designed for gallery or documentation contexts.

![triptych](composition_demo/triplych.png)

---

### mask_only

Just the colored saliency map with no original image. Pure algorithmic output — no human context, no original reference. The map exists alone.

![mask_only](composition_demo/mask_only.png)

---

### pure_white

Black background, saliency values mapped directly to white intensity. No color, no original — just the raw shape of what the algorithm attended to, like an x-ray or photogram. The contrast and threshold parameters still apply, making this mode range from soft and diffuse (low contrast, low threshold) to sparse and clinical (high contrast, high threshold).

![pure_white](composition_demo/pure_white.png)

---

## Colormaps

These apply a color gradient to the greyscale saliency values. Low values (less attended) map to one end of the gradient; high values (more attended) map to the other.

> **Demo settings:** model: vgg16 · method: Guided IG · composition: invert · intensity: 0.75 · contrast: 1.5 · threshold: 0.25 · class_index: -1 · input_size: 224 · use_smoothgrad: true · smoothgrad_samples: 25

---

| Colormap | Range | Character |
|----------|-------|-----------|
| **inferno** | Black → purple → orange → yellow | High contrast, dramatic. The default — feels urgent, slightly threatening |
| **plasma** | Purple → pink → yellow | Vibrant and synthetic-looking, digital, almost neon |
| **magma** | Black → deep purple → pink-white | Darker and moodier than inferno, more melancholic |
| **viridis** | Purple → teal → yellow-green | Standard scientific colormap, perceptually uniform, readable but less dramatic |
| **hot** | Black → red → orange → white | Classic heat vision / thermal camera aesthetic |
| **cool** | Cyan → magenta | Cold, clinical, almost medical |
| **spring** | Magenta → yellow | Neon, synthetic, maximally artificial |
| **copper** | Black → brown → copper-gold | Metallic, archival, almost daguerreotype-like |
| **twilight** | Purple → white → orange → purple (cyclical) | Smooth and atmospheric |
| **ocean** | Black-blue → cyan → white | Deep and watery |

---

**cool**
![cool](colormap_demo/cool.png)

**hot**
![hot](colormap_demo/hot.png)

**copper**
![copper](colormap_demo/copper.png)

**inferno**
![inferno](colormap_demo/inferno.png)

**magma**
![magma](colormap_demo/magma.png)

**ocean**
![ocean](colormap_demo/ocean.png)

**plasma**
![plasma](colormap_demo/plasma.png)

**spring**
![spring](colormap_demo/spring.png)

**twilight**
![twilight](colormap_demo/twilight.png)

**viridis**
![viridis](colormap_demo/viridis.png)

---

## Other Parameters

> **Demo settings:** model: vgg16 · method: Guided IG · composition: invert · colormap: magma · class_index: -1 · input_size: 224

### Intensity

Controls how strongly the saliency visualization blends onto the original. 0 = invisible, 1 = full replacement. Most compositions look best between 0.6–0.85.

Default: **0.75** — chosen as the midpoint that preserves enough of the original image for readability while letting the saliency map dominate visually.

**intensity = 0.5** (lower)
![low_intensity](extras_demo/intensity_0.5.png)

**intensity = 1.0** (higher / full)
![high_intensity](extras_demo/intensity_1.0.png)

---

### Contrast (Gamma)

Applies a power function to the saliency values before visualization: `value^(1/contrast)`. Values above 1.0 compress the low end and expand the high end — the most salient regions pop while less important regions fade further. Cranking to 2.5–3.0 produces extreme, high-drama results where only the very top attention regions survive. Values below 1.0 flatten differences, making attention look more uniformly distributed.

Default: **1.5** — chosen to add drama without becoming so extreme that only a few pixels survive.

**contrast = 1.0** (flat, no adjustment)
![low_contrast](extras_demo/contrast_1.0.png)

**contrast = 2.0** (more aggressive)
![high_contrast](extras_demo/contrast_2.0.png)

---

### Threshold

Sets the minimum saliency value to be considered "attended to," used primarily by `isolation` and `cutout` modes. Higher values make the visualization sparser — only the regions the model is most confident about survive. At 0.0, everything is included; at 0.5, only the top half of saliency values show.

Default: **0.25** — chosen to eliminate low-confidence gradient noise while preserving the meaningful attended region.

**threshold = 0.0** (nothing filtered)
![low_threshold](extras_demo/threshold_0.png)

**threshold = 0.5** (aggressive filtering)
![high_threshold](extras_demo/threshold_0.5.png)

---

### Class Index

ImageNet has 1,000 classes. By default (`-1`), the node auto-detects the model's top prediction and visualizes attention for that class — the attention pattern for the question the model is actually answering.

You can override this to force visualization of a different class. For example, forcing class 207 (golden retriever) on a human portrait produces attention patterns for a question the model was never meant to answer — it looks for golden retriever features in a human face. This produces attention patterns that feel alien and misaligned, which has significant artistic implications: it makes visible the categorical violence of being misclassified, or the absurdity of applying animal classification to a human subject.

The classification label output of the node always shows what class is actually being visualized.

**class_index = 1** (a modest override)
![index-1](extras_demo/class_index_1.png)

**class_index = 10** (a larger displacement)
![index-10](extras_demo/class_index_10.png)

---

### Input Size

All these models were trained on fixed-size square inputs. 224×224 is standard for most models; InceptionV3 requires 299×299 because it was trained at that resolution. The image is resized to this before processing, so the output appears at that resolution. Changing this for non-Inception models produces undefined behavior and is not recommended.

Default: **224** — not adjusted in demos because most models require it.

---

### Use SmoothGrad

When enabled on top of any method (except SmoothGrad itself), applies the SmoothGrad noise-averaging process to smooth the result. Generally produces more aesthetically pleasing, less noisy outputs at the cost of being proportionally slower (it runs the full forward/backward pass `smoothgrad_samples` times).

Default: **true** — the aesthetic improvement is significant and the default sample count of 25 is fast enough for most use.

**use_smoothgrad = false**
![smooth_false](extras_demo/smooth_grad_false.png)

---

### Smoothgrad Samples

How many noisy samples to average when SmoothGrad is active. 25 is a good balance between smoothness and speed. 50+ produces noticeably smoother results but takes proportionally longer — on CPU with VGG16, 50 samples takes roughly twice as long as 25.

Default: **25**

**samples = 50**
![samples_50](extras_demo/samples_50.png)

---

*Tool built with [PAIR Saliency](https://github.com/PAIR-code/saliency), [PyTorch](https://pytorch.org/), and [ComfyUI](https://github.com/comfyanonymous/ComfyUI).*
