# Composition Examples
Keeping everything the same, but changing the composition to demonstrate visual differences.

## Understanding compostion modes
These control how the saliency map is composited back onto the original image. This is where the artistic intent lives.
isolation — The default and most conceptually pointed for your project. Salient regions show in full color using the chosen colormap; everything else is converted to greyscale. The effect: the machine's gaze burns in color while everything it ignores fades to grey. Very direct visual commentary on algorithmic attention.
overlay — Classic heatmap blend. The colormap is alpha-composited over the original at the intensity level. Familiar, readable, less subversive.
spotlight — The salient regions are bright, everything else darkens dramatically. Feels like a searchlight or surveillance camera — forensic, clinical.
ghost — The original image fades to 25% opacity and the saliency map glows over it. The subject becomes spectral, haunted by its own classification. Good for an uncanny effect.
invert — Shows what the algorithm ignores instead of what it sees. The saliency map is flipped before compositing. Conceptually this is the most subversive mode — it visualizes the blind spots, the parts of a face that are invisible to the machine.
cutout — Hard binary mask. Pixels above the threshold value show the original image; everything below shows a dim version of the heatmap. Creates a stark, graphic effect — the subject literally cut out by the algorithm's attention boundary.
multiply — Photoshop-style multiply blend. Attention dims non-salient areas. Produces a dark, moody result where ignored regions sink into shadow.
screen — Photoshop-style screen blend. Attention brightens salient regions. Lighter and more luminous than multiply — glowing rather than shadowed.
triptych — Outputs three panels side by side: original | saliency map alone | composite. Designed for gallery or exhibition display — shows the full chain of the visualization in one image.
mask_only — Just the colored saliency map with no original image. Pure algorithmic output, no human context.

## What is it? 

Model_name: vgg16
method: guided IG
composition: THIS IS WHAT CHANGES
colormap: magma
intensity: 0.75
contrast: 1.5
threshold: 0.25
class_index: -1
input_size: 224
uses_smooth_grad: true
snoothgrad_samples: 25


## What is it?


cutout
![cutout](method_demo/cutout.png)

ghost
![ghost](method_demo/ghost.png)

invert
![invert](method_demo/invert.png)

isolation
![isolation](method_demo/isolation.png)

mask only
![mask](method_demo/mask_only.png)

multiply
![multiply](method_demo/multiply.png)

overlay
![overlay](method_demo/overlay.png)

pure white
![white](method_demo/pure_white.png)

screen
![screen](method_demo/screen.png)

triplych
![triplych](method_demo/triplych.png)