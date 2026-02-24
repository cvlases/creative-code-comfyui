# Colormap Examples
Keeping everything the same, but changing the color map to demonstrate visual differences.

## understanding Colormaps
These apply a color gradient to the greyscale saliency values. Low values (less important) get one end of the gradient, high values (more important) get the other.
inferno — Black → purple → orange → yellow. The default. Dramatic, high contrast.
plasma — Purple → pink → yellow. Vibrant and synthetic-looking. Feels digital, almost neon.
magma — Black → deep purple → pink-white. Darker and contrast than inferno.
viridis — Purple → teal → yellow-green. The standard scientific colormap, designed to be perceptually uniform. 
hot — Black → red → orange → white. Classic heat map / thermal camera. 
cool — Cyan → magenta. Cold, clinical. 
spring — Magenta → yellow. Neon, synthetic, light greens.
copper — Black → dark brown → copper-gold. Archival, almost daguerreotype-y.
twilight —  purple → white → orange → purple. Atmospheric brown. 
ocean — Black-blue → cyan → white. Deep and watery blue

## What is it? 

Model_name: vgg16
method: guided IG
composition: inverse
colormap: THIS IS WHAT CHANGES
intensity: 0.75
contrast: 1.5
threshold: 0.25
class_index: -1
input_size: 224
uses_smooth_grad: true
snoothgrad_samples: 25


## listed


cool
![cool](method_demo/cool.png)

hot
![hot](method_demo/hot.png)

copper
![copper](method_demo/copper.png)

inferno
![inferno](method_demo/inferno.png)

magma
![magma](method_demo/magma.png)

ocean
![ocean](method_demo/ocean.png)

plasma
![plasma](method_demo/plasma.png)

spring
![spring](method_demo/spring.png)

twilight
![twilight](method_demo/twilight.png)

viridis
![viridis](method_demo/viridis.png)