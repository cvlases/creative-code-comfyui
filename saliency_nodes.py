# Running the Saliency Art Node

This tool is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), an open-source node-based interface for running generative AI pipelines locally. The node interrogates pretrained image classifiers using gradient-based saliency methods, visualizing what the model attends to when it classifies an image.

---

## What is ComfyUI?

ComfyUI is a local, node-based interface for building AI image pipelines. Unlike web-based tools, it runs entirely on your machine — no API keys, no cloud, no data leaving your computer. You build workflows by connecting nodes: each node does one thing (load an image, run a model, save output), and you wire them together visually.

This project adds a custom node set to ComfyUI that exposes saliency visualization as a first-class pipeline step — meaning you can chain it with other nodes, batch process images, and experiment with parameters in real time.

---

## Running ComfyUI Locally

### Requirements

- Python 3.10+ (this project used **Python 3.13.5** via Anaconda)
- A conda or virtual environment is strongly recommended
- Mac Apple Silicon (M-series), Windows, or Linux
- GPU optional but recommended — this node runs on CPU by design (see [Technical Notes](#technical-notes))

### Installation
```bash
# Clone the ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create and activate a conda environment
conda create -n comfyenv python=3.13
conda activate comfyenv

# Install dependencies
pip install -r requirements.txt

# Start ComfyUI
python main.py
```

Then open `http://127.0.0.1:8188` in your browser. You'll see the node canvas.

---

## Installing the Saliency Node

### 1. Install Python dependencies

With your ComfyUI conda environment active:
```bash
pip install saliency matplotlib torchvision
```

### 2. Place the node files

Copy the `creative_code/` folder into ComfyUI's custom nodes directory:
```
ComfyUI/
└── custom_nodes/
    └── creative_code/
        ├── __init__.py
        └── saliency_nodes.py
```

`__init__.py` tells ComfyUI this folder is a node package. It should contain:
```python
from .saliency_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
```

### 3. Restart ComfyUI
```bash
python main.py
```

You should see this in the startup log, confirming the node loaded:
```
0.0 seconds: /path/to/ComfyUI/custom_nodes/creative_code
```

The nodes will appear in the node menu under **creative-code/art** and **creative-code/explainability**.

---

## Basic Workflow

The minimal working pipeline is three nodes:
```
[Load Image] → [SaliencyArt] → [Preview Image]
```

1. Right-click the canvas → Add Node → image → **Load Image**
2. Right-click → Add Node → creative-code/art → **Saliency Art — Algorithmic Attention**
3. Right-click → Add Node → image → **Preview Image**
4. Connect **Load Image** `IMAGE` output → **SaliencyArt** `image` input
5. Connect **SaliencyArt** `artwork` output → **Preview Image** `images` input
6. Upload an image in the Load Image node
7. Click **Queue Prompt**

The first run downloads the model weights (~500MB for VGG16) automatically. Subsequent runs use the cached weights.

The node also outputs a `classification_label` string — wire this to a **Show Text** node to see what the model classified the image as and which class is being visualized.

---

## How the Node Was Built

ComfyUI custom nodes are Python classes that follow a specific interface. Each node declares its inputs, outputs, and a function that gets called when the pipeline runs.

### Node structure

A minimal ComfyUI node looks like this:
```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "my_param": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "my-category"

    def run(self, image, my_param):
        # do something with image
        return (processed_image,)

NODE_CLASS_MAPPINGS = {"MyNode": MyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MyNode": "My Node (Display Name)"}
```

ComfyUI reads `NODE_CLASS_MAPPINGS` at startup and registers each class as a node. The `INPUT_TYPES` dict defines what appears in the node's UI — dropdowns for string lists, sliders for floats, toggles for booleans. `RETURN_TYPES` defines the output ports.

### The scaffolding problem

The recommended way to create a new node is via a cookiecutter template:
```bash
pip install cookiecutter
cookiecutter https://github.com/bronkula/comfyui-node-template
```

However, the template has a known issue: it generates `__init.py__` (wrong) instead of `__init__.py` (correct), which causes ComfyUI to silently skip the node at startup with no error. The fix is to rename the file manually before doing anything else.

### The key technical challenge: gradients inside ComfyUI

The core difficulty in building this node was that **ComfyUI wraps all node execution inside `torch.inference_mode()`** — a PyTorch context that disables the autograd engine entirely for performance. This is fine for diffusion models that only need forward passes, but saliency methods require computing gradients (backward passes), which inference mode makes impossible.

Three layers of fixes were required:

**1. Load models outside inference mode.** Model weights created inside `inference_mode()` are permanently marked as "inference tensors" and can't participate in gradient computation even if you later try to enable gradients. The solution is to wrap the entire model loading step in `torch.inference_mode(False)`:
```python
with torch.inference_mode(False):
    with torch.enable_grad():
        model = torchvision.models.vgg16(weights=...)
        model.eval().cpu()
```

**2. Escape inference mode at call time.** Even with models loaded correctly, the `call_model_function` that the saliency library calls repeatedly during computation runs inside ComfyUI's inference mode context. Each call needs to explicitly break out:
```python
def call_model_function(images, ...):
    with torch.inference_mode(False):
        with torch.enable_grad():
            images_tensor = torch.from_numpy(images).clone().requires_grad_(True)
            output = model(images_tensor)
            output[:, class_idx].sum().backward()
            gradients = images_tensor.grad.detach().numpy()
```

**3. Force CPU for gradient computation.** On Apple Silicon (MPS backend), PyTorch's MPS implementation drops `requires_grad` silently during certain operations, making backward passes fail even when autograd is enabled. All gradient computation runs on CPU; the classification forward pass (which doesn't need gradients) can use MPS for speed.

### File structure
```
creative_code/
├── __init__.py          # registers nodes with ComfyUI
└── saliency_nodes.py    # all node logic
    ├── _load_model()         # loads + caches torchvision models
    ├── _make_call_model_fn() # builds the saliency library callback
    ├── _predict_top_class()  # auto-detects ImageNet class
    ├── _run_saliency_method() # dispatches to PAIR saliency methods
    ├── _compose()            # applies composition modes
    ├── SaliencyMapNode       # single method output
    ├── SaliencyComparisonNode # all-methods grid
    └── SaliencyArtNode       # main art node
```

---

## Technical Notes

**Why CPU?** Gradient-based saliency on Apple Silicon (MPS) silently drops `requires_grad` during backward passes, producing incorrect results with no error message. All saliency computation runs on CPU to guarantee correctness. The classification forward pass (auto-detecting the top class) still uses MPS for speed since it doesn't require gradients.

**Why these specific library versions?** The PAIR saliency library's `BlurIG` method uses `steps` as its parameter name while all other IG-family methods use `x_steps` — an inconsistency in their API. The node handles this internally so you don't have to.

**XRAI is slower.** XRAI uses a fundamentally different algorithm (image segmentation + region attribution) rather than pixel-level gradients. It typically takes several minutes on CPU versus seconds for gradient methods.

**Model weights are cached.** The first time you select a model, PyTorch downloads the pretrained weights (~100–550MB depending on the model) to `~/.cache/torch/hub/checkpoints/`. Subsequent runs load from cache instantly.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `saliency` | Google PAIR saliency methods (GradCAM, IG, XRAI, etc.) |
| `torchvision` | Pretrained ImageNet models |
| `matplotlib` | Colormaps |
| `torch` | Included with ComfyUI |
| `numpy` | Included with ComfyUI |