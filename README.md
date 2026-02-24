# Running the Saliency Art Node

This is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a local node-based interface for building AI pipelines. The node lets you point a pretrained image classifier at a photo and visualize what it's actually paying attention to when it makes a decision.

---

## What is ComfyUI?

ComfyUI is essentially a visual programming environment for AI that runs entirely on your own machine. You build workflows by connecting nodes — each node does one thing, and you wire them together. No cloud, no API keys, nothing leaves your computer.

I used it for this project because it let me treat saliency visualization as just another step in a pipeline — something I could experiment with in real time, swap parameters on the fly, and chain with other nodes. It's a bit of a learning curve at first but once it clicks, it's a really satisfying way to work.

---

## Running ComfyUI Locally

### What you need

- Python 3.10+ (I used **Python 3.13.5** via Anaconda)
- A conda environment keeps things clean and is strongly recommended
- Mac, Windows, or Linux — I ran this on a MacBook Pro M-series

### Setup
```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create a conda environment
conda create -n comfyenv python=3.13
conda activate comfyenv

# Install dependencies
pip install -r requirements.txt

# Run it
python main.py
```

Open `http://127.0.0.1:8188` in your browser and you'll see the node canvas. That's it.

---

## Installing the Node

### 1. Install the extra Python packages

With your ComfyUI environment active:
```bash
pip install saliency matplotlib torchvision
```

### 2. Drop the files in

Copy the `creative_code/` folder into ComfyUI's custom nodes directory:
```
ComfyUI/
└── custom_nodes/
    └── creative_code/
        ├── __init__.py
        └── saliency_nodes.py
```

### 3. Restart ComfyUI
```bash
python main.py
```

If it worked, you'll see this in the startup log:
```
0.0 seconds: /path/to/ComfyUI/custom_nodes/creative_code
```

The nodes show up in the node menu under **creative-code/art** and **creative-code/explainability**.

---

## Basic Workflow

The simplest setup is three nodes:
```
[Load Image] → [SaliencyArt] → [Preview Image]
```

1. Right-click the canvas → Add Node → image → **Load Image**
2. Right-click → Add Node → creative-code/art → **Saliency Art**
3. Right-click → Add Node → image → **Preview Image**
4. Wire them: Load Image `IMAGE` → SaliencyArt `image` → Preview Image `images`
5. Upload a photo, hit **Queue Prompt**

The first run will download the model weights automatically (~500MB for VGG16) and cache them. Every run after that is instant.

You can also wire the `classification_label` output to a **Show Text** node — it'll tell you what class the model predicted and which one is being visualized.

---

## How I Built the Node (and What Went Wrong)

ComfyUI custom nodes are Python classes that follow a specific interface — you define your inputs, outputs, and a function that runs when the pipeline executes. Straightforward in theory.

The structure looks like this:
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
        return (processed_image,)

NODE_CLASS_MAPPINGS = {"MyNode": MyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MyNode": "My Node"}
```

ComfyUI reads `NODE_CLASS_MAPPINGS` at startup and registers each class. The `INPUT_TYPES` dict builds the UI automatically — dropdowns, sliders, toggles, all from the type declarations.

### The cookiecutter trap

The recommended starting point for a new node is a cookiecutter scaffold template. I used it. The template generated `__init.py__` instead of `__init__.py` — wrong placement of the double underscores — which caused ComfyUI to silently skip the node at startup with zero error messages. Took a while to figure out. Rename the file before you do anything else.

### The actual hard part: gradients inside ComfyUI

This was the real challenge. ComfyUI wraps all node execution inside `torch.inference_mode()` — a PyTorch context that disables the gradient engine entirely for performance. That's totally fine for diffusion models that only need forward passes. But saliency methods work by computing gradients (backward passes through the network), which inference mode makes completely impossible.

It took several iterations to get this right. The fix ended up needing three layers:

**1. Load models outside inference mode.**
Model weights created inside inference mode are permanently tagged as "inference tensors" — they can't participate in gradient computation even if you enable gradients later. Solution: wrap the entire model load in `torch.inference_mode(False)`:
```python
with torch.inference_mode(False):
    with torch.enable_grad():
        model = torchvision.models.vgg16(weights=...)
        model.eval().cpu()
```

**2. Escape inference mode on every call.**
The saliency library calls your model function repeatedly during computation, and each call happens inside ComfyUI's inference mode context. Each call needs to break out explicitly:
```python
def call_model_function(images, ...):
    with torch.inference_mode(False):
        with torch.enable_grad():
            images_tensor = torch.from_numpy(images).clone().requires_grad_(True)
            output = model(images_tensor)
            output[:, class_idx].sum().backward()
            gradients = images_tensor.grad.detach().numpy()
```

**3. Force CPU on Apple Silicon.**
On MPS (the Mac GPU backend), PyTorch silently drops `requires_grad` during certain operations — backward passes fail with no useful error. Everything gradient-related runs on CPU. The classification forward pass (which doesn't need gradients) still uses MPS for speed.

Getting here involved a lot of error messages that all looked identical but had different root causes. The errors were:

- `element 0 of tensors does not require grad` → inference mode not escaped
- `Inference tensors cannot be saved for backward` → model weights created inside inference mode
- `Input type (MPSFloatType) and weight type (torch.FloatTensor) should be the same` → device mismatch between tensor and model

Each one required a different fix. Fun times.

---

## A Few Things Worth Knowing

**The first run is slow.** PyTorch downloads pretrained weights the first time you select a model (~100–550MB depending on which one). After that it's cached at `~/.cache/torch/hub/checkpoints/` and loads instantly.

**XRAI takes a lot longer than the other methods.** It works completely differently — segmenting the image into regions rather than computing pixel-level gradients — so it's slower by nature. Budget a few minutes rather than a few seconds.

**The PAIR saliency library has one quirk.** `BlurIG` uses `steps` as its parameter name while every other IG-family method uses `x_steps`. Inconsistency in their API that the node handles internally so you don't have to think about it.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `saliency` | Google PAIR saliency methods |
| `torchvision` | Pretrained ImageNet models |
| `matplotlib` | Colormaps |
| `torch` | Comes with ComfyUI |
| `numpy` | Comes with ComfyUI |

## Resources

- [ComfyUI JS Extension Documentation](https://docs.comfy.org/custom-nodes/js/javascript_overview) - Official documentation for ComfyUI JavaScript Extensions
- [ComfyUI Registry Documentation](https://docs.comfy.org/registry/publishing) - Learn how to publish your extension
- [ComfyUI Frontend Repository](https://github.com/cvlases/ComfyUI-Frontend) - The main ComfyUI frontend codebase
- [Official ComfyUI Frontend Types](https://www.npmjs.com/package/@comfyorg/comfyui-frontend-types) - TypeScript definitions for ComfyUI
- [Claude Code](https://code.claude.com/docs/en/overview)
- [Saliency Tutorial](https://github.com/PAIR-code/saliency/blob/master/Examples_pytorch.ipynb)



## License

MIT license
