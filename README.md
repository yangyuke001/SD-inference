# Easy Stable Diffusion

Simple and easy stable diffusion inference with LightningModule on GPU, CPU and MPS (Possibly all devices supported by [Lightning](https://lightning.ai)).

**To install**

```
pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -q
pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q
```

## Model variants

| Name     | Variant                          | Image Size |
|----------|----------------------------------|------------|
| sd1      | Stable Diffusion 1.5             | 512        |
| sd1.5    | Stable Diffusion 1.5             | 512        |
| sd1.4    | Stable Diffusion 1.4             | 512        |
| sd2_base | SD 2.0 trained on image size 512 | 512        |
| sd2_high | SD 2.0 trained on image size 768 | 768        |


## Example

```python
from stable_diffusion_inference import create_text2image

# text2image = create_text2image("sd1")
# text2image = create_text2image("sd2_high")  # for SD 2.0 with 768 image size
text2image = create_text2image("sd2_base")  # for SD 2.0 with 512 image size

image = text2image("cats in hats", image_size=512, inference_steps=50)
image.save("cats in hats.png")
```
