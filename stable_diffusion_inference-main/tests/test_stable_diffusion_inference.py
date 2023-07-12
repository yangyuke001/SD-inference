import os.path
import tempfile

from PIL import Image

from stable_diffusion_inference import create_text2image


def test_create_text2image():
    cache_dir = tempfile.mkdtemp()
    text2image = create_text2image("sd1", cache_dir=cache_dir)
    image = text2image("cats in hats", image_size=512, inference_steps=1)
    assert isinstance(image, Image.Image)
    # assert os.path.exists(f"{cache_dir}/sd_weights/sd-v1-4.ckpt")  # model is ready only if weights are available
