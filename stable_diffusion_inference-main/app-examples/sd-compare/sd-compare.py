# !pip install taming-transformers-rom1504 -q
# !pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q
# !pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"
"""Gradio App to show difference between Stable diffusion 1.5 and 2.0"""
import lightning as L

from stable_diffusion_inference.cloud import SDComparison

component = L.LightningApp(
    SDComparison(cloud_compute=L.CloudCompute("gpu-fast", disk_size=30))
)
