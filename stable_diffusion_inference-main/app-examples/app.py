# !pip install taming-transformers-rom1504 -q
# !pip install -U "clip@ git+https://github.com/openai/CLIP.git@main" -q
# !pip install "sd_inference@git+https://github.com/aniketmaurya/stable_diffusion_inference@main"
# !pip install gradio==3.12.0
"""Serve Stable Diffusion on Lightning AI Cloud"""
import lightning as L

from stable_diffusion_inference.cloud import SDServe

component = SDServe(cloud_compute=L.CloudCompute("gpu", disk_size=30))
app = L.LightningApp(component)
