import os
import tarfile
import typing
import urllib.request
from functools import partial
from pathlib import Path
from typing import List

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import PromptDataset
from .lit_model import SUPPORTED_VERSIONS, StableDiffusionModule, clear_cuda


def download_checkpoints(
    ckpt_path: str,
    cache_dir: typing.Optional[str] = None,
    force_download: typing.Optional[bool] = None,
    ckpt_filename: typing.Optional[str] = None,
) -> str:
    if ckpt_path.startswith("http"):
        # Ex: pl-public-data.s3.amazonaws.com/dream_stable_diffusion/512-base-ema.ckpt
        ckpt_url = ckpt_path
        cache_path = Path(cache_dir) if cache_dir else Path()
        cache_path.mkdir(parents=True, exist_ok=True)
        dest = str(cache_path / os.path.basename(ckpt_path))
        # Ex: ./512-base-ema.ckpt
        if dest.endswith(".tar.gz"):
            # Ex: https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/sd_weights.tar.gz
            ckpt_folder = dest.replace(".tar.gz", "")  # Ex: ./sd_weights
            if not ckpt_filename:  # Ex: sd-v1-4.ckpt
                raise Exception("ckpt_filename must not be None")
            ckpt_path = str(
                Path(ckpt_folder) / ckpt_filename
            )  # Ex: ./sd_weights/sd-v1-4.ckpt
        else:
            ckpt_path = dest  # Ex: ./512-base-ema.ckpt
        if Path(ckpt_path).exists() and not force_download:
            return ckpt_path
        else:
            print("downloading checkpoints. This can take a while...")
            urllib.request.urlretrieve(ckpt_url, dest)
            if dest.endswith(".tar.gz"):
                file = tarfile.open(dest)
                file.extractall(cache_path)
                file.close()
                os.unlink(dest)
            return ckpt_path
    else:
        return ckpt_path  # Ex: ./sd_weights/sd-v1-4.ckpt


class SDInference:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        cache_dir: typing.Optional[str] = None,
        force_download: typing.Optional[bool] = None,
        ckpt_filename: typing.Optional[str] = None,
        accelerator: str = "auto",
        version="1.5",
    ):
        assert (
            version in SUPPORTED_VERSIONS
        ), f"supported version are {SUPPORTED_VERSIONS}"
        checkpoint_path = download_checkpoints(
            checkpoint_path, cache_dir, force_download, ckpt_filename
        )

        self.use_cuda: bool = torch.cuda.is_available() and accelerator in (
            "auto",
            "gpu",
        )
        precision = 16 if self.use_cuda else 32
        self.trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

        device = self.trainer.strategy.root_device.type

        clear_cuda()
        self.model = StableDiffusionModule(
            device=device,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            version=version,
        )
        if self.use_cuda:
            self.model = self.model.to(torch.float16)
            clear_cuda()

    def __call__(
        self, prompts: List[str], image_size: int = 768, inference_steps: int = 50
    ) -> typing.Union[List[Image.Image], Image.Image]:
        if isinstance(prompts, str):
            prompts = [prompts]
        trainer = self.trainer
        model = self.model

        img_dl = DataLoader(
            PromptDataset(prompts), batch_size=len(prompts), shuffle=False
        )
        model.predict_step = partial(
            model.predict_step,
            height=image_size,
            width=image_size,
            num_inference_steps=inference_steps,
        )
        pil_results = trainer.predict(model, dataloaders=img_dl)[0]
        if len(pil_results) == 1:
            return pil_results[0]
        return pil_results
