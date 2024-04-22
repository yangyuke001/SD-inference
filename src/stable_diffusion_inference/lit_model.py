import typing
from typing import Any, List

import lightning as L
import numpy as np
import torch
from PIL import Image

from ldm1.models.diffusion.ddim import DDIMSampler as SD1Sampler
from ldm2.models.diffusion.ddim import DDIMSampler as SD2Sampler

SAMPLERS = {"1.4": SD1Sampler, "1.5": SD1Sampler, "2.0": SD2Sampler}
SUPPORTED_VERSIONS = {"1.4", "1.5", "2.0"}


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


DOWNSAMPLING_FACTOR = 8
UNCONDITIONAL_GUIDANCE_SCALE = 9.0  # SD2 need higher than SD1 (~7.5)


def load_model_from_config(
    config: Any, ckpt: str, version: str, verbose: bool = False
) -> torch.nn.Module:
    if version == "2.0":
        from ldm2.util import instantiate_from_config

    elif version.startswith("1."):
        from ldm1.util import instantiate_from_config
    else:
        raise NotImplementedError(
            f"version={version} not supported. {SUPPORTED_VERSIONS}"
        )

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model


class StableDiffusionModule(L.LightningModule):
    def __init__(
        self, device: torch.device, config_path: str, checkpoint_path: str, version: str
    ):
        from omegaconf import OmegaConf

        if version == "2.0":
            SamplerCls = SAMPLERS[version]

        elif version.startswith("1."):
            SamplerCls = SAMPLERS[version]
        else:
            raise NotImplementedError(
                f"version={version} not supported. {SUPPORTED_VERSIONS}"
            )

        super().__init__()

        config = OmegaConf.load(f"{config_path}")
        config.model.params.cond_stage_config["params"] = {"device": device}
        self.model = load_model_from_config(
            config, f"{checkpoint_path}", version=version
        )
        self.sampler = SamplerCls(self.model)

    @typing.no_type_check
    @torch.inference_mode()
    def predict_step(
        self,
        prompts: List[str],
        batch_idx: int,
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> Any:
        batch_size = len(prompts)

        with self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompts)
            shape = [4, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR]
            samples_ddim, _ = self.sampler.sample(
                S=num_inference_steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=UNCONDITIONAL_GUIDANCE_SCALE,
                unconditional_conditioning=uc,
                eta=0.0,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_samples_ddim = (255.0 * x_samples_ddim).astype(np.uint8)
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]

        return pil_results
