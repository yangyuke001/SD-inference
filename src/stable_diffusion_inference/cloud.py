import base64
from functools import partial
from io import BytesIO

import gradio as gr
from lightning.app.components.serve import Image, PythonServer, ServeGradio
from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class SDServe(PythonServer):
    """
    **To use the API as a client:**
    ```python
    import requests
    response = requests.post("https://tyixm-01gkgswq452hy2n6grkpz4je6v.litng-ai-03.litng.ai/predict", json={
      "prompt": "data string"
    })
    print(response.json())
    ```
    """

    def __init__(self, sd_variant="sd1", **kwargs):
        super().__init__(input_type=Prompt, output_type=Image, **kwargs)
        self.sd_variant = sd_variant

    def setup(self, *args, **kwargs) -> None:
        from stable_diffusion_inference import create_text2image

        self._model = create_text2image(self.sd_variant)

    def predict(self, request: Prompt):
        image = self._model(request.prompt)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}


class SDComparison(ServeGradio):
    inputs = gr.Textbox(label="Prompt", value="Cats in hats")
    outputs = [gr.Image(label="SD 1"), gr.Image(label="SD 2")]
    title = "Compare images from Stable Diffusion 1 and 2.0"

    def build_model(self):
        from stable_diffusion_inference import create_text2image

        sd1 = create_text2image("sd1")
        sd2 = create_text2image("sd2_base")  # for SD 2.0 with 512 image size
        return {
            "sd1": partial(sd1, image_size=512),
            "sd2": partial(sd2, image_size=512),
        }

    def predict(self, prompt: str):
        image1 = self.model["sd1"](prompts=prompt)
        image2 = self.model["sd2"](prompts=prompt)
        return [image1, image2]
