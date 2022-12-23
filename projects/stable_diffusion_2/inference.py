# Third-party imports.
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-2-1"

scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "nemo having a steak dinner at totti's"
pipe_output = pipe(prompt)
image = pipe_output.images[0]

image.save("nemo_having_a_steak_dinner_at_tottis.png")
