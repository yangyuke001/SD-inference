from stable_diffusion_inference import create_text2image

text2image = create_text2image("sd1")
# text2image = create_text2image("sd2_high")  # for SD 2.0 with 768 image size
# text2image = create_text2image("sd2_base")  # for SD 2.0 with 512 image size


image = text2image("cats in hats", image_size=512, inference_steps=1)
image.save("cats in hats.png")
