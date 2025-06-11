import torch
import diffusers
import tiny_hf
import tiny_hf as thf
import numpy as np
import os
import tinygrad

from tg_adapter import F
import tg_adapter
from PIL import Image

from operator_tests import *
from module_tests import *
from testing_utils import compare_state_dicts, copy_state_dict, \
	inspect_state_dict_devices, make_test_data, test_function, \
	test_hf_reimplementation, mse, norm_mse, _test_key_errors, \
	get_submodules, test_all_submodules

	

def test_autoencoderkl(hf_module = None, my_module = None):
	inp = np.random.randn(2*3*64*64).reshape(2, 3, 64, 64).astype(np.float32)
	args = (3,)
	
	# from single file is broken atm
	hf_module = diffusers.AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors" )
	my_module = thf.diffusers.models.autoencoders.AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors" )
	
	#copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(inp, {}, hf_module, "__call__", my_module, "__call__")
	test_all_submodules(hf_module, my_module)

def test_state_dict():
	raise NotImplementedError

def test_autoencoderkl_from_single_file():
	inp = np.random.randn(2*3*128*128).reshape(2, 3, 128, 128).astype(np.float32)
	args = (3,)
	
	hf_module = diffusers.AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors" )
	hf_module.enable_tiling()
	
	my_module = thf.diffusers.models.autoencoders.AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors" )
	my_module.enable_tiling()
	
	compare_state_dicts(hf_module, my_module)
	#copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(inp, {}, hf_module, "__call__", my_module, "__call__")

	
	
def test_downsample2d():
	from tiny_hf.diffusers.models.downsampling import Downsample2D as thf_Downsample2D
	from diffusers.models.resnet import Downsample2D as hf_Downsample2D
	
	a = make_test_data(2, 3, 32, 32)
	
	hf_module = hf_Downsample2D(3)
	my_module = thf_Downsample2D(3)
	
	# now we need to transfer state dicts...
	copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(a, {}, hf_module, "forward", my_module, "forward")

def test_resnet_block2d():
	from diffusers.models.resnet import ResnetBlock2D as hf_ResnetBlock2D
	from tiny_hf.diffusers.models.resnet import ResnetBlock2D as thf_ResnetBlock2D
	
	a = make_test_data(2, 3, 32, 32)
	
	hf_module = hf_ResnetBlock2D(in_channels = 3, groups = 1, temb_channels = None)
	my_module = thf_ResnetBlock2D(in_channels = 3, groups = 1, temb_channels = None)
	
	args = (a, None)
	
	copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(args, {}, hf_module, "forward", my_module, "forward")

def test_vae_encoder():
	from diffusers.models.autoencoders.vae import Encoder as hf_Encoder
	from tiny_hf.diffusers.models.autoencoders.vae import Encoder as thf_Encoder
	
	a = make_test_data(1, 3, 8, 8)
	
	hf_module = hf_Encoder(in_channels = 3, out_channels = 4, mid_block_add_attention = True)
	my_module = thf_Encoder(in_channels = 3, out_channels = 4, mid_block_add_attention = True)
	
	args = (a,)
	
	copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(args, {}, hf_module, "forward", my_module, "forward")
	compare_state_dicts(hf_module, my_module)

def test_down_encoder_block_2d():
	from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D as hf_DownEncoderBlock2D
	from tiny_hf.diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D as thf_DownEncoderBlock2D
	
	a = make_test_data(2, 3, 32, 32)
	
	hf_module = hf_DownEncoderBlock2D(in_channels = 3, out_channels = 4, resnet_groups = 1)
	my_module = thf_DownEncoderBlock2D(in_channels = 3, out_channels = 4, resnet_groups = 1)
	
	args = (a,)
	
	copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(args, {}, hf_module, "forward", my_module, "forward")

def test_upsampling_2d():
	# I really need to reduce the silly code required for this...or do I?
	from tiny_hf.diffusers.models.upsampling import Upsample2D as thf_Upsample2D
	from diffusers.models.upsampling import Upsample2D as hf_Upsample2D
	
	a = make_test_data(2, 4, 32, 32)
	
	hf_module = hf_Upsample2D(4, out_channels = 3, )
	my_module = thf_Upsample2D(4, out_channels = 3, kernel_size = 3)
	
	args = (a,)
	
	copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(args, {}, hf_module, "forward", my_module, "forward")

def test_vae_decoder():
	from diffusers.models.autoencoders.vae import Decoder as hf_Decoder
	from tiny_hf.diffusers.models.autoencoders.vae import Decoder as thf_Decoder
	
	a = make_test_data(2, 4, 16, 16)
	
	hf_module = hf_Decoder(in_channels = 4, out_channels = 3, mid_block_add_attention = True)
	my_module = thf_Decoder(in_channels = 4, out_channels = 3, mid_block_add_attention = True)
	
	args = (a,)
	
	copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(args, {}, hf_module, "forward", my_module, "forward")

def test_UNetMidBlock2D():
	from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D as hf_UNetMidBlock2D
	from tiny_hf.diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D as thf_UNetMidBlock2D
	
	a = make_test_data(2, 64, 16, 16)
	
	hf_module =hf_UNetMidBlock2D(64, None, add_attention = False)
	my_module = thf_UNetMidBlock2D(64, None, add_attention = False)
	
	args = (a,)
	
	copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(args, {}, hf_module, "forward", my_module, "forward")

def test_unet_2d_condition(hf_module = None, thf_module = None,
		latent_shape = (2, 4, 32, 32), embed_shape = (2, 2, 1280) ):
	a = make_test_data(*latent_shape)
	emb = make_test_data(*embed_shape)
	
	if hf_module is None and thf_module is None:
		from diffusers import UNet2DConditionModel as hf_UNet2DConditionModel
		from tiny_hf.diffusers import UNet2DConditionModel as thf_UNet2DConditionModel
		hf_module = hf_UNet2DConditionModel()
		thf_module = thf_UNet2DConditionModel()
		copy_state_dict(hf_module, thf_module)
	
	args = (a, 20, emb)
	test_hf_reimplementation(args, {}, hf_module, "__call__", thf_module, "__call__")
	#input("lets do the submodule test!")
	#test_all_submodules(hf_module, thf_module)

def test_unet_2d():
	from diffusers.models.unets.unet_2d import UNet2DModel as hf_class
	from tiny_hf.diffusers.models.unets.unet_2d import UNet2DModel as thf_class
	
	a = make_test_data(2, 3, 32, 32)
	
	hf_module = hf_class()
	thf_module = thf_class()
	
	args = (a, 2)
	copy_state_dict(hf_module, thf_module)
	test_hf_reimplementation(args, {}, hf_module, "__call__", thf_module, "__call__")
"""
def test_named_modules():
	hf_module = diffusers.AutoencoderKL()
	my_module = thf.diffusers.models.autoencoders.AutoencoderKL()
	
	for name, module in hf_module.named_modules():
		print(name, module)
"""

def test_named_parameters():
	hf_module = diffusers.AutoencoderKL()
	my_module = thf.diffusers.models.autoencoders.AutoencoderKL()
	
	for (name, module), (my_name, my_module) in zip(hf_module.named_parameters(), my_module.named_parameters()):
		assert name == my_name
		
def test_functional_linear():
	from tg_adapter import F as tinyF
	import torch.nn.functional as torchF
	inp = make_test_data(40, 5)
	weight = make_test_data(5, 10)
	test_function(inp, weight, torchF.linear, tinyF.linear)



def test_clip_tokenizer():
	from tiny_hf.transformers import CLIPTokenizer as tg_module
	from transformers import CLIPTokenizer as hf_module
	tg = tg_module.from_pretrained("openai/clip-vit-base-patch32")
	hf = hf_module.from_pretrained("openai/clip-vit-base-patch32")
	args = ["the quick brown fox jumped over the lazy dog"]
	test_hf_reimplementation(args, {}, hf, "__call__", tg, "__call__")

def test_clip_tokenizer_fast():
	from tiny_hf.transformers import CLIPTokenizerFast as tg_module
	from transformers import CLIPTokenizerFast as hf_module
	tg = tg_module.from_pretrained("openai/clip-vit-base-patch32")
	hf = hf_module.from_pretrained("openai/clip-vit-base-patch32")
	args = ["the quick brown fox jumped over the lazy dog"]
	test_hf_reimplementation(args, {}, hf, "__call__", tg, "__call__")

def _convert_tokenizer_output(out):
	new_out = {}
	for k, v in out.items():
		if isinstance(v, list):
			v = np.array(v)
		new_out[k] = v
	return new_out
	
def test_clip_text_model():
	from tiny_hf.transformers import CLIPTextModel as tg_class
	from transformers import CLIPTextModel as hf_class
	
	hf_module = hf_class.from_pretrained("openai/clip-vit-large-patch14", use_safetensors = True)
	tg_module = tg_class.from_pretrained("openai/clip-vit-large-patch14", use_safetensors = True)
	
	# import the tokenizer first in order to do stuff correctly
	from transformers import CLIPTokenizer
	tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", use_safetensors = True)
	tokenizer_output = tokenizer(["the quick brown fox jumped over the lazy dog"], padding = True)
	#print(tokenizer_output)
	#input(type(tokenizer_output["input_ids"]) )
	tokenizer_output = _convert_tokenizer_output( tokenizer_output )
	test_hf_reimplementation([], tokenizer_output, hf_module, "__call__", tg_module, "__call__")


def test_audio_diffusion_pipeline():
	from tiny_hf.diffusers.pipelines import AudioDiffusionPipeline as tg_class
	from diffusers.pipelines import AudioDiffusionPipeline as hf_class
	
	hf_module = hf_class.from_pretrained("teticio/audio-diffusion-256", use_safetensors = False)
	tg_module = tg_class.from_pretrained("teticio/audio-diffusion-256", use_safetensors = False)
	
	test_hf_reimplementation([], {}, hf_module, "__call__", tg_module, "__call__")
	
def test_stable_diffusion_pipeline():
	from testing_rewrites import sd_pipeline_call, retrieve_timesteps
	from tiny_hf.diffusers.pipelines import StableDiffusionPipeline as tg_class
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	
	from diffusers.pipelines import StableDiffusionPipeline as hf_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	hf_module = hf_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, safety_checker = None, scheduler = hf_scheduler)
	tg_module = tg_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, safety_checker = None, scheduler = tg_scheduler)
	
	latents = make_test_data(1, 4, 64, 64)
	
	# then copy the state dict from the torch model to the tinygrad one and see if it helps at all
	copy_state_dict(hf_module.vae, tg_module.vae)
	#test_hf_reimplementation([img], {}, hf_module.vae, "__call__", tg_module.vae, "__call__")
	
	# lets do the submodule test on the vae just in case...
	#test_all_submodules(hf_module.vae, tg_module.vae)
	
	#input("does the vae work?")
	
	# test the unet
	#test_unet_2d_condition(hf_module.unet, tg_module.unet, latents.shape, (1, 77, 768) )
	copy_state_dict(hf_module.unet, tg_module.unet)
	#test_unet_2d_condition(hf_module.unet, tg_module.unet, latents.shape, (1, 77, 768) )
	#input("does the unet work?")
	
	
	
	# even after gelu was fixed, there is even more that aren't working :c
	# tiny_hf.diffusers.models.unets.unet_2d_blocks.UpBlock2D
	# tiny_hf.diffusers.models.attention_processor.Attention
	# tiny_hf.diffusers.models.attention.FeedForward
	# surprisingly conv2d is having problems in some circumstances
	# tg_adapter.nn.layers.Conv2d
	# 	Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	# 	Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	# 	output shape: (1, 1280, 16, 16)
	# tiny_hf.diffusers.models.resnet.ResnetBlock2D
	# tiny_hf.diffusers.models.unets.unet_2d_blocks.CrossAttnUpBlock2D
	
	
	# Looks like there may be a timestep problem.
	# Lets look at that...
	# First, we need to ensure that the timestep retrieval function isn't wrong
	#test_hf_reimplementation(
	
	hf_module.load_lora_weights("ybelkada/sd-1.5-pokemon-lora-peft")
	tg_module.load_lora_weights("ybelkada/sd-1.5-pokemon-lora-peft")
	
	# test prompt encoding
	#test_hf_reimplementation(["a squishy pp", "cpu", 1, True], {}, hf_module, "encode_prompt", tg_module, "encode_prompt")
	
	# test the image processor
	#test_hf_reimplementation([], {"prompt": "a fluffy bunny", "num_inference_steps": 2, "safety_checker": None, "latents": latents, "output_type": "latent"}, hf_module, sd_pipeline_call, tg_module, sd_pipeline_call, error_threshold = 1.0e-6)
	test_hf_reimplementation([], {"prompt": "a fluffy bunny pokemon", "num_inference_steps": 15, "safety_checker": None, "latents": latents, "output_type": "pil"}, hf_module, "__call__", tg_module, "__call__")

def test_named_modules():
	from testing_rewrites import sd_pipeline_call, retrieve_timesteps
	from tiny_hf.diffusers.pipelines import StableDiffusionPipeline as tg_class
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	
	from diffusers.pipelines import StableDiffusionPipeline as hf_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	hf_module = hf_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, safety_checker = None, scheduler = hf_scheduler)
	tg_module = tg_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, safety_checker = None, scheduler = tg_scheduler)
	
	copy_state_dict(hf_module.unet, tg_module.unet)
	def get_named_modules(x, _torch):
		modules = list(x.named_modules())
		keys = list(dict(modules).keys() )
		print(len(keys), len(modules))
		return keys
	test_hf_reimplementation([], {}, hf_module.unet, get_named_modules, tg_module.unet, get_named_modules)

def test_stable_diffusion_xl_pipeline():
	from diffusers import StableDiffusionXLPipeline as hf_class
	from tiny_hf.diffusers import StableDiffusionXLPipeline as tg_class
	
	hf_module = hf_class.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
	tg_module = tg_class.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
	
	# model loading is still broken :c
	copy_state_dict(hf_module.vae, tg_module.vae)
	copy_state_dict(hf_module.unet, tg_module.unet)
	
	test_hf_reimplementation([], {"prompt": "a cute fluffy bunny"}, hf_module, "__call__", tg_module, "__call__")

def test_ddim_scheduler():
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	hf_scheduler.set_timesteps(100)
	tg_scheduler.set_timesteps(100)
	
	_test_key_errors(hf_scheduler, tg_scheduler)
	
	
	def _test(scheduler, torchm):
		return scheduler.timesteps
	
	test_hf_reimplementation([], {}, hf_scheduler, _test, tg_scheduler, _test)
	
	latent = make_test_data(2, 4, 64, 64)
	noise = make_test_data(2, 4, 64, 64)
	
	def _test_denoise(scheduler, torchm, noise_, i_, latent_, *args, **kwargs):
		t = scheduler.timesteps[i]
		return scheduler.step(noise_, i_, latent_, *args, **kwargs)
		
	
	for i in range(100):
		#test_hf_reimplementation([noise, i, latent], {"return_dict": False}, hf_scheduler, "step", tg_scheduler, "step")
		test_hf_reimplementation([noise, i, latent], {"return_dict": False}, hf_scheduler, _test_denoise, tg_scheduler, _test_denoise)

def test_scheduler(hf_module = None, tg_module = None):
	if hf_module is None or tg_module is None:
		raise NotImplementedError
	raise NotImplementedError

def test_dtype_override():
	a = tg_adapter.arange(4, device = "cpu", dtype = tg_adapter.int64)
	b = a.to("cuda:0")
	print(b.dtype, b.tg.dtype)
	c = b.to("cpu")
	print(c.numpy() )
	print(c.dtype, c.tg.dtype)
	
def test_stable_diffusion_img2img():
	from testing_rewrites import sd_pipeline_call, retrieve_timesteps
	from tiny_hf.diffusers.pipelines import StableDiffusionImg2ImgPipeline as tg_class
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	
	from diffusers.pipelines import StableDiffusionImg2ImgPipeline as hf_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	hf_module = hf_class.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors = True, requires_safety_checker = False, safety_checker = None, scheduler = hf_scheduler)
	tg_module = tg_class.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors = True, requires_safety_checker = False, safety_checker = None, scheduler = tg_scheduler)
	print(hf_module.image_encoder)
	input(tg_module.image_encoder)
	latents = make_test_data(1, 4, 64, 64)
	
	# then copy the state dict from the torch model to the tinygrad one and see if it helps at all
	copy_state_dict(hf_module.vae, tg_module.vae)
	#test_hf_reimplementation([img], {}, hf_module.vae, "__call__", tg_module.vae, "__call__")
	
	# lets do the submodule test on the vae just in case...
	#test_all_submodules(hf_module.vae, tg_module.vae)
	
	#input("does the vae work?")
	
	# test the unet
	#test_unet_2d_condition(hf_module.unet, tg_module.unet, latents.shape, (1, 77, 768) )
	copy_state_dict(hf_module.unet, tg_module.unet)
	#test_unet_2d_condition(hf_module.unet, tg_module.unet, latents.shape, (1, 77, 768) )
	
	#test_hf_reimplementation([], {"prompt": "a fluffy bunny", "num_inference_steps": 2, "safety_checker": None, "latents": latents, "output_type": "latent"}, hf_module, sd_pipeline_call, tg_module, sd_pipeline_call, error_threshold = 1.0e-6)
	
	# So we need to load an image...
	img = Image.open("test_hf.png")
	# self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None
	# test_hf_reimplementation([img, 1, 1, 1, None, "cuda:1", 1], {}, hf_module, "prepare_latents", tg_module, "prepare_latents")
	test_hf_reimplementation([], {"prompt": "a fluffy bunny pokemon", "image": img, "num_inference_steps": 15, "safety_checker": None}, hf_module, "__call__", tg_module, "__call__")


@tinygrad.Tensor.test()
@tinygrad.Tensor.train(mode = False)
@torch.no_grad()
def main():
	test_stable_diffusion_img2img()
	input("pp")
	test_stable_diffusion_pipeline()
	#test_stable_diffusion_xl_pipeline()
	input("look at the outputs first you dumdum")
	test_clip_tokenizer_fast()
	test_clip_tokenizer()
	
	
	#input("anything beyond here will make me run out of memory :c")
	
	
	test_named_parameters()
	test_named_modules()
	test_autoencoderkl_from_single_file()
	
	test_interpolate()
	test_downsample2d()
	test_resnet_block2d()
	test_vae_encoder()
	test_down_encoder_block_2d()
	test_upsampling_2d()
	
	test_UNetMidBlock2D()
	test_vae_decoder()
	
if __name__ == "__main__":
	main()
