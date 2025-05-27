import torch
import diffusers
import tiny_hf
import tiny_hf as thf
import numpy as np
import os
import tinygrad

from tg_adapter import F
import tg_adapter

from operator_tests import test_cat, test_cumprod
from testing_utils import compare_state_dicts, copy_state_dict, inspect_state_dict_devices, make_test_data, test_function, test_hf_reimplementation, mse, norm_mse


def test_interpolate():
	shape = (2, 3, 6, 6)
	a = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
	args = (a, None, 2.0)
	test_function(*args, torch_function = torch.nn.functional.interpolate, tinygrad_function = F.interpolate)


	

def test_autoencoderkl():
	inp = np.random.randn(2*3*64*64).reshape(2, 3, 64, 64).astype(np.float32)
	args = (3,)
	
	# from single file is broken atm
	hf_module = diffusers.AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors" )
	my_module = thf.diffusers.models.autoencoders.AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors" )
	
	#copy_state_dict(hf_module, my_module)
	test_hf_reimplementation(inp, {}, hf_module, "__call__", my_module, "__call__")

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
	
	args = (a, 3, emb)
	test_hf_reimplementation(args, {}, hf_module, "__call__", thf_module, "__call__")

def test_unet_2d():
	from diffusers.models.unets.unet_2d import UNet2DModel as hf_class
	from tiny_hf.diffusers.models.unets.unet_2d import UNet2DModel as thf_class
	
	a = make_test_data(2, 3, 32, 32)
	
	hf_module = hf_class()
	thf_module = thf_class()
	
	args = (a, 2)
	copy_state_dict(hf_module, thf_module)
	test_hf_reimplementation(args, {}, hf_module, "__call__", thf_module, "__call__")

def test_named_modules():
	hf_module = diffusers.AutoencoderKL()
	my_module = thf.diffusers.models.autoencoders.AutoencoderKL()
	
	for name, module in hf_module.named_modules():
		print(name, module)

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
	from testing_rewrites import sd_pipeline_call
	from tiny_hf.diffusers.pipelines import StableDiffusionPipeline as tg_class
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	
	from diffusers.pipelines import StableDiffusionPipeline as hf_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	# ensure the scheduler is initialized properly
	_test_key_errors(hf_scheduler, tg_scheduler)
	
	
	
	hf_module = hf_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, scheduler = hf_scheduler, safety_checker = None)
	tg_module = tg_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, scheduler = tg_scheduler, safety_checker = None)
	
	# ensure there is no difference in state dict
	compare_state_dicts(hf_module.unet, tg_module.unet)
	
	# ensure all the weights and other thingies in the U-Net are the same
	#_test_key_errors(hf_module.unet, tg_module.unet)
	input("beep boop")
	
	
	# oh wait, i realized its impossible for them to have the same output image if the initial latents are not the same
	latents = make_test_data(1, 4, 64, 64)
	
	# test the unet
	# Ok, so the unet conditioner thingy is semi-broken, and I really don't know why...
	test_unet_2d_condition(hf_module.unet, tg_module.unet, latents.shape, (1, 77, 768) )
	
	# test prompt encoding
	test_hf_reimplementation(["a squishy pp", "cpu", 1, True], {}, hf_module, "encode_prompt", tg_module, "encode_prompt")
	
	# test the image processor
	test_hf_reimplementation([latents], {}, hf_module.image_processor, "postprocess", tg_module.image_processor, "postprocess")
	
	prepare_latents_test_args = [
		1,
		4,
		512,
		512,
		None, # this doesn't get checked if the latents are supplied
		"cpu",
		None,
	]
	test_hf_reimplementation(prepare_latents_test_args, {"latents": latents}, hf_module, "prepare_latents", tg_module, "prepare_latents")
	test_hf_reimplementation([], {"prompt": "a fluffy bunny", "num_inference_steps": 15, "safety_checker": None, "latents": latents, "output_type": "pil"}, hf_module, sd_pipeline_call, tg_module, sd_pipeline_call)

def test_ddim_scheduler():
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	hf_scheduler.set_timesteps(4)
	tg_scheduler.set_timesteps(4)
	
	_test_key_errors(hf_scheduler, tg_scheduler)
	
	latent = make_test_data(2, 4, 64, 64)
	noise = make_test_data(2, 4, 64, 64)
	
	test_hf_reimplementation([noise, 3, latent], {"return_dict": False}, hf_scheduler, "step", tg_scheduler, "step")

def test_dtype_override():
	a = tg_adapter.arange(4, device = "cpu", dtype = tg_adapter.int64)
	b = a.to("cuda:0")
	print(b.dtype, b.tg.dtype)
	c = b.to("cpu")
	print(c.numpy() )
	print(c.dtype, c.tg.dtype)

@tinygrad.Tensor.test()
@tinygrad.Tensor.train(mode = False)
@torch.no_grad()
def main():
	#test_dtype_override()
	#test_clip_text_model()
	#test_unet_2d()
	#test_unet_2d_condition()
	test_cat()
	test_ddim_scheduler()
	#test_autoencoderkl()
	test_cumprod()
	test_stable_diffusion_pipeline()
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
