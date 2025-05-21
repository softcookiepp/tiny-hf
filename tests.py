import torch
import diffusers
import tiny_hf
import tiny_hf as thf
import numpy as np
import os
import tinygrad
from tinygrad.nn.state import torch_load, safe_load, load_state_dict, get_state_dict
import safetensors
from safetensors.torch import save_file
from PIL import Image

from tg_adapter import F
import tg_adapter

def compare_state_dicts(torch_module, tga_module, error_threshold = 1.0e-3):
	print(type(torch_module), type(tga_module) )
	torch_sd = torch_module.state_dict()
	try:
		tga_sd = tga_module.state_dict()
	except AttributeError:
		tga_sd = get_state_dict(tga_module)
	print(len(torch_sd), len(tga_sd) )
	try:
		assert len(tga_sd.keys() ) == len(torch_sd.keys() )
		for torch_key, tga_key in zip(sorted(torch_sd.keys() ), sorted(tga_sd.keys() ) ):
			#print(torch_key, tga_key)
			assert torch_key == tga_key.replace("._tg", "")
			key = torch_key
			torch_value = torch_sd[key].detach().numpy()
			tga_value = tga_sd[tga_key].numpy()
			assert mse(torch_value, tga_value) < error_threshold
	except AssertionError:
		# keys are not equal
		tga_sd_norm = []
		for k in tga_sd.keys():
			tga_sd_norm.append(k.replace("._tg", "") )
		
		# keys that tga module doesn't have
		missing_keys_tg = list(set(torch_sd.keys() ) - set(tga_sd_norm ) )

		missing_keys_torch = list( set(tga_sd_norm ) - set(torch_sd.keys() ) )
		
		print("keys missing from tga module:")
		print(missing_keys_tg)
		print(f"\n{len(missing_keys_torch)} keys missing from torch module:")
		print(missing_keys_torch)
		input()
		raise ValueError



def inspect_state_dict_devices(module):
	for k, v in module.state_dict().items():
		print(k, v.device)
	input("meep")

def copy_state_dict(torch_module, tga_module):
	# this should work
	d = os.path.abspath(".")
	if os.path.exists("/dev/shm"):
		d = "/dev/shm"
	fn = os.path.join(d, "tmp.safetensors")
	save_file(torch_module.state_dict(), fn)
	state_dict = safe_load(fn)
	#print(tinygrad.device.Device.default)
	#input(list(state_dict.items())[0][1].device)
	tga_module.load_state_dict(state_dict)
	#load_state_dict(tga_module, state_dict)
	compare_state_dicts(torch_module, tga_module)
	os.remove(fn)
	

def mse(predicted, actual):
	return np.sum( (predicted - actual)**2 )

def make_test_data(*shape):
	return np.random.randn(np.prod(shape) ).reshape(shape).astype(np.float32)


def test_function(inp_args, torch_function, tinygrad_function, error_threshold = 1.0e-4):
	torch_inp = []
	tiny_inp = []
	for inp in inp_args:
		if isinstance(inp, np.ndarray):
			torch_inp.append(torch.tensor(inp) )
			tiny_inp.append(tinygrad.Tensor(inp) )
		else:
			torch_inp.append(inp)
			tiny_inp.append(inp)
	torch_out = torch_function(*tuple(torch_inp) )
	tiny_out = tinygrad_function(*tuple(tiny_inp) )
	
	#print(tiny_out.shape, torch_out.shape )
	error = mse(tiny_out.numpy(), torch_out.numpy())
	print(f"MSE for {torch_function.__name__} and {tinygrad_function.__name__}:",  error)
	if error > error_threshold:
		
		print(torch_out.numpy() )
		print( tiny_out.numpy() )
		print(torch_out.shape, tiny_out.shape)
		input()

def test_interpolate():
	shape = (2, 3, 6, 6)
	a = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
	args = (a, None, 2.0)
	test_function(*args, torch_function = torch.nn.functional.interpolate, tinygrad_function = F.interpolate)

def _print_dict_types(d):
	for k, v in d.items():
		print(k, type(v), v)

def _get_output(hf_out):
	for k in ["sample"]:
		try:
			return hf_out[k]
		except KeyError:
			pass
	#_print_dict_types(hf_out)
	raise ValueError
	
def _test_key_errors(hf_dict, tg_dict, error_threshold = 1.0e-4, print_values = True, display_images = False):
	print("types:", type(hf_dict), type(tg_dict) )
	if isinstance(hf_dict, dict):
		tested_keys = hf_dict.keys()
		if list(hf_dict.keys() ) != list(tg_dict.keys() ):
			print("Warning! keys don't match! Will only test the ones that do")
			print(hf_dict.keys() )
			print(tg_dict.keys() )
			if len(hf_dict.keys() ) > len(tg_dict.keys() ):
				tested_keys = tg_dict.keys()
		for k in tested_keys:
			hf_item = hf_dict[k]
			tg_item = tg_dict[k]
			
			if isinstance(hf_item, list):
				try:
					hf_item = np.array(hf_item).astype(np.float32)
				except TypeError:
					# list of other sort, non-numerical
					for hf_item2, tg_item2 in zip(hf_item, tg_item):
						_test_key_errors(hf_item2, tg_item2, error_threshold, display_images)
					continue
			elif isinstance(hf_item, Image.Image):
				input("gots us an image!")
				hf_item, tg_item = np.array(hf_item), np.array(tg_item)
			elif isinstance(hf_item, torch.Tensor):
				hf_item = hf_item.detach().numpy()
			elif hasattr(hf_item, "__dict__"):
				_test_key_errors(hf_item.__dict__, tg_item.__dict__, error_threshold, display_images)
				continue
			elif isinstance(hf_item, dict):
				_test_key_errors(hf_item, tg_item, error_threshold, display_images)
				continue
			elif hf_item is None and tg_item is None:
				continue
			else:
				#print(hf_item)
				hf_item, tg_item = float(hf_item), float(tg_item)
				#raise ValueError
				
			if isinstance(tg_item, list):
				tg_item = np.array(tg_item).astype(np.float32)
			elif isinstance(tg_item, tinygrad.Tensor) or isinstance(tg_item, tg_adapter.Tensor):
				tg_item = tg_item.numpy()
			
			val_mse = mse(tg_item, hf_item)
			print("key:", k, "\nvalue mse:", val_mse, "\n")
			if val_mse > error_threshold or np.isnan(val_mse):
				if print_values:
					print(hf_item)
					print(tg_item)
				input()
			elif display_images:
				pass
				#raise NotImplementedError
	elif isinstance(hf_dict, torch.Tensor):
		error = mse(tg_dict.numpy(), hf_dict.detach().numpy())
		print("single tensor output mse:", error, "\n")
		if error > error_threshold:
			#print("tiny:")
			#print(tiny_out.numpy())
			#print("torch")
			#print(torch_out.detach().numpy() )
			#print("difference:")
			#print(tiny_out.numpy() - torch_out.detach().numpy())
			input()
	elif isinstance(hf_dict, object) and hasattr(hf_dict, "__dict__"):
		_test_key_errors(hf_dict.__dict__, tg_dict.__dict__, error_threshold, display_images)
		
def _process_arg(arg, device):
	if isinstance(arg, np.ndarray):
		# convert to tensor
		tgt = tg_adapter.tensor(arg).to(device)
		tgt.tg.realize()
		return torch.tensor(arg), tgt
	else:
		# append as is
		return arg, arg

def test_hf_reimplementation(args, kwargs, hf_module, hf_method, my_module, my_method, error_threshold = 1.0e-4, device = "cuda:0", display_images = False):
	if not (isinstance(args, tuple) or isinstance(args, list) ):
		args = (args,)
	if hasattr(my_module, "to"):
		my_module = my_module.to(device)
	hf_args, my_args = [], []
	for arg in args:
		if isinstance(arg, np.ndarray):
			# convert to tensor
			torch_v, tg_v = _process_arg(arg, device)
			hf_args.append(torch_v)
			my_args.append(tg_v )
		else:
			# append as is
			hf_args.append(arg )
			my_args.append(arg )
	
	hf_args = tuple(hf_args)
	my_args = tuple(my_args)
	
	hf_kwargs = {}
	my_kwargs = {}
	for k, v in kwargs.items():
		torch_v, tg_v = _process_arg(v, device)
		hf_kwargs[k] = torch_v
		my_kwargs[k] = tg_v
	
	
	
	# compute tiny out first so we don't have to wait for torch
	tiny_out = my_module.__getattribute__(my_method)(*my_args, **my_kwargs)
	torch_out = hf_module.__getattribute__(hf_method)(*hf_args, **hf_kwargs)
	
	"""
	if isinstance(torch_out, tuple):
		torch_out = torch_out[0]
	elif not isinstance(torch_out, torch.Tensor):
		torch_out = _get_output(torch_out)
	"""
	
	
	
	#inspect_state_dict_devices(my_module)
	print(f"MSE for {hf_module} and {my_module}:")
	_test_key_errors(torch_out, tiny_out, display_images = display_images)
	
	del torch_out
	del tiny_out
	del hf_module
	del my_module
	

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

def test_unet_2d_condition():
	from diffusers import UNet2DConditionModel as hf_UNet2DConditionModel
	from tiny_hf.diffusers import UNet2DConditionModel as thf_UNet2DConditionModel
	
	a = make_test_data(2, 4, 32, 32)
	emb = make_test_data(2, 2, 1280)
	
	hf_module = hf_UNet2DConditionModel()
	thf_module = thf_UNet2DConditionModel()
	
	args = (a, 3, emb)
	copy_state_dict(hf_module, thf_module)
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

def test_cumprod():
	from tg_adapter import F as tinyF
	inp = np.arange(5*4).reshape(5, 4).astype(np.float32)
	test_function( (inp, 0), torch.cumprod, tinyF.cumprod)

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
	from tiny_hf.diffusers.pipelines import StableDiffusionPipeline as tg_class
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	
	from diffusers.pipelines import StableDiffusionPipeline as hf_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	hf_module = hf_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, scheduler = hf_scheduler)
	tg_module = tg_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, scheduler = tg_scheduler)
	
	test_hf_reimplementation([], {"prompt": "a fluffy bunny", "num_inference_steps": 1, "safety_checker": None, "output_type": "latent"}, hf_module, "__call__", tg_module, "__call__")

def test_stable_diffusion_pipeline_manual():
	# The from_pretrained method is broke, so lets just do it manually holy shit
	from tiny_hf.diffusers.pipelines import StableDiffusionPipeline as tg_class
	from tiny_hf.diffusers.schedulers import DDIMScheduler as tg_scheduler_class
	
	from diffusers.pipelines import StableDiffusionPipeline as hf_class
	from diffusers.schedulers import DDIMScheduler as hf_scheduler_class
	
	hf_scheduler = hf_scheduler_class()
	tg_scheduler = tg_scheduler_class()
	
	hf_module = hf_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, scheduler = hf_scheduler)
	tg_module = tg_class.from_pretrained("stablediffusionapi/anything-v5", use_safetensors = True, requires_safety_checker = False, scheduler = tg_scheduler)
	
	test_hf_reimplementation([], {"prompt": "a fluffy bunny", "num_inference_steps": 2, "safety_checker": None}, hf_module, "__call__", tg_module, "__call__")


def test_ddim_scheduler():
	raise NotImplementedError

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
	#test_ddim_scheduler()
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
