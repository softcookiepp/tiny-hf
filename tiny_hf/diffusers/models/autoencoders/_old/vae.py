import tinygrad
import tg_adapter as tga
from typing import Dict, Optional, Tuple, Union
from ..unet.blocks import *#get_down_block

class Encoder(tga.nn.Module):
	r"""
	The `Encoder` layer of a variational autoencoder that encodes its input into a latent representation.

	Args:
		in_channels (`int`, *optional*, defaults to 3):
			The number of input channels.
		out_channels (`int`, *optional*, defaults to 3):
			The number of output channels.
		down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
			The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
			options.
		block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
			The number of output channels for each block.
		layers_per_block (`int`, *optional*, defaults to 2):
			The number of layers per block.
		norm_num_groups (`int`, *optional*, defaults to 32):
			The number of groups for normalization.
		act_fn (`str`, *optional*, defaults to `"silu"`):
			The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
		double_z (`bool`, *optional*, defaults to `True`):
			Whether to double the number of output channels for the last block.
	"""

	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 3,
		down_block_types: Tuple[str, ...] = ("DownEncoderBlock2D",),
		block_out_channels: Tuple[int, ...] = (64,),
		layers_per_block: int = 2,
		norm_num_groups: int = 32,
		act_fn: str = "silu",
		double_z: bool = True,
		mid_block_add_attention=True,
	):
		super().__init__()
		self.layers_per_block = layers_per_block

		#self.conv_in = nn.Conv2d(
		self.conv_in = tinygrad.nn.Conv2d(
			in_channels,
			block_out_channels[0],
			kernel_size=3,
			stride=1,
			padding=1,
		)

		#self.down_blocks = nn.ModuleList([])
		self.down_blocks = []

		# down
		output_channel = block_out_channels[0]
		for i, down_block_type in enumerate(down_block_types):
			input_channel = output_channel
			output_channel = block_out_channels[i]
			is_final_block = i == len(block_out_channels) - 1

			down_block = get_down_block(
				down_block_type,
				num_layers=self.layers_per_block,
				in_channels=input_channel,
				out_channels=output_channel,
				add_downsample=not is_final_block,
				resnet_eps=1e-6,
				downsample_padding=0,
				resnet_act_fn=act_fn,
				resnet_groups=norm_num_groups,
				attention_head_dim=output_channel,
				temb_channels=None,
			)
			self.down_blocks.append(down_block)

		# mid
		self.mid_block = UNetMidBlock2D(
			in_channels=block_out_channels[-1],
			resnet_eps=1e-6,
			resnet_act_fn=act_fn,
			output_scale_factor=1,
			resnet_time_scale_shift="default",
			attention_head_dim=block_out_channels[-1],
			resnet_groups=norm_num_groups,
			temb_channels=None,
			add_attention=mid_block_add_attention,
		)

		# out
		#self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
		self.conv_norm_out = tinygrad.nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
		#self.conv_act = nn.SiLU()
		self.conv_act = lambda x: x.silu()

		conv_out_channels = 2 * out_channels if double_z else out_channels
		#self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)
		self.conv_out = tinygrad.nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

		self.gradient_checkpointing = False

	def forward(self, sample: tinygrad.Tensor) -> tinygrad.Tensor:
		r"""The forward method of the `Encoder` class."""

		sample = self.conv_in(sample)

		#if torch.is_grad_enabled() and self.gradient_checkpointing:
		if self.gradient_checkpointing:
			raise NotImplementedError
			# this will be a shitshow lmao
			# down
			for down_block in self.down_blocks:
				sample = self._gradient_checkpointing_func(down_block, sample)
			# middle
			sample = self._gradient_checkpointing_func(self.mid_block, sample)

		else:
			# down
			for down_block in self.down_blocks:
				sample = down_block(sample)

			# middle
			sample = self.mid_block(sample)

		# post-process
		sample = self.conv_norm_out(sample)
		sample = self.conv_act(sample)
		sample = self.conv_out(sample)

		return sample
		
		
class Decoder(tga.nn.Module):
	r"""
	The `Decoder` layer of a variational autoencoder that decodes its latent representation into an output sample.

	Args:
		in_channels (`int`, *optional*, defaults to 3):
			The number of input channels.
		out_channels (`int`, *optional*, defaults to 3):
			The number of output channels.
		up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
			The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
		block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
			The number of output channels for each block.
		layers_per_block (`int`, *optional*, defaults to 2):
			The number of layers per block.
		norm_num_groups (`int`, *optional*, defaults to 32):
			The number of groups for normalization.
		act_fn (`str`, *optional*, defaults to `"silu"`):
			The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
		norm_type (`str`, *optional*, defaults to `"group"`):
			The normalization type to use. Can be either `"group"` or `"spatial"`.
	"""

	def __init__(
		self,
		in_channels: int = 3,
		out_channels: int = 3,
		up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
		block_out_channels: Tuple[int, ...] = (64,),
		layers_per_block: int = 2,
		norm_num_groups: int = 32,
		act_fn: str = "silu",
		norm_type: str = "group",  # group, spatial
		mid_block_add_attention=True,
	):
		super().__init__()
		self.layers_per_block = layers_per_block

		#self.conv_in = nn.Conv2d(
		self.conv_in = tinygrad.nn.Conv2d(
			in_channels,
			block_out_channels[-1],
			kernel_size=3,
			stride=1,
			padding=1,
		)

		self.up_blocks = []#nn.ModuleList([])

		temb_channels = in_channels if norm_type == "spatial" else None

		# mid
		self.mid_block = UNetMidBlock2D(
			in_channels=block_out_channels[-1],
			resnet_eps=1e-6,
			resnet_act_fn=act_fn,
			output_scale_factor=1,
			resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
			attention_head_dim=block_out_channels[-1],
			resnet_groups=norm_num_groups,
			temb_channels=temb_channels,
			add_attention=mid_block_add_attention,
		)

		# up
		reversed_block_out_channels = list(reversed(block_out_channels))
		output_channel = reversed_block_out_channels[0]
		for i, up_block_type in enumerate(up_block_types):
			prev_output_channel = output_channel
			output_channel = reversed_block_out_channels[i]

			is_final_block = i == len(block_out_channels) - 1

			up_block = get_up_block(
				up_block_type,
				num_layers=self.layers_per_block + 1,
				in_channels=prev_output_channel,
				out_channels=output_channel,
				prev_output_channel=None,
				add_upsample=not is_final_block,
				resnet_eps=1e-6,
				resnet_act_fn=act_fn,
				resnet_groups=norm_num_groups,
				attention_head_dim=output_channel,
				temb_channels=temb_channels,
				resnet_time_scale_shift=norm_type,
			)
			self.up_blocks.append(up_block)
			prev_output_channel = output_channel

		# out
		if norm_type == "spatial":
			self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
		else:
			#self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
			self.conv_norm_out = tinygrad.nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
		#self.conv_act = nn.SiLU()
		self.conv_act = lambda x: x.silu()
		#self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)
		self.conv_out = tinygrad.nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

		self.gradient_checkpointing = False

	def forward(
		self,
		sample: tinygrad.Tensor,
		latent_embeds: Optional[tinygrad.Tensor] = None,
	) -> tinygrad.Tensor:
		r"""The forward method of the `Decoder` class."""

		sample = self.conv_in(sample)
		
		#upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
		#if torch.is_grad_enabled() and self.gradient_checkpointing:
		if self.gradient_checkpointing:
			raise NotImplementedError
			# middle
			sample = self._gradient_checkpointing_func(self.mid_block, sample, latent_embeds)
			sample = sample.to(upscale_dtype)

			# up
			for up_block in self.up_blocks:
				sample = self._gradient_checkpointing_func(up_block, sample, latent_embeds)
		else:
			# middle
			sample = self.mid_block(sample, latent_embeds)
			# disabling lool
			# tinygrad should handle this stuff automatically pretty well
			#sample = sample.to(upscale_dtype)

			# up
			for up_block in self.up_blocks:
				sample = up_block(sample, latent_embeds)

		# post-process
		if latent_embeds is None:
			sample = self.conv_norm_out(sample)
		else:
			sample = self.conv_norm_out(sample, latent_embeds)
		sample = self.conv_act(sample)
		sample = self.conv_out(sample)

		return sample
