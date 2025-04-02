import tinygrad
import tg_adapter as tga
from .normalization import RMSNorm
from typing import Optional

class Downsample2D(tga.nn.Module):
	"""A 2D downsampling layer with an optional convolution.

	Parameters:
		channels (`int`):
			number of channels in the inputs and outputs.
		use_conv (`bool`, default `False`):
			option to use a convolution.
		out_channels (`int`, optional):
			number of output channels. Defaults to `channels`.
		padding (`int`, default `1`):
			padding for the convolution.
		name (`str`, default `conv`):
			name of the downsampling 2D layer.
	"""

	def __init__(
		self,
		channels: int,
		use_conv: bool = False,
		out_channels: Optional[int] = None,
		padding: int = 1,
		name: str = "conv",
		kernel_size=3,
		norm_type=None,
		eps=None,
		elementwise_affine=None,
		bias=True,
	):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.padding = padding
		stride = 2
		self.name = name

		if norm_type == "ln_norm":
			#self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
			# check arguments later
			self.norm = tinygrad.nn.LayerNorm2d(channels, eps, elementwise_affine)
		elif norm_type == "rms_norm":
			self.norm = RMSNorm(channels, eps, elementwise_affine)
		elif norm_type is None:
			self.norm = None
		else:
			raise ValueError(f"unknown norm_type: {norm_type}")

		if use_conv:
			#conv = nn.Conv2d(
			conv = tinygrad.nn.Conv2d(
				self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
			)
		else:
			assert self.channels == self.out_channels
			#conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
			conv = tga.nn.AvgPool2d(kernel_size=stride, stride=stride)

		# TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
		if name == "conv":
			self.Conv2d_0 = conv
			self.conv = conv
		elif name == "Conv2d_0":
			self.conv = conv
		else:
			self.conv = conv

	def forward(self, hidden_states: tinygrad.Tensor, *args, **kwargs) -> tinygrad.Tensor:
		if len(args) > 0 or kwargs.get("scale", None) is not None:
			#deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
			#deprecate("scale", "1.0.0", deprecation_message)
			# YOU CAN'T TELL ME WHAT TO DO MOM
			pass
		assert hidden_states.shape[1] == self.channels

		if self.norm is not None:
			hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

		if self.use_conv and self.padding == 0:
			pad = (0, 1, 0, 1)
			#hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
			hidden_states = hidden_states.pad(pad, mode="constant", value=0)

		assert hidden_states.shape[1] == self.channels

		hidden_states = self.conv(hidden_states)

		return hidden_states
