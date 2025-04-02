import tg_adapter as tga
import tinygrad
from tg_adapter import F
from .attention_processor import SpatialNorm
from typing import Optional
from .activations import get_activation

from .normalization import RMSNorm, AdaGroupNorm
from .upsampling import Upsample2D

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

class Upsample1D(tga.nn.Module):
	"""
	An upsampling layer with an optional convolution.

	Parameters:
			channels: channels in the inputs and outputs.
			use_conv: a bool determining if a convolution is applied.
			use_conv_transpose:
			out_channels:
	"""

	def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.use_conv_transpose = use_conv_transpose
		self.name = name

		self.conv = None
		if use_conv_transpose:
			#self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
			self.conv = tinygrad.nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
		elif use_conv:
			#self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)
			self.conv = tinygrad.nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

	def forward(self, x):
		assert x.shape[1] == self.channels
		if self.use_conv_transpose:
			return self.conv(x)

		x = F.interpolate(x, scale_factor=2.0, mode="nearest")

		if self.use_conv:
			x = self.conv(x)

		return x
		


class Downsample1D(tga.nn.Module):
	"""
	A downsampling layer with an optional convolution.

	Parameters:
		channels: channels in the inputs and outputs.
		use_conv: a bool determining if a convolution is applied.
		out_channels:
		padding:
	"""

	def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
		super().__init__()
		self.channels = channels
		self.out_channels = out_channels or channels
		self.use_conv = use_conv
		self.padding = padding
		stride = 2
		self.name = name

		if use_conv:
			self.conv = tga.nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
		else:
			assert self.channels == self.out_channels
			self.conv = tga.nn.AvgPool1d(kernel_size=stride, stride=stride)

	def forward(self, x):
		assert x.shape[1] == self.channels
		return self.conv(x)


class ResnetBlock2D(tga.nn.Module):
	def __init__(self,
		*,
		in_channels,
		out_channels=None,
		conv_shortcut=False,
		dropout=0.0,
		temb_channels=512,
		groups=32,
		groups_out=None,
		pre_norm=True,
		eps=1e-6,
		non_linearity="swish",
		skip_time_act=False,
		time_embedding_norm="default",  # default, scale_shift, ada_group
		kernel=None,
		output_scale_factor=1.0,
		use_in_shortcut=None,
		up=False,
		down=False,
		conv_shortcut_bias: bool = True,
		conv_2d_out_channels: Optional[int] = None,
	):
		super().__init__()
		self.pre_norm = pre_norm
		self.pre_norm = True
		self.in_channels = in_channels
		out_channels = in_channels if out_channels is None else out_channels
		self.out_channels = out_channels
		self.use_conv_shortcut = conv_shortcut
		self.up = up
		self.down = down
		self.output_scale_factor = output_scale_factor
		self.time_embedding_norm = time_embedding_norm
		self.skip_time_act = skip_time_act
		
		if groups_out is None:
			groups_out = groups

		if self.time_embedding_norm == "ada_group":
			#self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
			raise NotImplementedError
		else:
			#self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
			self.norm1 = tinygrad.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

		#self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.conv1 = tinygrad.nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)
		
		if temb_channels is not None:
			if self.time_embedding_norm == "default":
				#self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
				self.time_emb_proj = tinygrad.nn.Linear(temb_channels, out_channels)
			elif self.time_embedding_norm == "scale_shift":
				#self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
				self.time_emb_proj = tinygrad.nn.Linear(temb_channels, 2 * out_channels)
			elif self.time_embedding_norm == "ada_group":
				self.time_emb_proj = None
			else:
				raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
		else:
			self.time_emb_proj = None
			
		
		if self.time_embedding_norm == "ada_group":
			#self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
			raise NotImplementedError
		else:
			#self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
			self.norm2 = tinygrad.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

		#self.dropout = torch.nn.Dropout(dropout)
		self.dropout = tga.nn.Dropout(dropout)
		conv_2d_out_channels = conv_2d_out_channels or out_channels
		#self.conv2 = torch.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)
		self.conv2 = tinygrad.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

		#if non_linearity == "swish":
		if non_linearity == "swish" or non_linearity == "silu":
			#self.nonlinearity = lambda x: F.silu(x)
			self.nonlinearity = lambda x: x.silu()
		elif non_linearity == "mish":
			#self.nonlinearity = nn.Mish()
			self.nonlinearity = lambda x: x.mish()
		#elif non_linearity == "silu":
		#	self.nonlinearity = nn.SiLU()
		elif non_linearity == "gelu":
			#self.nonlinearity = nn.GELU()
			self.nonlinearity = lambda x: x.gelu()

		self.upsample = self.downsample = None
		if self.up:
			if kernel == "fir":
				raise NotImplementedError
				fir_kernel = (1, 3, 3, 1)
				self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
			elif kernel == "sde_vp":
				raise NotImplementedError
				self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
			else:
				raise NotImplementedError
				self.upsample = Upsample2D(in_channels, use_conv=False)
		elif self.down:
			if kernel == "fir":
				raise NotImplementedError
				fir_kernel = (1, 3, 3, 1)
				self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
			elif kernel == "sde_vp":
				raise NotImplementedError
				self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
			else:
				self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

		self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

		self.conv_shortcut = None
		if self.use_in_shortcut:
			#self.conv_shortcut = torch.nn.Conv2d(
			self.conv_shortcut = tinygrad.nn.Conv2d(
				in_channels, conv_2d_out_channels, 	kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
			)
	
	def forward(self, input_tensor, temb):
		hidden_states = input_tensor

		if self.time_embedding_norm == "ada_group":
			hidden_states = self.norm1(hidden_states, temb)
		else:
			hidden_states = self.norm1(hidden_states)

		hidden_states = self.nonlinearity(hidden_states)

		if self.upsample is not None:
			# upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
			if hidden_states.shape[0] >= 64:
				input_tensor = input_tensor.contiguous()
				hidden_states = hidden_states.contiguous()
			input_tensor = self.upsample(input_tensor)
			hidden_states = self.upsample(hidden_states)
		elif self.downsample is not None:
			input_tensor = self.downsample(input_tensor)
			hidden_states = self.downsample(hidden_states)

		hidden_states = self.conv1(hidden_states)

		if self.time_emb_proj is not None:
			if not self.skip_time_act:
				temb = self.nonlinearity(temb)
			temb = self.time_emb_proj(temb)[:, :, None, None]

		if temb is not None and self.time_embedding_norm == "default":
			hidden_states = hidden_states + temb

		if self.time_embedding_norm == "ada_group":
			hidden_states = self.norm2(hidden_states, temb)
		else:
			hidden_states = self.norm2(hidden_states)

		if temb is not None and self.time_embedding_norm == "scale_shift":
			#scale, shift = torch.chunk(temb, 2, dim=1)
			scale, shift = tebm.chunk(2, dim=1)
			hidden_states = hidden_states * (1 + scale) + shift

		hidden_states = self.nonlinearity(hidden_states)

		hidden_states = self.dropout(hidden_states)
		hidden_states = self.conv2(hidden_states)

		if self.conv_shortcut is not None:
			input_tensor = self.conv_shortcut(input_tensor)

		output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

		return output_tensor
		
class ResnetBlockCondNorm2D(tga.nn.Module):
	r"""
	A Resnet block that use normalization layer that incorporate conditioning information.

	Parameters:
		in_channels (`int`): The number of channels in the input.
		out_channels (`int`, *optional*, default to be `None`):
			The number of output channels for the first conv2d layer. If None, same as `in_channels`.
		dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
		temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
		groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
		groups_out (`int`, *optional*, default to None):
			The number of groups to use for the second normalization layer. if set to None, same as `groups`.
		eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
		non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
		time_embedding_norm (`str`, *optional*, default to `"ada_group"` ):
			The normalization layer for time embedding `temb`. Currently only support "ada_group" or "spatial".
		kernel (`tinygrad.Tensor`, optional, default to None): FIR filter, see
			[`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
		output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
		use_in_shortcut (`bool`, *optional*, default to `True`):
			If `True`, add a 1x1 nn.conv2d layer for skip-connection.
		up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
		down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
		conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
			`conv_shortcut` output.
		conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
			If None, same as `out_channels`.
	"""

	def __init__(
		self,
		*,
		in_channels: int,
		out_channels: Optional[int] = None,
		conv_shortcut: bool = False,
		dropout: float = 0.0,
		temb_channels: int = 512,
		groups: int = 32,
		groups_out: Optional[int] = None,
		eps: float = 1e-6,
		non_linearity: str = "swish",
		time_embedding_norm: str = "ada_group",  # ada_group, spatial
		output_scale_factor: float = 1.0,
		use_in_shortcut: Optional[bool] = None,
		up: bool = False,
		down: bool = False,
		conv_shortcut_bias: bool = True,
		conv_2d_out_channels: Optional[int] = None,
	):
		super().__init__()
		self.in_channels = in_channels
		out_channels = in_channels if out_channels is None else out_channels
		self.out_channels = out_channels
		self.use_conv_shortcut = conv_shortcut
		self.up = up
		self.down = down
		self.output_scale_factor = output_scale_factor
		self.time_embedding_norm = time_embedding_norm

		if groups_out is None:
			groups_out = groups

		if self.time_embedding_norm == "ada_group":  # ada_group
			self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
		elif self.time_embedding_norm == "spatial":
			self.norm1 = SpatialNorm(in_channels, temb_channels)
		else:
			raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

		self.conv1 = tinygrad.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

		if self.time_embedding_norm == "ada_group":  # ada_group
			self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
		elif self.time_embedding_norm == "spatial":  # spatial
			self.norm2 = SpatialNorm(out_channels, temb_channels)
		else:
			raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

		self.dropout = tga.nn.Dropout(dropout)

		conv_2d_out_channels = conv_2d_out_channels or out_channels
		self.conv2 = tinygrad.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

		self.nonlinearity = get_activation(non_linearity)

		self.upsample = self.downsample = None
		if self.up:
			self.upsample = Upsample2D(in_channels, use_conv=False)
		elif self.down:
			self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

		self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

		self.conv_shortcut = None
		if self.use_in_shortcut:
			self.conv_shortcut = nn.Conv2d(
				in_channels,
				conv_2d_out_channels,
				kernel_size=1,
				stride=1,
				padding=0,
				bias=conv_shortcut_bias,
			)

	def forward(self, input_tensor: tinygrad.Tensor, temb: tinygrad.Tensor, *args, **kwargs) -> tinygrad.Tensor:
		if len(args) > 0 or kwargs.get("scale", None) is not None:
			deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
			deprecate("scale", "1.0.0", deprecation_message)

		hidden_states = input_tensor

		hidden_states = self.norm1(hidden_states, temb)

		hidden_states = self.nonlinearity(hidden_states)

		if self.upsample is not None:
			# upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
			if hidden_states.shape[0] >= 64:
				input_tensor = input_tensor.contiguous()
				hidden_states = hidden_states.contiguous()
			input_tensor = self.upsample(input_tensor)
			hidden_states = self.upsample(hidden_states)

		elif self.downsample is not None:
			input_tensor = self.downsample(input_tensor)
			hidden_states = self.downsample(hidden_states)

		hidden_states = self.conv1(hidden_states)

		hidden_states = self.norm2(hidden_states, temb)

		hidden_states = self.nonlinearity(hidden_states)

		hidden_states = self.dropout(hidden_states)
		hidden_states = self.conv2(hidden_states)

		if self.conv_shortcut is not None:
			input_tensor = self.conv_shortcut(input_tensor)

		output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

		return output_tensor

