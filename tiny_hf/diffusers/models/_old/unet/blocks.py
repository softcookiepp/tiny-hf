import tg_adapter as tga
from ..resnet import *
from ..attention_processor import Attention
from ..upsampling import Upsample2D



class UNetMidBlock2D(tga.nn.Module):
	"""
	A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

	Args:
		in_channels (`int`): The number of input channels.
		temb_channels (`int`): The number of temporal embedding channels.
		dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
		num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
		resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
		resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
			The type of normalization to apply to the time embeddings. This can help to improve the performance of the
			model on tasks with long-range temporal dependencies.
		resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
		resnet_groups (`int`, *optional*, defaults to 32):
			The number of groups to use in the group normalization layers of the resnet blocks.
		attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
		resnet_pre_norm (`bool`, *optional*, defaults to `True`):
			Whether to use pre-normalization for the resnet blocks.
		add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
		attention_head_dim (`int`, *optional*, defaults to 1):
			Dimension of a single attention head. The number of attention heads is determined based on this value and
			the number of input channels.
		output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

	Returns:
		`tinygrad.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size, in_channels,
		height, width)`.

	"""

	def __init__(
		self,
		in_channels: int,
		temb_channels: int,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_eps: float = 1e-6,
		resnet_time_scale_shift: str = "default",  # default, spatial
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		attn_groups: Optional[int] = None,
		resnet_pre_norm: bool = True,
		add_attention: bool = True,
		attention_head_dim: int = 1,
		output_scale_factor: float = 1.0,
	):
		super().__init__()
		resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
		self.add_attention = add_attention

		if attn_groups is None:
			attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

		# there is always at least one resnet
		if resnet_time_scale_shift == "spatial":
			resnets = [
				ResnetBlockCondNorm2D(
					in_channels=in_channels,
					out_channels=in_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm="spatial",
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
				)
			]
		else:
			resnets = [
				ResnetBlock2D(
					in_channels=in_channels,
					out_channels=in_channels,
					temb_channels=temb_channels,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
				)
			]
		attentions = []

		if attention_head_dim is None:
			logger.warning(
				f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
			)
			attention_head_dim = in_channels

		for _ in range(num_layers):
			if self.add_attention:
				attentions.append(
					Attention(
						in_channels,
						heads=in_channels // attention_head_dim,
						dim_head=attention_head_dim,
						rescale_output_factor=output_scale_factor,
						eps=resnet_eps,
						norm_num_groups=attn_groups,
						spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
						residual_connection=True,
						bias=True,
						upcast_softmax=True,
						_from_deprecated_attn_block=True,
					)
				)
			else:
				attentions.append(None)

			if resnet_time_scale_shift == "spatial":
				raise NotImplementedError
				resnets.append(
					ResnetBlockCondNorm2D(
						in_channels=in_channels,
						out_channels=in_channels,
						temb_channels=temb_channels,
						eps=resnet_eps,
						groups=resnet_groups,
						dropout=dropout,
						time_embedding_norm="spatial",
						non_linearity=resnet_act_fn,
						output_scale_factor=output_scale_factor,
					)
				)
			else:
				resnets.append(
					ResnetBlock2D(
						in_channels=in_channels,
						out_channels=in_channels,
						temb_channels=temb_channels,
						eps=resnet_eps,
						groups=resnet_groups,
						dropout=dropout,
						time_embedding_norm=resnet_time_scale_shift,
						non_linearity=resnet_act_fn,
						output_scale_factor=output_scale_factor,
						pre_norm=resnet_pre_norm,
					)
				)

		#self.attentions = nn.ModuleList(attentions)
		self.attentions = attentions
		#self.resnets = nn.ModuleList(resnets)
		self.resnets = resnets

		self.gradient_checkpointing = False

	def forward(self, hidden_states: tinygrad.Tensor, temb: Optional[tinygrad.Tensor] = None) -> tinygrad.Tensor:
		hidden_states = self.resnets[0](hidden_states, temb)
		for attn, resnet in zip(self.attentions, self.resnets[1:]):
			#if torch.is_grad_enabled() and self.gradient_checkpointing:
			if self.gradient_checkpointing:
				raise NotImplementedError
				if attn is not None:
					hidden_states = attn(hidden_states, temb=temb)
				hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
			else:
				if attn is not None:
					hidden_states = attn(hidden_states, temb=temb)
				hidden_states = resnet(hidden_states, temb)

		return hidden_states

class UpDecoderBlock2D(tga.nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		resolution_idx: Optional[int] = None,
		dropout: float = 0.0,
		num_layers: int = 1,
		resnet_eps: float = 1e-6,
		resnet_time_scale_shift: str = "default",  # default, spatial
		resnet_act_fn: str = "swish",
		resnet_groups: int = 32,
		resnet_pre_norm: bool = True,
		output_scale_factor: float = 1.0,
		add_upsample: bool = True,
		temb_channels: Optional[int] = None,
	):
		super().__init__()
		resnets = []

		for i in range(num_layers):
			input_channels = in_channels if i == 0 else out_channels

			if resnet_time_scale_shift == "spatial":
				resnets.append(
					ResnetBlockCondNorm2D(
						in_channels=input_channels,
						out_channels=out_channels,
						temb_channels=temb_channels,
						eps=resnet_eps,
						groups=resnet_groups,
						dropout=dropout,
						time_embedding_norm="spatial",
						non_linearity=resnet_act_fn,
						output_scale_factor=output_scale_factor,
					)
				)
			else:
				resnets.append(
					ResnetBlock2D(
						in_channels=input_channels,
						out_channels=out_channels,
						temb_channels=temb_channels,
						eps=resnet_eps,
						groups=resnet_groups,
						dropout=dropout,
						time_embedding_norm=resnet_time_scale_shift,
						non_linearity=resnet_act_fn,
						output_scale_factor=output_scale_factor,
						pre_norm=resnet_pre_norm,
					)
				)

		self.resnets = resnets

		if add_upsample:
			self.upsamplers = [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
		else:
			self.upsamplers = None

		self.resolution_idx = resolution_idx

	def forward(self, hidden_states: tinygrad.Tensor, temb: Optional[tinygrad.Tensor] = None) -> tinygrad.Tensor:
		for resnet in self.resnets:
			hidden_states = resnet(hidden_states, temb=temb)

		if self.upsamplers is not None:
			for upsampler in self.upsamplers:
				hidden_states = upsampler(hidden_states)

		return hidden_states



class DownEncoderBlock2D(tga.nn.Module):
	def __init__( self,
				in_channels: int,
				out_channels: int,
				dropout: float = 0.0,
				num_layers: int = 1,
				resnet_eps: float = 1e-6,
				resnet_time_scale_shift: str = "default",
				resnet_act_fn: str = "swish",
				resnet_groups: int = 32,
				resnet_pre_norm: bool = True,
				output_scale_factor=1.0,
				add_downsample=True,
				downsample_padding=1,
			):
		super().__init__()
		resnets = []
		
		for i in range(num_layers):
			in_channels = in_channels if i == 0 else out_channels
			resnets.append(
				ResnetBlock2D(
					in_channels=in_channels,
					out_channels=out_channels,
					temb_channels=None,
					eps=resnet_eps,
					groups=resnet_groups,
					dropout=dropout,
					time_embedding_norm=resnet_time_scale_shift,
					non_linearity=resnet_act_fn,
					output_scale_factor=output_scale_factor,
					pre_norm=resnet_pre_norm,
				)
			)
		
		# tinygrad doesn't need this silly stuff
		#self.resnets = nn.ModuleList(resnets)
		self.resnets = resnets

		if add_downsample:
			"""
			self.downsamplers = nn.ModuleList(
				[
					Downsample2D(
						out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
					)
				]
			)
			"""
			self.downsamplers = [
				Downsample2D(
					out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
				)
			]
			
		else:
			self.downsamplers = None

	def forward(self, hidden_states):
		for resnet in self.resnets:
			hidden_states = resnet(hidden_states, temb=None)

		if self.downsamplers is not None:
			for downsampler in self.downsamplers:
				hidden_states = downsampler(hidden_states)

		return hidden_states

def get_up_block(
	up_block_type: str,
	num_layers: int,
	in_channels: int,
	out_channels: int,
	prev_output_channel: int,
	temb_channels: int,
	add_upsample: bool,
	resnet_eps: float,
	resnet_act_fn: str,
	resolution_idx: Optional[int] = None,
	transformer_layers_per_block: int = 1,
	num_attention_heads: Optional[int] = None,
	resnet_groups: Optional[int] = None,
	cross_attention_dim: Optional[int] = None,
	dual_cross_attention: bool = False,
	use_linear_projection: bool = False,
	only_cross_attention: bool = False,
	upcast_attention: bool = False,
	resnet_time_scale_shift: str = "default",
	attention_type: str = "default",
	resnet_skip_time_act: bool = False,
	resnet_out_scale_factor: float = 1.0,
	cross_attention_norm: Optional[str] = None,
	attention_head_dim: Optional[int] = None,
	upsample_type: Optional[str] = None,
	dropout: float = 0.0,
) -> tga.nn.Module:
	# If attn head dim is not defined, we default it to the number of heads
	if attention_head_dim is None:
		logger.warning(
			f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
		)
		attention_head_dim = num_attention_heads

	up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
	if up_block_type == "UpBlock2D":
		return UpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif up_block_type == "ResnetUpsampleBlock2D":
		return ResnetUpsampleBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			resnet_time_scale_shift=resnet_time_scale_shift,
			skip_time_act=resnet_skip_time_act,
			output_scale_factor=resnet_out_scale_factor,
		)
	elif up_block_type == "CrossAttnUpBlock2D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
		return CrossAttnUpBlock2D(
			num_layers=num_layers,
			transformer_layers_per_block=transformer_layers_per_block,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			cross_attention_dim=cross_attention_dim,
			num_attention_heads=num_attention_heads,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
			resnet_time_scale_shift=resnet_time_scale_shift,
			attention_type=attention_type,
		)
	elif up_block_type == "SimpleCrossAttnUpBlock2D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D")
		return SimpleCrossAttnUpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			cross_attention_dim=cross_attention_dim,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
			skip_time_act=resnet_skip_time_act,
			output_scale_factor=resnet_out_scale_factor,
			only_cross_attention=only_cross_attention,
			cross_attention_norm=cross_attention_norm,
		)
	elif up_block_type == "AttnUpBlock2D":
		if add_upsample is False:
			upsample_type = None
		else:
			upsample_type = upsample_type or "conv"  # default to 'conv'

		return AttnUpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
			upsample_type=upsample_type,
		)
	elif up_block_type == "SkipUpBlock2D":
		return SkipUpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif up_block_type == "AttnSkipUpBlock2D":
		return AttnSkipUpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			prev_output_channel=prev_output_channel,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif up_block_type == "UpDecoderBlock2D":
		return UpDecoderBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			resnet_time_scale_shift=resnet_time_scale_shift,
			temb_channels=temb_channels,
		)
	elif up_block_type == "AttnUpDecoderBlock2D":
		return AttnUpDecoderBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
			temb_channels=temb_channels,
		)
	elif up_block_type == "KUpBlock2D":
		return KUpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
		)
	elif up_block_type == "KCrossAttnUpBlock2D":
		return KCrossAttnUpBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			resolution_idx=resolution_idx,
			dropout=dropout,
			add_upsample=add_upsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			cross_attention_dim=cross_attention_dim,
			attention_head_dim=attention_head_dim,
		)

	raise ValueError(f"{up_block_type} does not exist.")

def get_down_block(
	down_block_type: str,
	num_layers: int,
	in_channels: int,
	out_channels: int,
	temb_channels: int,
	add_downsample: bool,
	resnet_eps: float,
	resnet_act_fn: str,
	transformer_layers_per_block: int = 1,
	num_attention_heads: Optional[int] = None,
	resnet_groups: Optional[int] = None,
	cross_attention_dim: Optional[int] = None,
	downsample_padding: Optional[int] = None,
	dual_cross_attention: bool = False,
	use_linear_projection: bool = False,
	only_cross_attention: bool = False,
	upcast_attention: bool = False,
	resnet_time_scale_shift: str = "default",
	attention_type: str = "default",
	resnet_skip_time_act: bool = False,
	resnet_out_scale_factor: float = 1.0,
	cross_attention_norm: Optional[str] = None,
	attention_head_dim: Optional[int] = None,
	downsample_type: Optional[str] = None,
	dropout: float = 0.0,
):
	# If attn head dim is not defined, we default it to the number of heads
	if attention_head_dim is None:
		logger.warning(
			f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
		)
		attention_head_dim = num_attention_heads

	down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
	if down_block_type == "DownBlock2D":
		return DownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif down_block_type == "ResnetDownsampleBlock2D":
		return ResnetDownsampleBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			resnet_time_scale_shift=resnet_time_scale_shift,
			skip_time_act=resnet_skip_time_act,
			output_scale_factor=resnet_out_scale_factor,
		)
	elif down_block_type == "AttnDownBlock2D":
		if add_downsample is False:
			downsample_type = None
		else:
			downsample_type = downsample_type or "conv"  # default to 'conv'
		return AttnDownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
			downsample_type=downsample_type,
		)
	elif down_block_type == "CrossAttnDownBlock2D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
		return CrossAttnDownBlock2D(
			num_layers=num_layers,
			transformer_layers_per_block=transformer_layers_per_block,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			cross_attention_dim=cross_attention_dim,
			num_attention_heads=num_attention_heads,
			dual_cross_attention=dual_cross_attention,
			use_linear_projection=use_linear_projection,
			only_cross_attention=only_cross_attention,
			upcast_attention=upcast_attention,
			resnet_time_scale_shift=resnet_time_scale_shift,
			attention_type=attention_type,
		)
	elif down_block_type == "SimpleCrossAttnDownBlock2D":
		if cross_attention_dim is None:
			raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D")
		return SimpleCrossAttnDownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			cross_attention_dim=cross_attention_dim,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
			skip_time_act=resnet_skip_time_act,
			output_scale_factor=resnet_out_scale_factor,
			only_cross_attention=only_cross_attention,
			cross_attention_norm=cross_attention_norm,
		)
	elif down_block_type == "SkipDownBlock2D":
		return SkipDownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			downsample_padding=downsample_padding,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif down_block_type == "AttnSkipDownBlock2D":
		return AttnSkipDownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif down_block_type == "DownEncoderBlock2D":
		return DownEncoderBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif down_block_type == "AttnDownEncoderBlock2D":
		return AttnDownEncoderBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			resnet_groups=resnet_groups,
			downsample_padding=downsample_padding,
			attention_head_dim=attention_head_dim,
			resnet_time_scale_shift=resnet_time_scale_shift,
		)
	elif down_block_type == "KDownBlock2D":
		return KDownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
		)
	elif down_block_type == "KCrossAttnDownBlock2D":
		return KCrossAttnDownBlock2D(
			num_layers=num_layers,
			in_channels=in_channels,
			out_channels=out_channels,
			temb_channels=temb_channels,
			dropout=dropout,
			add_downsample=add_downsample,
			resnet_eps=resnet_eps,
			resnet_act_fn=resnet_act_fn,
			cross_attention_dim=cross_attention_dim,
			attention_head_dim=attention_head_dim,
			add_self_attention=True if not add_downsample else False,
		)
	raise ValueError(f"{down_block_type} does not exist.")
