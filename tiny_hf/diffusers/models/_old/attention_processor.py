import tg_adapter as tga
from tg_adapter import F
import tinygrad
from typing import Optional, Tuple, Callable
import inspect

class SpatialNorm(tga.nn.Module):
	"""
	Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

	Args:
		f_channels (`int`):
			The number of channels for input to group normalization layer, and output of the spatial norm layer.
		zq_channels (`int`):
			The number of channels for the quantized vector as described in the paper.
	"""

	def __init__(
		self,
		f_channels: int,
		zq_channels: int,
	):
		super().__init__()
		#self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
		self.norm_layer = tinygrad.nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
		#self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
		self.conv_y = tinygrad.nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
		#self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
		self.conv_b = tinygrad.nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

	def forward(self, f: tinygrad.Tensor, zq: tinygrad.Tensor) -> tinygrad.Tensor:
		f_size = f.shape[-2:]
		zq = F.interpolate(zq, size=f_size, mode="nearest")
		norm_f = self.norm_layer(f)
		new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
		return new_f
		
class Attention(tga.nn.Module):
	r"""
	A cross attention layer.

	Parameters:
		query_dim (`int`):
			The number of channels in the query.
		cross_attention_dim (`int`, *optional*):
			The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
		heads (`int`,  *optional*, defaults to 8):
			The number of heads to use for multi-head attention.
		kv_heads (`int`,  *optional*, defaults to `None`):
			The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
			`kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
			Query Attention (MQA) otherwise GQA is used.
		dim_head (`int`,  *optional*, defaults to 64):
			The number of channels in each head.
		dropout (`float`, *optional*, defaults to 0.0):
			The dropout probability to use.
		bias (`bool`, *optional*, defaults to False):
			Set to `True` for the query, key, and value linear layers to contain a bias parameter.
		upcast_attention (`bool`, *optional*, defaults to False):
			Set to `True` to upcast the attention computation to `float32`.
		upcast_softmax (`bool`, *optional*, defaults to False):
			Set to `True` to upcast the softmax computation to `float32`.
		cross_attention_norm (`str`, *optional*, defaults to `None`):
			The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
		cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
			The number of groups to use for the group norm in the cross attention.
		added_kv_proj_dim (`int`, *optional*, defaults to `None`):
			The number of channels to use for the added key and value projections. If `None`, no projection is used.
		norm_num_groups (`int`, *optional*, defaults to `None`):
			The number of groups to use for the group norm in the attention.
		spatial_norm_dim (`int`, *optional*, defaults to `None`):
			The number of channels to use for the spatial normalization.
		out_bias (`bool`, *optional*, defaults to `True`):
			Set to `True` to use a bias in the output linear layer.
		scale_qk (`bool`, *optional*, defaults to `True`):
			Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
		only_cross_attention (`bool`, *optional*, defaults to `False`):
			Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
			`added_kv_proj_dim` is not `None`.
		eps (`float`, *optional*, defaults to 1e-5):
			An additional value added to the denominator in group normalization that is used for numerical stability.
		rescale_output_factor (`float`, *optional*, defaults to 1.0):
			A factor to rescale the output by dividing it with this value.
		residual_connection (`bool`, *optional*, defaults to `False`):
			Set to `True` to add the residual connection to the output.
		_from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
			Set to `True` if the attention block is loaded from a deprecated state dict.
		processor (`AttnProcessor`, *optional*, defaults to `None`):
			The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
			`AttnProcessor` otherwise.
	"""

	def __init__(
		self,
		query_dim: int,
		cross_attention_dim: Optional[int] = None,
		heads: int = 8,
		kv_heads: Optional[int] = None,
		dim_head: int = 64,
		dropout: float = 0.0,
		bias: bool = False,
		upcast_attention: bool = False,
		upcast_softmax: bool = False,
		cross_attention_norm: Optional[str] = None,
		cross_attention_norm_num_groups: int = 32,
		qk_norm: Optional[str] = None,
		added_kv_proj_dim: Optional[int] = None,
		added_proj_bias: Optional[bool] = True,
		norm_num_groups: Optional[int] = None,
		spatial_norm_dim: Optional[int] = None,
		out_bias: bool = True,
		scale_qk: bool = True,
		only_cross_attention: bool = False,
		eps: float = 1e-5,
		rescale_output_factor: float = 1.0,
		residual_connection: bool = False,
		_from_deprecated_attn_block: bool = False,
		processor: Optional["AttnProcessor"] = None,
		out_dim: int = None,
		out_context_dim: int = None,
		context_pre_only=None,
		pre_only=False,
		elementwise_affine: bool = True,
		is_causal: bool = False,
	):
		super().__init__()

		# To prevent circular import.
		#from .normalization import FP32LayerNorm, LpNorm, RMSNorm
		from .normalization import RMSNorm

		self.inner_dim = out_dim if out_dim is not None else dim_head * heads
		self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
		self.query_dim = query_dim
		self.use_bias = bias
		self.is_cross_attention = cross_attention_dim is not None
		self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
		self.upcast_attention = upcast_attention
		self.upcast_softmax = upcast_softmax
		self.rescale_output_factor = rescale_output_factor
		self.residual_connection = residual_connection
		self.dropout = dropout
		self.fused_projections = False
		self.out_dim = out_dim if out_dim is not None else query_dim
		self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
		self.context_pre_only = context_pre_only
		self.pre_only = pre_only
		self.is_causal = is_causal

		# we make use of this private variable to know whether this class is loaded
		# with an deprecated state dict so that we can convert it on the fly
		self._from_deprecated_attn_block = _from_deprecated_attn_block

		self.scale_qk = scale_qk
		self.scale = dim_head**-0.5 if self.scale_qk else 1.0

		self.heads = out_dim // dim_head if out_dim is not None else heads
		# for slice_size > 0 the attention score computation
		# is split across the batch axis to save memory
		# You can set slice_size with `set_attention_slice`
		self.sliceable_head_dim = heads

		self.added_kv_proj_dim = added_kv_proj_dim
		self.only_cross_attention = only_cross_attention

		if self.added_kv_proj_dim is None and self.only_cross_attention:
			raise ValueError(
				"`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
			)

		if norm_num_groups is not None:
			self.group_norm = tinygrad.nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
		else:
			self.group_norm = None

		if spatial_norm_dim is not None:
			self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
		else:
			self.spatial_norm = None

		if qk_norm is None:
			self.norm_q = None
			self.norm_k = None
		elif qk_norm == "layer_norm" or qk_norm == "fp32_layer_norm":
			# lol
			self.norm_q = tinygrad.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
			self.norm_k = tinygrad.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
		elif qk_norm == "fp32_layer_norm":
			raise NotImplementedError
			self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
			self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
		elif qk_norm == "layer_norm_across_heads":
			# Lumina applies qk norm across all heads
			self.norm_q = tinygrad.nn.LayerNorm(dim_head * heads, eps=eps)
			self.norm_k = tinygrad.nn.LayerNorm(dim_head * kv_heads, eps=eps)
		elif qk_norm == "rms_norm":
			self.norm_q = RMSNorm(dim_head, eps=eps)
			self.norm_k = RMSNorm(dim_head, eps=eps)
		elif qk_norm == "rms_norm_across_heads":
			# LTX applies qk norm across all heads
			self.norm_q = RMSNorm(dim_head * heads, eps=eps)
			self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
		elif qk_norm == "l2":
			raise NotImplementedError
			self.norm_q = LpNorm(p=2, dim=-1, eps=eps)
			self.norm_k = LpNorm(p=2, dim=-1, eps=eps)
		else:
			raise ValueError(
				f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."
			)

		if cross_attention_norm is None:
			self.norm_cross = None
		elif cross_attention_norm == "layer_norm":
			self.norm_cross = tinygrad.nn.LayerNorm(self.cross_attention_dim)
		elif cross_attention_norm == "group_norm":
			if self.added_kv_proj_dim is not None:
				# The given `encoder_hidden_states` are initially of shape
				# (batch_size, seq_len, added_kv_proj_dim) before being projected
				# to (batch_size, seq_len, cross_attention_dim). The norm is applied
				# before the projection, so we need to use `added_kv_proj_dim` as
				# the number of channels for the group norm.
				norm_cross_num_channels = added_kv_proj_dim
			else:
				norm_cross_num_channels = self.cross_attention_dim

			self.norm_cross = tinygrad.nn.GroupNorm(
				num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
			)
		else:
			raise ValueError(
				f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
			)

		self.to_q = tinygrad.nn.Linear(query_dim, self.inner_dim, bias=bias)

		if not self.only_cross_attention:
			# only relevant for the `AddedKVProcessor` classes
			self.to_k = tinygrad.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
			self.to_v = tinygrad.nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
		else:
			self.to_k = None
			self.to_v = None

		self.added_proj_bias = added_proj_bias
		if self.added_kv_proj_dim is not None:
			self.add_k_proj = tinygrad.nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
			self.add_v_proj = tinygrad.nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
			if self.context_pre_only is not None:
				self.add_q_proj = tinygrad.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
		else:
			self.add_q_proj = None
			self.add_k_proj = None
			self.add_v_proj = None

		if not self.pre_only:
			self.to_out = []#nn.ModuleList([])
			self.to_out.append(tinygrad.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
			self.to_out.append(tga.nn.Dropout(dropout))
		else:
			self.to_out = None

		if self.context_pre_only is not None and not self.context_pre_only:
			self.to_add_out = tinygrad.nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)
		else:
			self.to_add_out = None

		if qk_norm is not None and added_kv_proj_dim is not None:
			if qk_norm == "layer_norm" or qk_norm == "fp32_layer_norm":
				# might as well lool
				self.norm_added_q = tinygrad.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
				self.norm_added_k = tinygrad.nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
			elif qk_norm == "fp32_layer_norm":
				raise NotImplementedError
				self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
				self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
			elif qk_norm == "rms_norm":
				self.norm_added_q = RMSNorm(dim_head, eps=eps)
				self.norm_added_k = RMSNorm(dim_head, eps=eps)
			elif qk_norm == "rms_norm_across_heads":
				# Wan applies qk norm across all heads
				# Wan also doesn't apply a q norm
				self.norm_added_q = None
				self.norm_added_k = RMSNorm(dim_head * kv_heads, eps=eps)
			else:
				raise ValueError(
					f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
				)
		else:
			self.norm_added_q = None
			self.norm_added_k = None

		# set attention processor
		# We use the AttnProcessor2_0 by default when torch 2.x is used which uses
		# torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
		# but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
		if processor is None:
			processor = (
				AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
			)
		self.set_processor(processor)

	def set_use_xla_flash_attention(
		self,
		use_xla_flash_attention: bool,
		partition_spec: Optional[Tuple[Optional[str], ...]] = None,
		is_flux=False,
	) -> None:
		r"""
		Set whether to use xla flash attention from `torch_xla` or not.

		Args:
			use_xla_flash_attention (`bool`):
				Whether to use pallas flash attention kernel from `torch_xla` or not.
			partition_spec (`Tuple[]`, *optional*):
				Specify the partition specification if using SPMD. Otherwise None.
		"""
		if use_xla_flash_attention:
			if not is_torch_xla_available:
				raise "torch_xla is not available"
			elif is_torch_xla_version("<", "2.3"):
				raise "flash attention pallas kernel is supported from torch_xla version 2.3"
			elif is_spmd() and is_torch_xla_version("<", "2.4"):
				raise "flash attention pallas kernel using SPMD is supported from torch_xla version 2.4"
			else:
				if is_flux:
					processor = XLAFluxFlashAttnProcessor2_0(partition_spec)
				else:
					processor = XLAFlashAttnProcessor2_0(partition_spec)
		else:
			processor = (
				AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
			)
		self.set_processor(processor)

	def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
		r"""
		Set whether to use npu flash attention from `torch_npu` or not.

		"""
		if use_npu_flash_attention:
			processor = AttnProcessorNPU()
		else:
			# set attention processor
			# We use the AttnProcessor2_0 by default when torch 2.x is used which uses
			# torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
			# but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
			processor = (
				AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
			)
		self.set_processor(processor)

	def set_use_memory_efficient_attention_xformers(
		self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
	) -> None:
		r"""
		Set whether to use memory efficient attention from `xformers` or not.

		Args:
			use_memory_efficient_attention_xformers (`bool`):
				Whether to use memory efficient attention from `xformers` or not.
			attention_op (`Callable`, *optional*):
				The attention operation to use. Defaults to `None` which uses the default attention operation from
				`xformers`.
		"""
		is_custom_diffusion = hasattr(self, "processor") and isinstance(
			self.processor,
			(CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0),
		)
		is_added_kv_processor = hasattr(self, "processor") and isinstance(
			self.processor,
			(
				AttnAddedKVProcessor,
				AttnAddedKVProcessor2_0,
				SlicedAttnAddedKVProcessor,
				XFormersAttnAddedKVProcessor,
			),
		)
		is_ip_adapter = hasattr(self, "processor") and isinstance(
			self.processor,
			(IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor),
		)
		is_joint_processor = hasattr(self, "processor") and isinstance(
			self.processor,
			(
				JointAttnProcessor2_0,
				XFormersJointAttnProcessor,
			),
		)

		if use_memory_efficient_attention_xformers:
			if is_added_kv_processor and is_custom_diffusion:
				raise NotImplementedError(
					f"Memory efficient attention is currently not supported for custom diffusion for attention processor type {self.processor}"
				)
			if not is_xformers_available():
				raise ModuleNotFoundError(
					(
						"Refer to https://github.com/facebookresearch/xformers for more information on how to install"
						" xformers"
					),
					name="xformers",
				)
			elif not torch.cuda.is_available():
				raise ValueError(
					"torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
					" only available for GPU "
				)
			else:
				try:
					# Make sure we can run the memory efficient attention
					dtype = None
					if attention_op is not None:
						op_fw, op_bw = attention_op
						dtype, *_ = op_fw.SUPPORTED_DTYPES
					q = torch.randn((1, 2, 40), device="cuda", dtype=dtype)
					_ = xformers.ops.memory_efficient_attention(q, q, q)
				except Exception as e:
					raise e

			if is_custom_diffusion:
				processor = CustomDiffusionXFormersAttnProcessor(
					train_kv=self.processor.train_kv,
					train_q_out=self.processor.train_q_out,
					hidden_size=self.processor.hidden_size,
					cross_attention_dim=self.processor.cross_attention_dim,
					attention_op=attention_op,
				)
				processor.load_state_dict(self.processor.state_dict())
				if hasattr(self.processor, "to_k_custom_diffusion"):
					processor.to(self.processor.to_k_custom_diffusion.weight.device)
			elif is_added_kv_processor:
				# TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
				# which uses this type of cross attention ONLY because the attention mask of format
				# [0, ..., -10.000, ..., 0, ...,] is not supported
				# throw warning
				logger.info(
					"Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
				)
				processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
			elif is_ip_adapter:
				processor = IPAdapterXFormersAttnProcessor(
					hidden_size=self.processor.hidden_size,
					cross_attention_dim=self.processor.cross_attention_dim,
					num_tokens=self.processor.num_tokens,
					scale=self.processor.scale,
					attention_op=attention_op,
				)
				processor.load_state_dict(self.processor.state_dict())
				if hasattr(self.processor, "to_k_ip"):
					processor.to(
						device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
					)
			elif is_joint_processor:
				processor = XFormersJointAttnProcessor(attention_op=attention_op)
			else:
				processor = XFormersAttnProcessor(attention_op=attention_op)
		else:
			if is_custom_diffusion:
				attn_processor_class = (
					CustomDiffusionAttnProcessor2_0
					if hasattr(F, "scaled_dot_product_attention")
					else CustomDiffusionAttnProcessor
				)
				processor = attn_processor_class(
					train_kv=self.processor.train_kv,
					train_q_out=self.processor.train_q_out,
					hidden_size=self.processor.hidden_size,
					cross_attention_dim=self.processor.cross_attention_dim,
				)
				processor.load_state_dict(self.processor.state_dict())
				if hasattr(self.processor, "to_k_custom_diffusion"):
					processor.to(self.processor.to_k_custom_diffusion.weight.device)
			elif is_ip_adapter:
				processor = IPAdapterAttnProcessor2_0(
					hidden_size=self.processor.hidden_size,
					cross_attention_dim=self.processor.cross_attention_dim,
					num_tokens=self.processor.num_tokens,
					scale=self.processor.scale,
				)
				processor.load_state_dict(self.processor.state_dict())
				if hasattr(self.processor, "to_k_ip"):
					processor.to(
						device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
					)
			else:
				# set attention processor
				# We use the AttnProcessor2_0 by default when torch 2.x is used which uses
				# torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
				# but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
				processor = (
					AttnProcessor2_0()
					if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
					else AttnProcessor()
				)

		self.set_processor(processor)

	def set_attention_slice(self, slice_size: int) -> None:
		r"""
		Set the slice size for attention computation.

		Args:
			slice_size (`int`):
				The slice size for attention computation.
		"""
		if slice_size is not None and slice_size > self.sliceable_head_dim:
			raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

		if slice_size is not None and self.added_kv_proj_dim is not None:
			processor = SlicedAttnAddedKVProcessor(slice_size)
		elif slice_size is not None:
			processor = SlicedAttnProcessor(slice_size)
		elif self.added_kv_proj_dim is not None:
			processor = AttnAddedKVProcessor()
		else:
			# set attention processor
			# We use the AttnProcessor2_0 by default when torch 2.x is used which uses
			# torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
			# but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
			processor = (
				AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
			)

		self.set_processor(processor)

	def set_processor(self, processor: "AttnProcessor") -> None:
		r"""
		Set the attention processor to use.

		Args:
			processor (`AttnProcessor`):
				The attention processor to use.
		"""
		# if current processor is in `self._modules` and if passed `processor` is not, we need to
		# pop `processor` from `self._modules`
		if (
			hasattr(self, "processor")
			and isinstance(self.processor, torch.nn.Module)
			and not isinstance(processor, torch.nn.Module)
		):
			logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
			self._modules.pop("processor")

		self.processor = processor

	def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
		r"""
		Get the attention processor in use.

		Args:
			return_deprecated_lora (`bool`, *optional*, defaults to `False`):
				Set to `True` to return the deprecated LoRA attention processor.

		Returns:
			"AttentionProcessor": The attention processor in use.
		"""
		if not return_deprecated_lora:
			return self.processor

	def forward(
		self,
		hidden_states: tinygrad.Tensor,
		encoder_hidden_states: Optional[tinygrad.Tensor] = None,
		attention_mask: Optional[tinygrad.Tensor] = None,
		**cross_attention_kwargs,
	) -> tinygrad.Tensor:
		r"""
		The forward method of the `Attention` class.

		Args:
			hidden_states (`tinygrad.Tensor`):
				The hidden states of the query.
			encoder_hidden_states (`tinygrad.Tensor`, *optional*):
				The hidden states of the encoder.
			attention_mask (`tinygrad.Tensor`, *optional*):
				The attention mask to use. If `None`, no mask is applied.
			**cross_attention_kwargs:
				Additional keyword arguments to pass along to the cross attention.

		Returns:
			`tinygrad.Tensor`: The output of the attention layer.
		"""
		# The `Attention` class can call different attention processors / attention functions
		# here we simply pass along all tensors to the selected processor class
		# For standard processors that are defined here, `**cross_attention_kwargs` is empty

		attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
		quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
		unused_kwargs = [
			k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
		]
		if len(unused_kwargs) > 0:
			logger.warning(
				f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
			)
		cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

		return self.processor(
			self,
			hidden_states,
			encoder_hidden_states=encoder_hidden_states,
			attention_mask=attention_mask,
			**cross_attention_kwargs,
		)

	def batch_to_head_dim(self, tensor: tinygrad.Tensor) -> tinygrad.Tensor:
		r"""
		Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
		is the number of heads initialized while constructing the `Attention` class.

		Args:
			tensor (`tinygrad.Tensor`): The tensor to reshape.

		Returns:
			`tinygrad.Tensor`: The reshaped tensor.
		"""
		head_size = self.heads
		batch_size, seq_len, dim = tensor.shape
		tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
		tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
		return tensor

	def head_to_batch_dim(self, tensor: tinygrad.Tensor, out_dim: int = 3) -> tinygrad.Tensor:
		r"""
		Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
		the number of heads initialized while constructing the `Attention` class.

		Args:
			tensor (`tinygrad.Tensor`): The tensor to reshape.
			out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
				reshaped to `[batch_size * heads, seq_len, dim // heads]`.

		Returns:
			`tinygrad.Tensor`: The reshaped tensor.
		"""
		head_size = self.heads
		if tensor.ndim == 3:
			batch_size, seq_len, dim = tensor.shape
			extra_dim = 1
		else:
			batch_size, extra_dim, seq_len, dim = tensor.shape
		tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
		tensor = tensor.permute(0, 2, 1, 3)

		if out_dim == 3:
			tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

		return tensor

	def get_attention_scores(
		self, query: tinygrad.Tensor, key: tinygrad.Tensor, attention_mask: Optional[tinygrad.Tensor] = None
	) -> tinygrad.Tensor:
		r"""
		Compute the attention scores.

		Args:
			query (`tinygrad.Tensor`): The query tensor.
			key (`tinygrad.Tensor`): The key tensor.
			attention_mask (`tinygrad.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

		Returns:
			`tinygrad.Tensor`: The attention probabilities/scores.
		"""
		dtype = query.dtype
		if self.upcast_attention:
			query = query.float()
			key = key.float()

		if attention_mask is None:
			baddbmm_input = torch.empty(
				query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
			)
			beta = 0
		else:
			baddbmm_input = attention_mask
			beta = 1

		attention_scores = torch.baddbmm(
			baddbmm_input,
			query,
			key.transpose(-1, -2),
			beta=beta,
			alpha=self.scale,
		)
		del baddbmm_input

		if self.upcast_softmax:
			attention_scores = attention_scores.float()

		attention_probs = attention_scores.softmax(dim=-1)
		del attention_scores

		attention_probs = attention_probs.to(dtype)

		return attention_probs

	def prepare_attention_mask(
		self, attention_mask: tinygrad.Tensor, target_length: int, batch_size: int, out_dim: int = 3
	) -> tinygrad.Tensor:
		r"""
		Prepare the attention mask for the attention computation.

		Args:
			attention_mask (`tinygrad.Tensor`):
				The attention mask to prepare.
			target_length (`int`):
				The target length of the attention mask. This is the length of the attention mask after padding.
			batch_size (`int`):
				The batch size, which is used to repeat the attention mask.
			out_dim (`int`, *optional*, defaults to `3`):
				The output dimension of the attention mask. Can be either `3` or `4`.

		Returns:
			`tinygrad.Tensor`: The prepared attention mask.
		"""
		head_size = self.heads
		if attention_mask is None:
			return attention_mask

		current_length: int = attention_mask.shape[-1]
		if current_length != target_length:
			if attention_mask.device.type == "mps":
				# HACK: MPS: Does not support padding by greater than dimension of input tensor.
				# Instead, we can manually construct the padding tensor.
				padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
				padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
				attention_mask = torch.cat([attention_mask, padding], dim=2)
			else:
				# TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
				#       we want to instead pad by (0, remaining_length), where remaining_length is:
				#       remaining_length: int = target_length - current_length
				# TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
				attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

		if out_dim == 3:
			if attention_mask.shape[0] < batch_size * head_size:
				attention_mask = attention_mask.repeat_interleave(
					head_size, dim=0, output_size=attention_mask.shape[0] * head_size
				)
		elif out_dim == 4:
			attention_mask = attention_mask.unsqueeze(1)
			attention_mask = attention_mask.repeat_interleave(
				head_size, dim=1, output_size=attention_mask.shape[1] * head_size
			)

		return attention_mask

	def norm_encoder_hidden_states(self, encoder_hidden_states: tinygrad.Tensor) -> tinygrad.Tensor:
		r"""
		Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
		`Attention` class.

		Args:
			encoder_hidden_states (`tinygrad.Tensor`): Hidden states of the encoder.

		Returns:
			`tinygrad.Tensor`: The normalized encoder hidden states.
		"""
		assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

		if isinstance(self.norm_cross, nn.LayerNorm):
			encoder_hidden_states = self.norm_cross(encoder_hidden_states)
		elif isinstance(self.norm_cross, nn.GroupNorm):
			# Group norm norms along the channels dimension and expects
			# input to be in the shape of (N, C, *). In this case, we want
			# to norm along the hidden dimension, so we need to move
			# (batch_size, sequence_length, hidden_size) ->
			# (batch_size, hidden_size, sequence_length)
			encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
			encoder_hidden_states = self.norm_cross(encoder_hidden_states)
			encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
		else:
			assert False

		return encoder_hidden_states
	
	# TODO: figure out how to disable gradient calculation during this thingy
	#@torch.no_grad()
	#@tinygrad.inference_mode()
	def fuse_projections(self, fuse=True):
		device = self.to_q.weight.data.device
		dtype = self.to_q.weight.data.dtype

		if not self.is_cross_attention:
			# fetch weight matrices.
			concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
			in_features = concatenated_weights.shape[1]
			out_features = concatenated_weights.shape[0]

			# create a new single projection layer and copy over the weights.
			self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
			self.to_qkv.weight.copy_(concatenated_weights)
			if self.use_bias:
				concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
				self.to_qkv.bias.copy_(concatenated_bias)

		else:
			concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
			in_features = concatenated_weights.shape[1]
			out_features = concatenated_weights.shape[0]

			self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
			self.to_kv.weight.copy_(concatenated_weights)
			if self.use_bias:
				concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
				self.to_kv.bias.copy_(concatenated_bias)

		# handle added projections for SD3 and others.
		if (
			getattr(self, "add_q_proj", None) is not None
			and getattr(self, "add_k_proj", None) is not None
			and getattr(self, "add_v_proj", None) is not None
		):
			concatenated_weights = torch.cat(
				[self.add_q_proj.weight.data, self.add_k_proj.weight.data, self.add_v_proj.weight.data]
			)
			in_features = concatenated_weights.shape[1]
			out_features = concatenated_weights.shape[0]

			self.to_added_qkv = nn.Linear(
				in_features, out_features, bias=self.added_proj_bias, device=device, dtype=dtype
			)
			self.to_added_qkv.weight.copy_(concatenated_weights)
			if self.added_proj_bias:
				concatenated_bias = torch.cat(
					[self.add_q_proj.bias.data, self.add_k_proj.bias.data, self.add_v_proj.bias.data]
				)
				self.to_added_qkv.bias.copy_(concatenated_bias)

		self.fused_projections = fuse

class AttnProcessor2_0:
	r"""
	Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
	"""

	def __init__(self):
		#if not hasattr(F, "scaled_dot_product_attention"):
		if not hasattr(tinygrad.Tensor, "scaled_dot_product_attention"):
			#raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
			raise ImportError("Tinygrad should have this, no idea why it doesn't.")

	def __call__(
		self,
		attn: Attention,
		hidden_states: tinygrad.Tensor,
		encoder_hidden_states: Optional[tinygrad.Tensor] = None,
		attention_mask: Optional[tinygrad.Tensor] = None,
		temb: Optional[tinygrad.Tensor] = None,
		*args,
		**kwargs,
	) -> tinygrad.Tensor:
		# don't care
		"""
		if len(args) > 0 or kwargs.get("scale", None) is not None:
			deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
			deprecate("scale", "1.0.0", deprecation_message)
		"""

		residual = hidden_states
		if attn.spatial_norm is not None:
			hidden_states = attn.spatial_norm(hidden_states, temb)

		input_ndim = hidden_states.ndim

		if input_ndim == 4:
			batch_size, channel, height, width = hidden_states.shape
			# same thing as reshape f
			hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

		batch_size, sequence_length, _ = (
			hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
		)

		if attention_mask is not None:
			attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
			# scaled_dot_product_attention expects attention_mask shape to be
			# (batch, heads, source_length, target_length)
			attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

		if attn.group_norm is not None:
			hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

		query = attn.to_q(hidden_states)

		if encoder_hidden_states is None:
			encoder_hidden_states = hidden_states
		elif attn.norm_cross:
			encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

		key = attn.to_k(encoder_hidden_states)
		value = attn.to_v(encoder_hidden_states)

		inner_dim = key.shape[-1]
		head_dim = inner_dim // attn.heads

		query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
		value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

		if attn.norm_q is not None:
			query = attn.norm_q(query)
		if attn.norm_k is not None:
			key = attn.norm_k(key)

		# the output of sdp = (batch, num_heads, seq_len, head_dim)
		# TODO: add support for attn.scale when we move to Torch 2.1
		hidden_states = F.scaled_dot_product_attention(
			query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
		)

		hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
		#hidden_states = hidden_states.to(query.dtype)
		hidden_states = hidden_states.cast(query.dtype)

		# linear proj
		hidden_states = attn.to_out[0](hidden_states)
		# dropout
		hidden_states = attn.to_out[1](hidden_states)

		if input_ndim == 4:
			hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

		if attn.residual_connection:
			hidden_states = hidden_states + residual

		hidden_states = hidden_states / attn.rescale_output_factor

		return hidden_states
