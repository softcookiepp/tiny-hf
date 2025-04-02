import tg_adapter as tga
from tg_adapter import F
import numpy as np
import tinygrad
from typing import Optional

class RMSNorm(tga.nn.Module):
	r"""
	RMS Norm as introduced in https://arxiv.org/abs/1910.07467 by Zhang et al.

	Args:
		dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
		eps (`float`): Small value to use when calculating the reciprocal of the square-root.
		elementwise_affine (`bool`, defaults to `True`):
			Boolean flag to denote if affine transformation should be applied.
		bias (`bool`, defaults to False): If also training the `bias` param.
	"""

	def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
		super().__init__()

		self.eps = eps
		self.elementwise_affine = elementwise_affine

		if isinstance(dim, numbers.Integral):
			dim = (dim,)

		#self.dim = torch.Size(dim)
		self.dim = dim
		if not isinstance(dim, tuple):
			self.dim = (dim,)

		self.weight = None
		self.bias = None

		if elementwise_affine:
			#self.weight = nn.Parameter(torch.ones(dim))
			self.weight = tinygrad.Tensor.ones(dim)
			if bias:
				#self.bias = nn.Parameter(torch.zeros(dim))
				self.bias = tinygrad.Tensor.zeros(dim)
		
		# since tinygrad's RMSNorm doesn't support affine parameters,
		# it is better to juse have it as an attribute on this layer
		self._rmsnorm = tinygrad.nn.RMSNorm(dim, eps)

	def forward(self, hidden_states):
		"""
		raise NotImplementedError
		if is_torch_npu_available():
			import torch_npu

			if self.weight is not None:
				# convert into half-precision if necessary
				if self.weight.dtype in [torch.float16, torch.bfloat16]:
					hidden_states = hidden_states.to(self.weight.dtype)
			hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
			if self.bias is not None:
				hidden_states = hidden_states + self.bias
		elif is_torch_version(">=", "2.4"):
			if self.weight is not None:
				# convert into half-precision if necessary
				if self.weight.dtype in [torch.float16, torch.bfloat16]:
					hidden_states = hidden_states.to(self.weight.dtype)
			hidden_states = nn.functional.rms_norm(
				hidden_states, normalized_shape=(hidden_states.shape[-1],), weight=self.weight, eps=self.eps
			)
			if self.bias is not None:
				hidden_states = hidden_states + self.bias
		else:
			input_dtype = hidden_states.dtype
			variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
			hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

			if self.weight is not None:
				# convert into half-precision if necessary
				if self.weight.dtype in [torch.float16, torch.bfloat16]:
					hidden_states = hidden_states.to(self.weight.dtype)
				hidden_states = hidden_states * self.weight
				if self.bias is not None:
					hidden_states = hidden_states + self.bias
			else:
				hidden_states = hidden_states.to(input_dtype)

		return hidden_states
		"""
		# much easier c:
		hidden_states = self._rmsnorm(hidden_states)
		if not self.weight is None:
			hidden_states = hidden_states*self.weight
			if not self.bias is None:
				hidden_states = hidden_states*self.bias
		return hidden_states

class AdaGroupNorm(tga.nn.Module):
	r"""
	GroupNorm layer modified to incorporate timestep embeddings.

	Parameters:
		embedding_dim (`int`): The size of each embedding vector.
		num_embeddings (`int`): The size of the embeddings dictionary.
		num_groups (`int`): The number of groups to separate the channels into.
		act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
		eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
	"""

	def __init__(
		self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
	):
		super().__init__()
		self.num_groups = num_groups
		self.eps = eps

		if act_fn is None:
			self.act = None
		else:
			raise NotImplementedError
			# this is going to be one of those silly utilities
			self.act = get_activation(act_fn)

		#self.linear = nn.Linear(embedding_dim, out_dim * 2)
		self.linear = tinygrad.nn.Linear(embedding_dim, out_dim * 2)

	def forward(self, x: tinygrad.Tensor, emb: tinygrad.Tensor) -> tinygrad.Tensor:
		if self.act:
			emb = self.act(emb)
		emb = self.linear(emb)
		emb = emb[:, :, None, None]
		scale, shift = emb.chunk(2, dim=1)
		
		# all goodie now!
		x = F.group_norm(x, self.num_groups, eps=self.eps)
		x = x * (1 + scale) + shift
		return x
