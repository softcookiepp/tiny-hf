from .module import Module
import tinygrad
from tinygrad.helpers import make_tuple, prod
import math
from ..tensor import AdapterTensor as AT
from ..tensor import convert_to_torch, convert_to_tg

class AvgPool2d(Module):
	def __init__(self, kernel_size, stride=None, padding=0,
			ceil_mode=False, count_include_pad=True):
		super().__init__()
		self._kernel_size = kernel_size
		self._stride = stride
		if stride is None:
			self._stride = kernel_size
		self._padding = padding
		self._ceil_mode = ceil_mode
		self._count_include_pad = count_include_pad
	
	def forward(self, inp):
		inp = _disinherit(inp)
		out = inp.avg_pool2d(self._kernel_size, self._stride,
			1, self._padding, self._ceil_mode, self._count_include_pad)
		return AT(out)

class SequentialIterator:
	def __init__(self, module):
		self._elem = 0
		self._module = module
	
	def __next__(self):
		if self._elem < len(self._module.args):
			elem = self._module.args[elem]
			self._elem += 1
			return elem
		else:
			raise StopIteration

class Sequential(Module):
	def __init__(self, *args):
		super().__init__()
		self._args = args
	
	@property
	def args(self):
		return self._args
		
	def forward(self, x):
		for arg in self._args:
			x = arg(x)
		return x
		
	def __iter__(self):
		return SequentialIterator(self)
		
	
class Dropout(Module):
	def __init__(self, p = 0.5, inplace = False):
		if inplace:
			# tinygrad has no inplace operator for dropout and I feel way too lazy to make one
			raise NotImplementedError
		
		self._p = p
	
	def forward(self, inp):
		return AT(inp.tg.dropout(self._p) )

class AdaGroupNorm(Module):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError
		
class Identity(Module):
	def __init__(self, *args, **kwargs):
		pass
	def forward(self, *args, **kwargs):
		if len(args) == 1:
			return args[0]
		else:
			return args
		
class Upsample(Module):
	def __init__(self, *args, **kwargs):
		pass
	
	def forward(self, *args, **kwargs):
		raise NotImplementedError

class ConvNd(Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		
		# must have dimensionality
		assert not dim is None
		if isinstance(padding, str):
			assert padding in ["valid", "same"]
		self.kernel_size = make_tuple(kernel_size, dim)
		self.stride, self.dilation, self.groups, self.padding = stride, dilation, groups, padding
		scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
		
		#self.weight = tinygrad.Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
		self.weight = AT(tinygrad.Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale) )
		
		self.bias = None
		if bias:
			self.bias = AT( tinygrad.Tensor.uniform(out_channels, low=-scale, high=scale) )
	
	def forward(self, x):
		x, weight, bias = x.tg, self.weight.tg, self.bias.tg
		x = x.conv2d(weight, bias, self.groups, self.stride, self.dilation, self.padding)
		return AT(x)

# ugh, I forgot that torch is going to expect this crap as a type :c

class Conv1d(ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 1)
	
class Conv2d(ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 2)

class Conv3d(ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 3)
			
class LayerNorm(Module):
	def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True,
			bias=True, device=None, dtype=None):
		self.normalized_shape: tuple[int, ...] = make_tuple(normalized_shape, 1)
		self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
		self.weight = AT(tinygrad.Tensor.ones(*self.normalized_shape) ) if elementwise_affine else None
		self.bias = AT(tinygrad.Tensor.zeros(*self.normalized_shape) ) if bias and elementwise_affine else None
	
	def forward(self, x):
		x, weight, bias = _disinherit(x, self.weight, self.bias)
		assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
		x = x.layernorm(eps=self.eps, axis=self.axis)
		if not self.elementwise_affine: return AT(x)
		return AT(x * weight + bias)
		
class Linear(Module):
	def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
		bound = 1 / math.sqrt(in_features)
		self.weight = AT(tinygrad.Tensor.uniform(out_features, in_features, low=-bound, high=bound) )
		self.bias = AT( tinygrad.Tensor.uniform(out_features, low=-bound, high=bound) ) if bias else None
	
	def forward(self, x):
		# disinherit stuff
		x, weight, bias = convert_to_tg(x, self.weight, self.bias)
		x = x.linear(weight.transpose(), bias)
		return convert_to_torch(x)
		
class Embedding(Module):
	def __init__(self, vocab_size:int, embed_size:int):
		self.vocab_sz, self.embed_sz, self.weight = vocab_size, embed_size, convert_to_torch(tinygrad.Tensor.glorot_uniform(vocab_size, embed_size) )
	
	def forward(self, idx):
		vocab_sz, embed_sz, weight = convert_to_tg(self.vocab_sz, self.embed_sz, self.weight)
		if not hasattr(self, 'arange'): self.arange = tinygrad.Tensor.arange(vocab_sz, requires_grad=False, device=weight.device).unsqueeze(-1)
		big_shp = idx.shape+(vocab_sz, embed_sz)
		arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), weight.expand(big_shp)
		return _cb( (arange == idx).mul(vals).sum(-2, acc_dtype=vals.dtype) )
		

class GroupNorm(Module):
	def __init__(self, num_groups, num_channels, eps=1e-05, affine=True,
			device=None, dtype=None):
		self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
		self.weight = AT(tinygrad.Tensor.ones(num_channels) ) if affine else None
		self.bias = AT( tinygrad.Tensor.zeros(num_channels) ) if affine else None
	
	def forward(self, x):
		# disinherit stuff
		x, weight, bias = convert_to_tg(x, self.weight, self.bias)
		x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)
		
		
		
		if weight is None or bias is None: return _cb(x)
		out = x * weight.reshape(1, -1, *[1] * (x.ndim-2)) + bias.reshape(1, -1, *[1] * (x.ndim-2))
		return AT(out)

