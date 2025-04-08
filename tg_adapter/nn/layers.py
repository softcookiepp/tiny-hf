from .module import Module
from tinygrad import Tensor
import tinygrad
from tinygrad.helpers import make_tuple, prod
from ..tensor import _convert_base as _cb
from ..tensor import _disinherit
import math

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
		return _cb(out)

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
		return _cb(_disinherit(inp).dropout(self._p) )

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
		
		self.weight = _cb(Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale) )
		
		self.bias = None
		if bias:
			self.bias = _cb(tinygrad.Tensor.uniform(out_channels, low=-scale, high=scale) )
	
	def forward(self, x):
		return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)

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
	def __init__(self, *args, **kwargs):
		raise NotImplementedError
		
class Linear(Module):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError
		
class Embedding(Module):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError
		

class GroupNorm(Module):
	def __init__(self, num_groups, num_channels, eps=1e-05, affine=True,
			device=None, dtype=None):
		self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
		self.weight: Tensor|None = Tensor.ones(num_channels) if affine else None
		self.bias: Tensor|None = Tensor.zeros(num_channels) if affine else None
	
	def forward(x):
		x = _disinherit(x)
		x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)

		if self.weight is None or self.bias is None: return _cb(x)
		out = x * self.weight.reshape(1, -1, *[1] * (x.ndim-2)) + self.bias.reshape(1, -1, *[1] * (x.ndim-2))
		return _cb(x)

