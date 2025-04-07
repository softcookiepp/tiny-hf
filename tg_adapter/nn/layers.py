from .module import Module
from tinygrad import Tensor
from ..tensor import _convert_base as _cb
from ..tensor import _disinherit

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
		
class Conv1d
