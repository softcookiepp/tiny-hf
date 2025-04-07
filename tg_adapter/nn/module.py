import tinygrad
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, torch_load
from typing import Iterable
from ..tensor import _convert_base as _cb
from ..tensor import _disinherit
import inspect
from ..device import device

# adapter for https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class Module:
	def __init__(self, *args, **kwargs):
		self._train = True
		self._buffers = {}
	
	def add_module(self, name: str, module = None):
		assert not name in self.__dict__.keys() or self.__getattribute__(name) == module
		self.__dict__[name] = module
		self.__getattribute__(name)
	
	def apply(self, fn):
		# fn should be a function that accepts a single module as an argument
		fn(self)
		for _, module in self.named_modules():
			fn(module)
	
	def buffers(self):
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				for b in v.buffers():
					yield b
			elif isinstance(v, tinygrad.Tensor):
				# generally no idea what criteria is supposed to be used for this :c
				pass
	
	def bfloat16(self):
		raise NotImplementedError

	def named_children(self):
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				yield k, v

	def children(self):
		# Return an iterator over immediate children modules.
		for _, v in self.named_children():
			yield v
	 
	def parameters(self, recurse = True):
		if not recurse:
			raise NotImplementedError
		params = []
		for k, v in self.named_parameters():
			params.append(v)
		if isinstance(self, Iterable):
			for i, elem in enumerate(self):
				for k, mod in elem.named_parameters():
					params.append(mod)
		return params
		"""
		params = []
		for k, v in enumerate(self):
			print(v)
			if isinstance(v, tinygrad.Tensor):
				params.append(v)
			elif isinstance(v, list):
				for item in v:
					if isinstance(v, tinygrad.Tensor):
						params.append(v)
		for k, v in self.__dict__.items():
			print(k, v)
			if isinstance(v, tinygrad.Tensor):
				params.append(v)
			elif isinstance(v, list):
				for item in v:
					if isinstance(v, tinygrad.Tensor):
						params.append(v)
		for child in self.children():
			for param in self.children.params():
				params.append(param)
		return params
		"""

	def compile(self):
		# tinygrad has no compile equivalent lol
		pass
		
	def to(self, device):
		raise NotImplementedError
		
	def cuda(self, device = None):
		raise NotImplementedError
		
		
		
	def train(self, train = True):
		self._train = train
		
	def eval(self):
		self.train(False)
		
	def __call__(self, *args, **kwargs):
		# get parent function, check if it is not forward or __call__,
		# then invoke realize() if that is the case
		parent_function = inspect.stack()[1].function
		
		# actually disinheriting is not a good idea here
		#args = _disinherit(args)
		#kwargs = _disinherit(kwargs)
		print("SELF TYPE:", type(self) )
		out = self.forward(*args, **kwargs)
		if (not parent_function in ["__call__", "forward"]) and isinstance(out, tinygrad.Tensor):
			out = out.realize()
		return _cb(out)
		
	def forward(self, *args, **kwargs):
		raise NotImplementedError
		
	def load_state_dict(self, state_dict, strict = True, assign = False):
		tinygrad.nn.state.load_state_dict(self, state_dict, strict = strict, verbose = True)
		_cb(self)
		# expected and missing keys are not implemented yet
		return [], []
	
	def state_dict(self):
		return tinygrad.nn.state.get_state_dict(self)
	
	def __repr__(self):
		return f"{self.__class__}"
	
	def named_parameters(self, memo = None, prefix = "", remove_duplicate = True):
		#raise NotImplementedError
		for name, param in self.state_dict().items():
			yield name, param
			
	def named_modules(self, memo=None, prefix="", remove_duplicate=True):
		#raise NotImplementedError
		# prefix indicates this method is called recursively
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				k = prefix + k
				yield k, v
				for subk, subv in v.named_modules(prefix = f"{k}."):
					# use recursion c:
					yield subk, subv
	
	def modules(self, remove_duplicate = True):
		for k, v in self.named_modules(remove_duplicate = remove_duplicate):
			yield v
	
	def register_buffer(self, name, tensor, persistent = True):
		assert not name in self.__dict__.keys()
		self.__dict__[name] = tensor
	
	@property
	def dtype(self):
		raise NotImplementedError
		
		
	@property
	def training(self):
		# TODO: actually implement
		return False
