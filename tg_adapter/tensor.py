import tinygrad

from .backend_environment_config import get_backend_override, tinygrad_device_to_torch_device
from .device import device as Device
import inspect
import numpy as np
from .types import get_torch_dtype

class AdapterTensor:
	def __init__(self, data, dtype = None, device = None,
			requires_grad = False, pin_memory = False):
		# pin memory is unused, but kept for compatibility
		# convert device, duh
		if isinstance(device, Device):
			raise NotImplementedError
		
		if isinstance(data, tinygrad.Tensor):
			self._tg = data
		elif isinstance(data, np.ndarray):
			self._tg = tinygrad.Tensor(data)
		else:
			raise Exception(f"Tensor creationw with {type(data)} not yet implemented.")
	
	@property
	def tg(self):
		return self._tg
		
	@property
	def shape(self):
		return self.tg.shape
	
	def size(self):
		return self.shape
		
	@property
	def ndim(self):
		return len(self.shape)
	
	@property
	def dtype(self):
		# feeling pretty damn lazy, will make a dedicated dtype class later maybe.
		# just maybe.
		return self.tg.dtype
	
	def cuda(device = None, non_blocking = False, memory_format = "torch.preserve_format"):
		if not device is None:
			raise NotImplementedError
		return self.to("cuda")
	
	def cpu(memory_format = "torch.preserve_format"):
		return self.to("cpu")
	
	def to(self, *args, **kwargs):
		assert len(args) > 0 or len(kwargs) > 0
		
		dtype = None
		device = None
		
		for arg in args:
			if isinstance(arg, tinygrad.dtype.DType):
				dtype = arg
			elif isinstance(arg, str):
				device = arg
		if "dtype" in kwargs.keys():
			dtype = kwargs["dtype"]
		if "device" in kwargs.keys():
			device = kwargs["device"]
		
		device = get_backend_override(device)
		assert device is None or (not "CUDA" in device)
		if dtype is None and (not device is None):
			new_tensor = self.tg.to(device)
		elif (not dtype is None) and device is None:
			new_tensor = self.cast(dtype)
		elif not (dtype is None or device is None):
			return super().to(device).cast(dtype)
			
		return convert_to_torch(new_tensor)
	
	@property
	def device(self):
		for frame_info in inspect.stack():
			pass#input(frame_info)
		# TODO: convert tinygrad device to torch device
		
		dev = tinygrad_device_to_torch_device(self.tg.device)
		return Device(dev)
	
	def _tg_override(self, *args, **kwargs):
		# Method for automatically wrapping stuff coded in tinygrad so
		# stuff works correctly
		
		# this will be the function that gets wrapped
		tg_attr = inspect.stack()[1].function
		
		# convert everything back to tinygrad.Tensor temporarily
		tg_self = self.tg
		tg_args = convert_to_tg(args)
		tg_kwargs = convert_to_tg(kwargs)
		
		if len(tg_kwargs) == 0:
			# fix for methods that don't support **kwargs
			output = tg_self.__getattribute__(tg_attr)(*tg_args)
		else:
			output = tg_self.__getattribute__(tg_attr)(*tg_args, **tg_kwargs)
		return convert_to_torch(output)
		
	
	
	def __add__(self, other):
		return self._tg_override(other)
	
	def __radd__(self, other):
		return self._tg_override(other)
		
	def __sub__(self, other):
		return self._tg_override(other)
		
	def __rsub__(self, other):
		return self._tg_override(other)
	
	def __mul__(self, other):
		return self._tg_override(other)
	
	def __rmul__(self, other):
		return self._tg_override(other)
	
	def __truediv__(self, other):
		return self._tg_override(other)
	
	def __rtruediv__(self, other):
		return self._tg_override(other)
		
	def numpy(self):
		return self.tg.numpy()
	
	def _reimplement_exact(self, function, *args, **kwargs):
		newself, args, kwargs = convert_to_tg(self, args, kwargs)
		return convert_to_torch(newself.__getattribute__(function)(*args, **kwargs) )
	
	def masked_fill(self, *args, **kwargs):
		args, kwargs = _disinherit(args, kwargs)
		return convert_to_tg(_disinherit(self).masked_fill(*args, **kwargs) )

	def argmax(self, *args, **kwargs):
		print(args, kwargs)
		return self._reimplement_exact("argmax", *args, **kwargs)
	
	def view(self, *shape):
		return self._reimplement_exact("view", *shape)
	
	def transpose(self, *args, **kwargs):
		return self._reimplement_exact("transpose", *args, **kwargs)
	
	def reshape(self, *args, **kwargs):
		return self._reimplement_exact("reshape", *args, **kwargs)
	
	def cast(self, dtype):
		# is this even a torch function? I don't know :c
		return AdapterTensor(self.tg.cast(dtype) )
	
	def expand(self, *args, **kwargs):
		return self._reimplement_exact("expand", *args, **kwargs)
		
	def __getitem__(self, *args, **kwargs):
		return self._reimplement_exact("__getitem__", *args, **kwargs)
	
def convert_to_torch(*inp):
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, AdapterTensor):
		return inp
	if isinstance(inp, tinygrad.Tensor):
		return AdapterTensor(inp)
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(convert_to_torch(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = convert_to_torch(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = convert_to_torch(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp
	
def convert_to_tg(*inp):
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, AdapterTensor):
		return inp.tg
	if isinstance(inp, tinygrad.Tensor):
		# do nothing
		return inp
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(convert_to_tg(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		for elem in new:
			assert not isinstance(elem, AdapterTensor)
			
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = convert_to_tg(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = convert_to_tg(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp
