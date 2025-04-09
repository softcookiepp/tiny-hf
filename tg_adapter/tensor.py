import tinygrad

from .backend_environment_config import get_backend_override, tinygrad_device_to_torch_device
from .device import device as Device
import inspect

class AdapterTensor(tinygrad.Tensor):
	def __init__(self, data, dtype = None, device = None,
			requires_grad = False, pin_memory = False):
		# pin memory is unused, but kept for compatibility
		# convert device, duh
		if isinstance(device, Device):
			raise NotImplementedError
		device = get_backend_override(device)
		if not device is None:
			if "CUDA" in device.upper():
				input("UH OH WE HAVE A COODA")
		super().__init__(data, device, dtype, requires_grad)
		if not device is None:
			assert not "CUDA" in super().device
		self._adapter_device = None
	
	def cuda(device = None, non_blocking = False, memory_format = "torch.preserve_format"):
		if not device is None:
			raise NotImplementedError
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
			new_tensor = super().to(device)
		elif (not dtype is None) and device is None:
			new_tensor = self.cast(dtype)
		elif not (dtype is None or device is None):
			return super().to(device).cast(dtype)
			
		return _convert_base(new_tensor)
	
	@property
	def device(self):
		for frame_info in inspect.stack():
			pass#input(frame_info)
		# TODO: convert tinygrad device to torch device
		assert not "CUDA" in super().device
		if self._adapter_device is None:
			dev = tinygrad_device_to_torch_device(super().device)
			self._adapter_device = Device(dev)
		return self._adapter_device
	
	def _tg_override(self, *args, **kwargs):
		# Method for automatically wrapping stuff coded in tinygrad so
		# stuff works correctly
		
		# this will be the function that gets wrapped
		tg_attr = inspect.stack()[1].function
		
		# convert everything back to tinygrad.Tensor temporarily
		tg_self = _disinherit(self)
		tg_args = _disinherit(args)
		tg_kwargs = _disinherit(kwargs)
		
		if len(tg_kwargs) == 0:
			# fix for methods that don't support **kwargs
			output = tg_self.__getattribute__(tg_attr)(*tg_args)
		else:
			output = tg_self.__getattribute__(tg_attr)(*tg_args, **tg_kwargs)
		return _convert_base(output)
	
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
		input(super().device)
		return _disinherit(self).numpy()
	
	def _reimplement_exact(self, function, *args, **kwargs):
		newself, args, kwargs = _disinherit(self, args, kwargs)
		print(newself, args, kwargs)
		print(type(newself) )
		input(newself.device)
		return _convert_base(newself.__getattribute__(function)(*args, **kwargs) )
	
	def masked_fill(self, *args, **kwargs):
		args, kwargs = _disinherit(args, kwargs)
		return _convert_base(_disinherit(self).masked_fill(*args, **kwargs) )

	def argmax(self, *args, **kwargs):
		print(args, kwargs)
		return self._reimplement_exact("argmax", *args, **kwargs)
	
def _convert_base(inp):
	if isinstance(inp, AdapterTensor):
		# do nothing
		return inp
	if isinstance(inp, tinygrad.Tensor):
		assert not "CUDA" in inp.device
		t = AdapterTensor(None)
		# oh god this is hacky
		t.lazydata = inp.lazydata
		assert isinstance(t, AdapterTensor)
		return t
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(_convert_base(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = _convert_base(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = _convert_base(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp

def _disinherit(*inp):
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, AdapterTensor):
		t = tinygrad.Tensor(None)
		t.lazydata = inp.lazydata
		assert not "CUDA" in t.device
		return t
	if isinstance(inp, tinygrad.Tensor):
		# do nothing
		assert not "CUDA" in inp.device
		return inp
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(_disinherit(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		for elem in new:
			assert not isinstance(elem, AdapterTensor)
			
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = _disinherit(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = _disinherit(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp

