import tinygrad

from .backend_environment_config import get_backend_override, tinygrad_device_to_torch_device
from .device import device as Device
import inspect
import numpy as np

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
		tg_self = self.tg
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

def convert_to_tg(*args):
	
