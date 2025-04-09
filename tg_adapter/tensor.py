import tinygrad

from .backend_environment_config import get_backend_override

class AdapterTensor(tinygrad.Tensor):
	def __init__(self, data, dtype = None, device = None,
			requires_grad = False, pin_memory = False):
		# pin memory is unused, but kept for compatibility
		super().__init__(data, device, dtype, requires_grad)
	
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
		
		if dtype is None and (not device is None):
			new_tensor = super().to(device)
		elif (not dtype is None) and device is None:
			new_tensor = self.cast(dtype)
		elif not (dtype is None or device is None):
			return super().to(device).cast(dtype)
			
		return _convert_base(new_tensor)
	
	
def _convert_base(inp):
	if isinstance(inp, AdapterTensor):
		# do nothing
		return inp
	if isinstance(inp, tinygrad.Tensor):
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

	def _getitem(self, key):
		raise NotImplementedError
