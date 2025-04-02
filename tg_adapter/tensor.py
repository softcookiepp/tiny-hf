import tinygrad

class AdapterTensor(tinygrad.Tensor):
	def __init__(self, data, dtype = None, device = None,
			requires_grad = False, pin_memory = False):
		# pin memory is unused, but kept for compatibility
		super().__init__(data, device, dtype, requires_grad)
	
	
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
		
		if dtype is None and (not device is None):
			new_tensor = self.to(device)
		elif (not dtype is None) and device is None:
			new_tensor = self.cast(dtype)
		elif not (dtype is None or device is None):
			new_tensor
			
		return _convert_base(new_tensor)
	"""
	def cast(self, *args, **kwargs):
		raise NotImplementedError
	"""
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
