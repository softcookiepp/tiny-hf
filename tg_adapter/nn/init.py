import tinygrad
from ..tensor import _convert_base as _cb
from ..tensor import _disinherit

def uniform_(tensor, a = 0.0, b = 1.0, generator = None):
	tensor = _disinherit(tensor)
	if not generator is None:
		raise NotImplementedError
	uni = tinygrad.Tensor.uniform(*tensor.shape, low = a, high = b,
		dtype = tensor.dtype, requires_grad = tensor.requires_grad,
		device = tensor.device)
	return _cb(tensor.assign(uni) )

uniform = uniform_

# hopefully this works
def normal_(tensor, mean = 0.0, std = 1.0, generator = None):
	tensor = _disinherit(tensor)
	if not generator is None:
		raise NotImplementedError
	norm = tinygrad.Tensor.normal(*tensor.shape, mean, std,
		tensor.requires_grad,
		dtype = tensor.dtype,
		device = tensor.device)
	return _cb(tensor.assign(norm) )

normal = normal_

def trunc_normal_(tensor, mean = 0.0, std = 1.0, a = -2.0, b = 2.0, generator = None):
	tensor = _disinherit(tensor)
	if not generator is None:
		raise NotImplementedError
	norm = tinygrad.Tensor.normal(*tensor.shape, mean, std,
		tensor.requires_grad,
		dtype = tensor.dtype,
		device = tensor.device)
	norm = norm.clamp(a, b)
	return _cb(tensor.assign(norm) )

def constant_(tensor, val):
	tensor = _disinherit(tensor)
	full = tensor.full_like(val)
	return _cb(tensor.assign(full) )

def xavier_uniform_(tensor, *args, **kwargs):
	raise NotImplementedError

xavier_uniform = xavier_uniform_

def xavier_normal_(tensor, *args, **kwargs):
	raise NotImplementedError

xavier_normal = xavier_normal_

def kaiming_uniform_(tensor, *args, **kwargs):
	raise NotImplementedError

kaiming_uniform = kaiming_uniform_

def kaiming_normal_(tensor, *args, **kwargs):
	raise NotImplementedError

kaiming_normal = kaiming_normal_
