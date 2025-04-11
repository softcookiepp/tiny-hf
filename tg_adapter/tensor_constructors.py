import tinygrad
from .types import get_default_dtype
from .tensor import AdapterTensor as AT
import numpy as np
from .backend_environment_config import torch_dev_to_tiny

def _convert_size(size):
	if len(size) == 0:
		return size
	if isinstance(size[0], tuple) or isinstance(size[0], list):
		size = size[0]
	return size

def ones(*size, out=None, dtype=None, layout = None, device=None, requires_grad=False): #layout=torch.strided,
	device = torch_dev_to_tiny(device)
	size = _convert_size(size)
	return AT(tinygrad.Tensor.ones(*size, device = device, dtype = dtype) )


def arange(start, end = None, step=1, out=None, dtype=None, layout="torch.strided", device=None, requires_grad=False):
	device = torch_dev_to_tiny(device)
	if end is None:
		end = start
		start = 0
	if dtype is None:
		dtype = get_default_dtype()
	return AT(tinygrad.Tensor.arange(start, end, step, dtype = dtype, device = device) )

def empty(size = None, out=None, dtype=None, layout="torch.strided", device=None,
		requires_grad=False, pin_memory=False, memory_format="torch.contiguous_format"):
	assert not size is None
	#device = get_backend_override(device)
	size = _convert_size(size)
	#input(device)
	return AT(tinygrad.Tensor.empty(*size) )

def full(size, fill_value, out=None, dtype=None, layout="torch.strided", device=None, requires_grad=False):
	#device = get_backend_override(device)
	device = torch_dev_to_tiny(device)
	return AT(tinygrad.Tensor.full(size, fill_value, device = device) )

def tensor(data, dtype = None, device = None,
			requires_grad = False, pin_memory = False):
	# get_backend_override is done internally for this one
	return AT(data, dtype, device,
			requires_grad, pin_memory)

def linspace(start, end, steps, *, out=None, dtype=None,
		layout="torch.strided", device=None, requires_grad=False):
	device = get_backend_override(device)
	#input(device)
	t = tinygrad.Tensor.linspace(start, end, steps, dtype = dtype, device = device)
	return AT(t)
	
def from_numpy(a: np.ndarray):
	return AT(a)
