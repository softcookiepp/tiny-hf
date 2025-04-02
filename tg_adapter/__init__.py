from .backend_environment_config import *

from . import nn, F
from .generator import *
from .lambdas import *
from .types import *
from .tensor_constructors import *
from . import distributed
from . import version

from .io import *

import tinygrad
from tinygrad import Tensor
from .tensor import _convert_base as _cb

# aliases to twist shit into working at least somewhat
FloatTensor = Tensor
LongTensor = Tensor
BoolTensor = Tensor

# reimplementation of torch.cat,
# since it is more convenient than tinygrad's cat method
def cat(tensors, dim = 0):
	tbase = tensors[0]
	trest = tuple(tensors[1:])
	return _cb(tbase.cat(*trest, dim = dim) )

# easier than rearranging huggingface code lol
def chunk(inp, chunks: int, dim: int = 0):
	return _cb(inp.chunk(chunks, dim) )
	
def clamp(inp, min = None, max = None):
	return _cb(inp.clamp(min, max) )

dtype = tinygrad.dtype.DType

def device(dev):
	dev = dev.upper()
	return dev

# @torch.no_grad()
# may also have to call @Tensor.train(mode = False)
no_grad = Tensor.test

def is_grad_enabled():
	# pretty sure this will work
	return not Tensor.no_grad

def is_tensor(a):
	return isinstance(a, Tensor)

Size = tuple

__version__ = "2.6.0"
