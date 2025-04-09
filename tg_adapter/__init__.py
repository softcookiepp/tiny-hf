from .backend_environment_config import *

from . import nn, F
from .generator import *
from .lambdas import *
from .types import *
from .tensor_constructors import *
from . import distributed
from . import version
from . import compiler
from . import cuda

from .io import *

import tinygrad
from .tensor import AdapterTensor
from .tensor import _convert_base as _cb

# aliases to twist shit into working at least somewhat
FloatTensor = AdapterTensor
LongTensor = AdapterTensor
IntTensor = AdapterTensor
BoolTensor = AdapterTensor
Tensor = AdapterTensor


# reimplementation of torch.cat,
# since it is more convenient than tinygrad's cat method
def cat(tensors, dim = 0):
	tbase = tensors[0]
	trest = tuple(tensors[1:])
	return _cb(tbase.cat(*trest, dim = dim) )



dtype = tinygrad.dtype.DType

#def device(dev):
#	dev = dev.upper()
#	return dev

# TODO: make an actual device class, then rejigger the state dict loader to do some other stuff
from .device import device

# @torch.no_grad()
# may also have to call @Tensor.train(mode = False)
no_grad = Tensor.test

def is_grad_enabled():
	# pretty sure this will work
	return not AdapterTensor.no_grad

def is_tensor(a):
	return isinstance(a, AdapterTensor)

Size = tuple

__version__ = "2.6.0"

from .F import chunk, clamp
