import tinygrad
import numpy as np
from .tensor import _convert_base as _cb

def interpolate(inp,
		size=None,
		scale_factor=None,
		mode='nearest',
		align_corners=None,
		recompute_scale_factor=None,
		antialias=False
		):
	if align_corners is None:
		align_corners = False
	if recompute_scale_factor is True:
		# not dealing with this crap for now
		raise NotImplementedError
	if antialias:
		# or this crap either lol
		raise NotImplementedError
	
	size = list(inp.shape)
	len_size = len(size)
	if not scale_factor is None:
		if isinstance(scale_factor, tuple):
			assert len(scale_factor) == len(size) - 2
			for i, sf in enumerate(scale_factor):
				size[i+2] = int(size[i+2]*scale_factor)
		else:
			for i in range(len_size - 2):
				print(size[i+2]*scale_factor)
				size[i+2] = int(size[i+2] * scale_factor)
		size = tuple(size)
	else:
		assert isinstance(size, tuple)
	return _cb( inp.interpolate(size, mode, align_corners) )
	
def group_norm(x, num_groups, weight = None, bias = None, eps = 1.0e-5):
	# derived from the tinygrad source code c:
	x = x.reshape(x.shape[0], num_groups, -1).layernorm(eps=eps).reshape(x.shape)

	if weight is None or bias is None: return x
	# elementwise_affine on channels
	return _cb(x * weight.reshape(1, -1, *[1] * (x.ndim-2)) + bias.reshape(1, -1, *[1] * (x.ndim-2)) )
	
def scaled_dot_product_attention(query, key, value, attn_mask=None,
		dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> tinygrad.Tensor:
	if enable_gqa:
		# not ever sure what this is
		raise NotImplementedError
	if not scale is None:
		# divide the custom scale factor by the internal scale factor
		internal_scale = 1.0 / np.sqrt(query.size(-1))
		scale_factor = scale/internal_scale
		
		# then multiply the query by it
		query = query * scale_factor
	return _cb( query.scaled_dot_product_attention(key, value, attn_mask,
		dropout_p, is_causal) )

def pad(inp, pad, mode='constant', value=None):
	if value is None:
		value = 0.0
	if mode != "constant":
		raise NotImplementedError
	return _cb(inp.pad(pad, mode, value) )
	
def embedding(inp, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
	raise NotImplementedError

gelu = lambda x: _cb(x.gelu() )
mish = lambda x: _cb(x.mish() )
sigmoid = lambda x: _cb(x.sigmoid() )


def cumprod(inp, dim, dtype=None, out=None):
	out = None
	
	# first, get the slices used in the __getitem__ call for each element
	slices = []
	for i in range(len(inp.shape)):
		slices.append(slice(None, None, None) )
	
	for i in range(inp.shape[dim] ):
		slices[dim] = i
		if out is None:
			out = inp[slices]
		else:
			out = out*inp[slices]
	return _cb(out)
