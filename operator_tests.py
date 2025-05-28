import torch
import tg_adapter
from testing_utils import *

def test_cumprod():
	from tg_adapter import F as tinyF
	inp = np.arange(5*4).reshape(5, 4).astype(np.float32)
	test_function( (inp, 0), {}, torch.cumprod, tinyF.cumprod)


def test_cat():
	a = make_test_data(40, 2, 5)
	b = make_test_data(2, 2, 5)
	test_function( ([a, b], 0), {}, torch.cat, tg_adapter.cat)

def test_interpolate():
	shape = (2, 3, 6, 6)
	a = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
	args = (a, None, 2.0)
	test_function(args, {}, torch_function = torch.nn.functional.interpolate, tinygrad_function = tg_adapter.F.interpolate)

def _test_unary(torch_function, tinygrad_function, data = None):
	if data is None:
		shape = (4, 2, 6, 8)
		data = make_test_data(*shape)
	test_function( (data), {}, torch_function, tinygrad_function)
	

def test_scaled_dot_product_attention():
	q = make_test_data(1, 8, 4096, 40)
	k = make_test_data(1, 8, 4096, 40)
	v = make_test_data(1, 8, 4096, 40)
	test_function( [q, k, v], {}, torch.nn.functional.scaled_dot_product_attention, tg_adapter.F.scaled_dot_product_attention)

def test_gelu():
	_test_unary(torch.nn.functional.gelu, tg_adapter.F.gelu, np.arange(1000).astype(np.float32) - 500.0 )

def test_all_operators():
	test_cumprod()
	test_cat()
	test_interpolate()
	test_scaled_dot_product_attention()
	test_gelu()
