import torch
import tg_adapter


def import_non_recursive():
	from tests import make_test_data, test_function, test_hf_reimplementation


def test_cumprod():
	import_non_recursive()
	from tg_adapter import F as tinyF
	inp = np.arange(5*4).reshape(5, 4).astype(np.float32)
	test_function( (inp, 0), torch.cumprod, tinyF.cumprod)


def test_cat():
	import_non_recursive()
	a = make_test_data(40, 2, 5)
	b = make_test_data(2, 2, 5)
	test_function( ([a, b], 0), {}, torch.cat, tg_adapter.cat)

