import torch
import tg_adapter
from testing_utils import *


def _test_geglu(use_bias = True):
	from tiny_hf.diffusers.models.activations import GEGLU as tg_class
	from diffusers.models.activations import GEGLU as hf_class
	
	hf_module = hf_class(8, 4, use_bias)
	tg_module = tg_class(8, 4, use_bias)
	copy_state_dict(hf_module, tg_module)
	a = np.arange(64*8).reshape(64, 8).astype(np.float32)*100
	input(a)
	test_hf_reimplementation([a], {}, hf_module, "__call__", tg_module, "__call__")

def test_geglu():
	_test_geglu(True)
	_test_geglu(False)
	

def test_modules():
	test_geglu()
