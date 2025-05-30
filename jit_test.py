import tinygrad
import inspect
import os

def is_jitted():
	for item in inspect.stack():
		if os.path.basename(item.filename) == "jit.py" and item.function == "__call__":
			return True
	return False

def test_function():
	print("Function jitted:", is_jitted() )
	return tinygrad.Tensor.arange(4).realize()

@tinygrad.TinyJit
def run_test_function():
	return test_function()


if __name__ == "__main__":
	test_function()
	run_test_function()
