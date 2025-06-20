import tinygrad

def tg_device_supports_longlong(dev: str, big_shp = (1, 77, 49408, 768) ) -> bool:
	# The test will make a big enough tensor that requires long long to index
	# If a compile error is thrown, then we return false.
	# If it succeeds, we return true
	a = tinygrad.Tensor.randn(big_shp)
	try:
		a.realize()
	except tinygrad.device.CompileError:
		return False
	del a
	return True
