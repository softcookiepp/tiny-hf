import tinygrad

def _device_supports_type(tg_device: str, dt: tinygrad.dtype.DType):
	try:
		t = tinygrad.Tensor.randn(1, device = tg_device, dtype = dt).realize().numpy()
		return True
	except tinygrad.device.CompileError:
		return False
	except ZeroDivisionError:
		# WebGPU requires itemsize to not be zero somewhere in its code,
		# and dtypes.void has an itemsize of 0
		return False
	except KeyError:
		return False
		
def iter_tg_dtypes():
	already_done = []
	for k, maybe_dtype in tinygrad.dtypes.__dict__.items():
		if isinstance(maybe_dtype, tinygrad.dtype.DType) and (not maybe_dtype in already_done):
			# deduplicate, then yield
			already_done.append(maybe_dtype)
			yield maybe_dtype

def probe_tg_dtypes(tg_device: str):
	supported_dtypes = []
	unsupported_dtypes = []
	for dt in iter_tg_dtypes():
		# this is where we probe
		if _device_supports_type(tg_device, dt):
			supported_dtypes.append(dt)
		else:
			unsupported_dtypes.append(dt)
	return supported_dtypes, unsupported_dtypes

if __name__ == "__main__":
	dev = "WEBGPU:0"
	supported, unsupported = probe_tg_dtypes(dev)
	
	print(f"Supported dtypes for {dev}")
	for dt in supported:
		print(dt)
		
	print(f"\nUnsupported dtypes for {dev}")
	for dt in unsupported:
		print(dt)
