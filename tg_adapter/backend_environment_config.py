# Allows user override of which backends torch.Tensor.cuda(), torch.Tensor.cuda(), etc. point to

import os

_ENVIRON = {
	"TGA_ALL_ACCELERATOR_OVERRIDE" : "GPU", # just for now hehe
	
	"TGA_CUDA_OVERRIDE" : "CUDA",
	"TGA_CPU_OVERRIDE" : "CPU",
	"TGA_MPS_OVERRIDE" : "METAL"
}

_ENV_TABLE = {
	"CUDA" : "TGA_CUDA_OVERRIDE",
	"CPU" : "TGA_CPU_OVERRIDE",
	"MPS" : "TGA_MPS_OVERRIDE"
}

for key in _ENVIRON.keys():
	if key in os.environ.keys():
		_ENVIRON[key] = os.environ[key]

# Override all backend-specific garbage with whatever backend the user specifies
if not _ENVIRON["TGA_ALL_ACCELERATOR_OVERRIDE"] is None:
	for key in _ENV_TABLE.keys():
		if not key == "CPU":
			_ENV_TABLE[key] = "TGA_ALL_ACCELERATOR_OVERRIDE"

def _get_true_backend(backend):
	return _ENVIRON[_ENV_TABLE[backend] ]

def get_backend_override(backend):
	if backend is None:
		return backend
	backend = backend.upper()
	index = 0
	if ":" in backend:
		backend, index = tuple(backend.split(":") )
		index = int(index)
	backend = _get_true_backend(backend)
	if index > 0:
		return f"{backend}:{index}"
	return backend
		
def tinygrad_device_to_torch_device(device):
	idx = None
	name = device
	if ":" in device:
		name, idx = tuple(device.split(":"))
		idx = int(idx)
	for ek, ev in _ENVIRON.items():
		print(f"ek: {ek}, ev: {ev}, device: {device}")
		if ev == device:
			for etk, etv in _ENV_TABLE.items():
				if ek == etv:
					name = etk.lower()
					if not idx is None:
						name = f"{name}:{idx}"
					print(device, name, idx)
					input()
					return name
	
	raise ValueError
