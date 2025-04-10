from .backend_environment_config import get_backend_override, tinygrad_device_to_torch_device



class device:
	def __init__(self, name):
		self._name = name
		self._idx = 0
		if ":" in name:
			self._name, self._idx = tuple(name.split(":"))
			self._idx = int(self._idx)
		real_name = f"{get_backend_override(self._name)}:{self._idx}"
	
	@property
	def type(self):
		# pretty sure this is how it is done
		return self._name
	
	@property
	def tg(self):
		# Tinygrad device corresponding to this one
		raise NotImplementedError
