from .backend_environment_config import torch_dev_to_tiny, tiny_dev_to_torch



class device:
	def __init__(self, name):
		self._name = name
		self._idx = None
		if ":" in name:
			self._name, self._idx = tuple(name.split(":"))
			self._idx = int(self._idx)
	
	@property
	def type(self):
		# pretty sure this is how it is done
		return self._name
	
	@property
	def name(self):
		return self._name
	
	@property
	def idx(self):
		return self._idx
	
	@property
	def tg(self):
		# Tinygrad device corresponding to this one
		return torch_dev_to_tiny(self._name, self._idx)
