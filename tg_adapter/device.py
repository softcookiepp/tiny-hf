
class device:
	def __init__(self, name):
		self._name = name
		self._idx = 0
		if ":" in name:
			self._name, self._idx = tuple(name.split(":"))
			self._idx = int(self._idx)
	
	@property
	def type(self):
		# pretty sure this is how it is done
		return self._name
	
	def to_tinygrad(self):
		raise NotImplementedError
