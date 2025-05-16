import tinygrad
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, torch_load
from typing import Iterable
import inspect
from ..device import device
from ..utils import recursive_realize
from ..tensor import AdapterTensor as AT
from ..tensor import convert_to_torch, _parse_to_arguments

# adapter for https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class Module:
	__modules = {}
	def __init__(self, *args, **kwargs):
		self._train = True
		self._buffers = {}
	
	def add_module(self, name: str, module = None):
		assert not name in self.__dict__.keys() or self.__getattribute__(name) == module
		self.__dict__[name] = module
		
		
		# pretty sure this is just to check to see if it shits itself
		self.__getattribute__(name)
	
	def apply(self, fn):
		# fn should be a function that accepts a single module as an argument
		fn(self)
		for _, module in self.named_modules():
			fn(module)
	
	def buffers(self):
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				for b in v.buffers():
					yield b
			elif isinstance(v, tinygrad.Tensor):
				# generally no idea what criteria is supposed to be used for this :c
				pass
	
	def bfloat16(self):
		raise NotImplementedError

	def named_children(self):
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				yield k, v

	def children(self):
		# Return an iterator over immediate children modules.
		for _, v in self.named_children():
			yield v
	 
	def parameters(self, recurse = True):
		if not recurse:
			raise NotImplementedError
		#params = []
		for k, v in self.named_parameters():
			#params.append(v)
			yield v
		if isinstance(self, Iterable):
			for i, elem in enumerate(self):
				for k, mod in elem.named_parameters():
					#params.append(mod)
					yield mod
		#return params
		"""
		params = []
		for k, v in enumerate(self):
			print(v)
			if isinstance(v, tinygrad.Tensor):
				params.append(v)
			elif isinstance(v, list):
				for item in v:
					if isinstance(v, tinygrad.Tensor):
						params.append(v)
		for k, v in self.__dict__.items():
			print(k, v)
			if isinstance(v, tinygrad.Tensor):
				params.append(v)
			elif isinstance(v, list):
				for item in v:
					if isinstance(v, tinygrad.Tensor):
						params.append(v)
		for child in self.children():
			for param in self.children.params():
				params.append(param)
		return params
		"""

	def compile(self):
		# tinygrad has no compile equivalent lol
		pass
		
	def to(self, *args, **kwargs):
		# convert any floating point parameters to dtype
		# move parameters to device, convert unsupported types to supported types if required
		# how do we do this?
		
		dtype, device = _parse_to_arguments(*args, **kwargs)
		if dtype is None and device is None:
			raise ValueError
		assert dtype == None or dtype.is_floating_point
		for k, v in self.state_dict().items():
			if isinstance(v, AT) and v.is_floating_point():
				# typecast it
				v.to_(dtype, device)
			elif isinstance(v, AT) and not device is None:
				v.to_(device)
		return self
		
	def cuda(self, device = None):
		raise NotImplementedError
		
		
		
	def train(self, train = True):
		self._train = train
		
	def eval(self):
		self.train(False)
		
	def __call__(self, *args, **kwargs):
		# get parent function, check if it is not forward or __call__,
		# then invoke realize() if that is the case
		parent_function = inspect.stack()[1].function
		
		args, kwargs = convert_to_torch(args, kwargs)
		
		# gotta do this first hehe
		out = self.forward(*args, **kwargs)
		"""
		if False or ( (not parent_function in ["__call__", "forward"]) and isinstance(out, tinygrad.Tensor) ):
			out = recursive_realize(out)
		"""
		return out
		
	def forward(self, *args, **kwargs):
		raise NotImplementedError
	@property
	def _modules(self):
		modules = {}
		# immediate modules
		for k, v in self.__dict__.items():
			assert not v is self
			if isinstance(v, Module):
				modules[k] = v
			# TODO: reimplement modulelist
		self.__modules.update(modules)
		return self.__modules
	
	def _load_from_state_dict(self,
			state_dict,
			prefix,
			local_metadata,
			strict,
			missing_keys,
			unexpected_keys,
			error_msgs):
				
		# just because huggingface doesn't seem to understand that you REALLY SHOULD NOT BE USING PRIVATE METHODS
		# goodness gracious
		
		# refresh modules just in case...
		raise NotImplementedError
		
		for k, v in self.__dict__.items():
			full_key = prefix + k
			if full_key in state_dict.keys():
				sd_item = state_dict[full_key]
				if isinstance(sd_item, AT):
					raise NotImplementedError
				elif isinstance(sd_item, tinygrad.Tensor):
					print(type(v), type(sd_item) )
					v.tg.replace(state_dict[full_key].to(v.tg.device) ).realize()
	
	def _load_elem_state_dict_recursive(self, k, v, state_dict, prefix):
		if isinstance(v, Module):
			v._load_state_dict_recursive(state_dict, prefix = prefix)
		elif isinstance(v, AT):
			new_key = prefix.strip(".")
			if new_key in state_dict.keys():
				sd_item = state_dict[new_key]
				if isinstance(sd_item, tinygrad.Tensor):
					v.tg.replace(state_dict[new_key].to(v.tg.device) ).realize()
				elif isinstance(sd_item, AT):
					raise NotImplementedError
				else:
					print(type(sd_item) )
					raise ValueError
			else:
				# TODO: warn user or something, i forget
				pass
	
	def _load_state_dict_recursive(self, state_dict, prefix = ""):
		for k, v in self.__dict__.items():
			if isinstance(v, list):
				for i in range(len(v) ):
					vi = v[i]
					self._load_elem_state_dict_recursive(str(i), vi, state_dict, prefix = f"{prefix}.{k}.{i}")
			else:
				self._load_elem_state_dict_recursive(k, v, state_dict, f"{prefix}.{k}")
	
		
	def load_state_dict(self, state_dict, strict = True, assign = False, prefix = ""):
		
		"""
		for k, v in self.__dict__.items():
			if isinstance(v, list):
				for i in range(len(v) ):
					new_prefix = ".".join( [prefix, k, str(i)] )
		"""
		# actually keep doing it this way, using tinygrad's method is going to screw shit up once we get into ModuleList s
		#self._load_state_dict_recursive(state_dict)
		#return [], []

		# Ok so we actually have to check which parameters the provided state dict contains vs does not contain.
		# How do we do that?
		# It should
		missing_keys = []
		unexpected_keys = []
		
		own_state_dict = get_state_dict(self)
		
		new_state_dict = {}
		for k, v in state_dict.items():
			#print(k, v)
			k = k + "._tg"
			#input(k)
			new_state_dict[k] = v
		#input(new_state_dict.keys() )
		
		own_sd = get_state_dict(self)
		final_state_dict = {}
		for k in own_sd.keys():
			if k in new_state_dict.keys():
				# key exists, transfer parameters
				final_state_dict[k] = new_state_dict[k]
			else:
				# key is missing
				missing_keys.append(k[0:len(k) - 4])
		
		for k in new_state_dict.keys():
			if not k in own_sd.keys():
				unexpected_keys.append(k[0:len(k) - 4] )
		
		tinygrad.nn.state.load_state_dict(self, final_state_dict, strict = strict, verbose = True)
		return missing_keys, unexpected_keys
		
	
	def state_dict(self, prefix = ""):
		if False:
			#return _disinherit(tinygrad.nn.state.get_state_dict(self) )
			# Can no longer do that, as AdapterTensor objects are no longer
			# a subclass of tinygrad.Tensor.
			# we will have to make a dedicated method...
			state_dict = {}
			for k, v in self.__dict__.items():
				if isinstance(v, list):
					for i in range(len(v) ):
						l_prefix = ".".join([prefix, f"{k}.{i}"])
						if isinstance(v[i], Module):
							state_dict.update(v[i].state_dict(l_prefix) )
						
				elif isinstance(v, Module):
					new_prefix = prefix + f".{k}"
					state_dict.update(v.state_dict(new_prefix) )
				elif isinstance(v, AT):
					sd_key = ".".join([prefix, k]).strip(".")
					state_dict[sd_key] = v
			return state_dict
		else:
			pre_sd = get_state_dict(self)
			post_sd = {}
			for k, v in pre_sd.items():
				# convert to regular state dict from tinygrad
				new_key = k[0:len(k) - 4]
				new_v = AT(v)
				post_sd[new_key] = new_v
			return post_sd
	
	def __repr__(self):
		return f"{self.__class__}"
	
	def named_parameters(self, memo = None, prefix = "", remove_duplicate = True):
		#raise NotImplementedError
		for name, param in self.state_dict().items():
			yield name, param
			
	def named_modules(self, memo=None, prefix="", remove_duplicate=True):
		#raise NotImplementedError
		# prefix indicates this method is called recursively
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				k = prefix + k
				yield k, v
				for subk, subv in v.named_modules(prefix = f"{k}."):
					# use recursion c:
					yield subk, subv
	
	def modules(self, remove_duplicate = True):
		
		for k, v in self.named_modules(remove_duplicate = remove_duplicate):
			yield v
	
	def register_buffer(self, name, tensor, persistent = True):
		assert not name in self.__dict__.keys()
		self.__dict__[name] = tensor
		
	@property
	def training(self):
		# TODO: actually implement
		return False
