import tinygrad
from dataclasses import dataclass
from .device import backend_from_device
import numpy as np

TYPE_KEYS = ["float32", "float64", "complex64", "complex128", "float16",
	"uint8", "int8", "int16", "int32", "int64", "bool", "bfloat16",
	"uint16", "uint32", "uint64", "float8_e4m3fn", "float8_e5m2"]
	
FLOAT_KEYS = ["float32", "float64", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"]

# Not all tinygrad backends support all data types.
# So we need to create a map that overrides the types somehow...
TG_BACKEND_TYPE_MAP = {
	"DEFAULT" : {
		"float32": tinygrad.dtypes.float,
		"float64": tinygrad.dtypes.double,
		"complex64": tinygrad.dtypes.float, # not implemented in tinygrad, will be a pain in the ass :c
		"complex128" : tinygrad.dtypes.double, # same as above
		"float16": tinygrad.dtypes.float16,
		"uint8": tinygrad.dtypes.uchar,
		"int8" : tinygrad.dtypes.int8,
		"int16": tinygrad.dtypes.int16,
		"int32": tinygrad.dtypes.int32,
		"int64": tinygrad.dtypes.int64,
		"uint16": tinygrad.dtypes.uint16,
		"uint32": tinygrad.dtypes.uint32,
		"uint64": tinygrad.dtypes.uint64,
		"bool": tinygrad.dtypes.bool,
		"bfloat16" : tinygrad.dtypes.bfloat16,
		"float8_e4m3fn": tinygrad.dtypes.int8, # no idea how this garbage will work
		"float8_e5m2": tinygrad.dtypes.int8 # no idea how this garbage will work
	},
	
	# only the types to be overridden must be present c:
	"WEBGPU" : {
		"int64": tinygrad.dtypes.int32,
		"float16": tinygrad.dtypes.float32,
		"float64": tinygrad.dtypes.float32
	}
}

NP_TG_TYPE_MAP = {
	tinygrad.dtypes.float: np.dtype("float32"),
	tinygrad.dtypes.double: np.dtype("float64"),
	tinygrad.dtypes.float16: np.dtype("float16"),
	tinygrad.dtypes.uchar: np.dtype("uint8"),
	tinygrad.dtypes.int8: np.dtype("int8"),
	tinygrad.dtypes.int16: np.dtype("int16"),
	tinygrad.dtypes.int32: np.dtype("int32"),
	tinygrad.dtypes.int64: np.dtype("int64"),
	tinygrad.dtypes.uint16: np.dtype("uint16"),
	tinygrad.dtypes.uint32: np.dtype("uint32"),
	tinygrad.dtypes.uint64: np.dtype("uint64"),
	tinygrad.dtypes.bool: np.dtype("bool"),
	tinygrad.dtypes.bfloat16: np.dtype("float16") # no numpy equivalent :c
}

def get_np_type_from_tg(tgt):
	return NP_TG_TYPE_MAP[tgt]

def get_tg_type_from_np(npt):
	for k, v in NP_TG_TYPE_MAP.items():
		if v == npt:
			return k
	raise KeyError

def convert_np_type_correctly(array, backend):
	tgt = get_tg_type_from_np(array.dtype)
	dt = get_type_from_tg(tgt, backend)
	tgt = dt.tgt(backend)
	npt = get_np_type_from_tg(tgt)
	return array.astype(npt)

_type_aliases = {
	"float": "float32",
	"double": "float64",
	"cfloat": "complex64",
	"cdouble": "complex128",
	"half": "float16",
	"short": "int16",
	"int": "int32",
	"long": "int64"
}

_types_map = {}

def parse_alias(attr):
	if attr in _type_aliases.keys():
		attr = _type_aliases[attr]
	return attr

def get_type_from_tg(tg_type, backend, other_type = None):
	type_key = None
	otk = ""
	if not other_type is None:
		if isinstance(other_type, str):
			otk = other_type
		else:
			otk = other_type.key
	if backend in TG_BACKEND_TYPE_MAP.keys():
		for k, v in TG_BACKEND_TYPE_MAP[backend].items():
			if v == tg_type:
				type_key = k
				if otk == k:
					break
	if type_key is None:
		# wasn't in any of the overrides, should be in default map
		for k, v in TG_BACKEND_TYPE_MAP["DEFAULT"].items():
			if v == tg_type:
				type_key = k
				if otk == k:
					break
	if type_key is None:
		raise ValueError
	return _types_map[type_key]

class dtype:
	def __init__(self, key):
		self._key = parse_alias(key)
		self._is_complex = False
		if "complex" in key:
			self._is_complex = True
		self._is_fp = self._key in FLOAT_KEYS
	
	@property
	def is_complex(self):
		return self._is_complex
	
	@property
	def key(self):
		return self._key
	
	@property
	def is_floating_point(self):
		return self._is_fp
	
	def tgt(self, tg_backend = None):
		dt = None
		tg_backend = backend_from_device(tg_backend)
		if tg_backend in TG_BACKEND_TYPE_MAP.keys() and self._key in TG_BACKEND_TYPE_MAP[tg_backend].keys():
			dt = TG_BACKEND_TYPE_MAP[tg_backend][self._key]
		else:
			dt = TG_BACKEND_TYPE_MAP["DEFAULT"][self._key]
		if dt is None:
			print(self._key)
			raise ValueError
		return dt
	
	def __eq__(self, other):
		if hasattr(other, "tgt"):
			other = other.tgt()
		return other == self.tgt()
	
def get_tgt(t, tg_backend):
	if t is None:
		return None
	return t.tgt(tg_backend)

for k in TYPE_KEYS:
	_types_map[k] = dtype(k)

def _get_type(attr):
	attr = parse_alias(attr)
	if attr in _types_map.keys():
		return _types_map[attr]
	return None

def get_default_dtype():
	# is this right? idk
	return _types_map["float32"]

# placeholder for now hehe
def set_default_dtype(dtype):
	pass

@dataclass
class FInfo:
	min: int
	max: int
	# TODO: add more
	
FINFO_MAP = {}	
for dtk in TYPE_KEYS:
	dt = _get_type(dtk)
	FINFO_MAP[dtk] = FInfo(tinygrad.dtypes.min(dt.tgt() ), tinygrad.dtypes.max(dt.tgt() ) )
	
def finfo(t):
	if isinstance(t, dtype):
		t = t.key
	return FINFO_MAP[t]

def is_floating_point(data):
	return data.dtype.is_floating_point
