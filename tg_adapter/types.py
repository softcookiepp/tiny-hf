import tinygrad
from dataclasses import dataclass

float32 = tinygrad.dtypes.float
float = float32

float64 = tinygrad.dtypes.double
bfloat16 = tinygrad.dtypes.bfloat16
float16 = tinygrad.dtypes.float16
long = tinygrad.dtypes.int32 #long, overriding because of dumb garbage
bool = tinygrad.dtypes.bool
uint8 = tinygrad.dtypes.uint8
int8 = tinygrad.dtypes.int8
int16 = tinygrad.dtypes.int16
int32 = tinygrad.dtypes.int32
int64 = tinygrad.dtypes.int64
uint16 = tinygrad.dtypes.uint16
uint32 = tinygrad.dtypes.uint32
uint64 = tinygrad.dtypes.uint64

# No support for this yet
float8_e4m3fn = int8
float8_e5m2 = int8

def get_default_dtype():
	# is this right? idk
	return float32

# placeholder for now hehe
def set_default_dtype(dtype):
	pass

@dataclass
class FInfo:
	min: int
	max: int
	# TODO: add more
	
FINFO_MAP = {}	
for dt in [float32, float64, bfloat16, float16, long, uint8, int8, int16, int32, int64, uint16, uint32, uint64]:
	FINFO_MAP[dt] = FInfo(tinygrad.dtypes.min(dt), tinygrad.dtypes.max(dt) )
	
def finfo(t):
	return FINFO_MAP[t]


# defining "int" later because it is already a reserved word
int = tinygrad.dtypes.int


class AdapterDType:
	def __init__(*args, **kwargs):
		# This will be a silly little wrapper around the existing tinygrad dtypes
		# to make them more torch-compatible
		raise NotImplementedError
		
