from .tensor import AdapterTensor as T
from .tensor import convert_to_tg
from .tensor import assert_same_device

exp = lambda x: T( x.tg.exp() )

# non-unaries actually have to be a function because of
# same device checking
def pow(x, y):
	x, y = convert_to_tg(x, y)
	assert_same_device(x.device, x, y)
	return T( x.pow(y) )

sum = lambda x, axis: T( convert_to_tg(x).sum(axis) )
sin = lambda x: T( x.tg.sin() )
cos = lambda x: T( convert_to_tg(x).cos() )
tan = lambda x: T( convert_to_tg(x).tan() )

from .F import sigmoid


from .F import cumprod
