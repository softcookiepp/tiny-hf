from .tensor import AdapterTensor as T
from .tensor import convert_to_tg

exp = lambda x: T( x.tg.exp() )
pow = lambda x, y: T( convert_to_tg(x).pow(convert_to_tg(x)) )
sum = lambda x, axis: T( convert_to_tg(x).sum(axis) )
sin = lambda x: T( x.tg.sin() )
cos = lambda x: T( convert_to_tg(x).cos() )
tan = lambda x: T( convert_to_tg(x).tan() )

from .F import sigmoid


from .F import cumprod
