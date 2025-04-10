from .tensor import AdapterTensor as T

exp = lambda x: T( x.tg.exp() )
pow = lambda x, y: T( _disinherit(x).pow(_disinherit(x)) )
sum = lambda x, axis: T( _disinherit(x).sum(axis) )
sin = lambda x: T( x.tg.sin() )
cos = lambda x: T( _disinherit(x).cos() )
tan = lambda x: T( _disinherit(x).tan() )

from .F import sigmoid


from .F import cumprod
