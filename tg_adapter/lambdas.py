from .tensor import _convert_base as _cb
from .tensor import _disinherit

exp = lambda x: _cb( _disinherit(x).exp() )
pow = lambda x, y: _cb( _disinherit(x).pow(_disinherit(x)) )
sum = lambda x, axis: _cb( _disinherit(x).sum(axis) )
sin = lambda x: _cb( _disinherit(x).sin() )
cos = lambda x: _cb( _disinherit(x).cos() )
tan = lambda x: _cb( _disinherit(x).tan() )

from .F import sigmoid


from .F import cumprod
