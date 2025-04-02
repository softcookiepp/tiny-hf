import tg_adapter as tga
torch = tga # heh
from tg_adapter import nn
import tinygrad

ACT2CLS = {
	"swish": lambda x: x.silu(),
	"silu": lambda x: x.silu(),
	"mish": lambda x: x.mish(),
	"gelu": lambda x: x.gelu(),
	"relu": lambda x: x.relu(),
}


def get_activation(act_fn: str):
	"""Helper function to get activation function from string.

	Args:
		act_fn (str): Name of activation function.

	Returns:
		nn.Module: Activation function.
	"""

	act_fn = act_fn.lower()
	if act_fn in ACT2CLS:
		return ACT2CLS[act_fn]()
	else:
		raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")

class FP32SiLU(nn.Module):
	r"""
	SiLU activation function with input upcasted to torch.float32.
	"""

	def __init__(self):
		super().__init__()

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return F.silu(inputs.cast(tinygrad.dtypes.float32), inplace=False).to(inputs.dtype)
