import tinygrad
import tg_adapter as tga
from tg_adapter import F
from .normalization import RMSNorm
from typing import Optional

class Upsample2D(tga.nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            self.norm = tinygrad.nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = tinygrad.nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = tinygrad.nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states: tinygrad.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> tinygrad.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            #deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            #deprecate("scale", "1.0.0", deprecation_message)
            pass

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        
        # can't guarantee that bf16 will be available on all devices, so disabling for now
        """
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16 until PyTorch 2.1
        # https://github.com/pytorch/pytorch/issues/86679#issuecomment-1783978767
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(torch.float32)
		"""

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        # looks like this still works with tinygrad...though I am unsure if it will be required
        # will keep it in for now just in case
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            # upsample_nearest_nhwc also fails when the number of output elements is large
            # https://github.com/pytorch/pytorch/issues/141831
            scale_factor = (
                2 if output_size is None else max([f / s for f, s in zip(output_size, hidden_states.shape[-2:])])
            )
            if hidden_states.numel() * scale_factor > pow(2, 31):
                hidden_states = hidden_states.contiguous()

            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        
        
        """
        # Cast back to original dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(dtype)
		"""

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        # looks like that will not be the case c:
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states
