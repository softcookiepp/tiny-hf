# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The implementation is based on "Parameter-Efficient Orthogonal Finetuning
# via Butterfly Factorization" (https://huggingface.co/papers/2311.06243) in ICLR 2024.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from tiny_hf.peft.config import PeftConfig
from tiny_hf.peft.utils import PeftType


@dataclass
class BOFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`BOFTModel`].

    Args:
        boft_block_size (`int`): BOFT block size across different layers.
        boft_block_num (`int`): Number of BOFT blocks per injected layer.
        boft_n_butterfly_factor (`int`): Number of butterfly factors across different layers.
        target_modules (`Union[List[str],str]`): The names of the modules to apply the adapter to.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        boft_dropout (`float`):
            The multiplicative dropout probability, by setting OFT blocks to identity during training, similar to the
            dropout layer in LoRA.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
            For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set
            to `True`.
        bias (`str`): Bias type for BOFT. Can be 'none', 'all' or 'boft_only'. If 'all' or 'boft_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):List of modules apart from BOFT layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the BOFT transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the BOFT
            transformations on the layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the model, which is
            often called `'layers'` or `'h'`.
    """

    boft_block_size: int = field(
        default=4,
        metadata={
            "help": "BOFT block size across different layers.",
            "note": "You can only specify either boft_block_size or boft_block_num, but not both simultaneously, because boft_block_size x boft_block_num = layer dimension.",
        },
    )
    boft_block_num: int = field(
        default=0,
        metadata={
            "help": "Number of BOFT blocks per injected layer.",
            "note": "You can only specify either boft_block_size or boft_block_num, but not both simultaneously, because boft_block_size x boft_block_num = layer dimension.",
        },
    )
    boft_n_butterfly_factor: int = field(
        default=1,
        metadata={
            "help": "Number of butterfly factors.",
            "note": (
                "for example, boft_n_butterfly_factor=2, the effective block size of OFT becomes twice as big and the number of blocks become half.",
                "note: for boft_n_butterfly_factor=1, BOFT is the same as vanilla OFT.",
            ),
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with BOFT.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from BOFT."},
    )
    boft_dropout: float = field(
        default=0.0,
        metadata={
            "help": "BOFT multiplicative dropout, randomly setting blocks of OFT to be identity matrix, similar to the dropout layer in LoRA."
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for BOFT. Can be 'none', 'all' or 'boft_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from BOFT layers to be set as trainable and saved in the final checkpoint. ",
            "note": (
                "For example, in Sequence Classification or Token Classification tasks, ",
                "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved.",
            ),
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the BOFT layers with their default initialization. Don't change ",
                "this setting, except if you know exactly what you're doing.",
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.BOFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if self.boft_block_size == 0 and self.boft_block_num == 0:
            raise ValueError(
                f"Either `boft_block_size` or `boft_block_num` must be non-zero. Currently, boft_block_size = {self.boft_block_size} and boft_block_num = {self.boft_block_num}."
            )
        if not (self.boft_block_size != 0) ^ (self.boft_block_num != 0):
            raise ValueError(
                f"You can only specify either boft_block_size ({self.boft_block_size}) or boft_block_num ({self.boft_block_num}), but not both simultaneously, because boft_block_size x boft_block_num == in_features."
            )
