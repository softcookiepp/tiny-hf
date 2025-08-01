# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import argparse

import tg_adapter as torch
from huggingface_hub import hf_hub_download

from tiny_hf.transformers.import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    VideoLlavaConfig,
    VideoLlavaForConditionalGeneration,
    VideoLlavaImageProcessor,
    VideoLlavaProcessor,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/video_llava/convert_video_llava_weights_to_hf.py --text_model_id lmsys/vicuna-7b-v1.5 --vision_model_id openai/clip-vit-large-patch14 --output_hub_path org/video_llava-7b --old_state_dict_id LanguageBind/Video-LLaVA-7B

Example for creating the old state dict file with Python:

    import tg_adapter as torch
    from video_llava.model.language_model.video_llava import VideoLlavaForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = VideoLlavaForCausalLM.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/video_llava-7b/model_state_dict.bin")
"""

KEYS_TO_MODIFY_MAPPING = {
    "model.video_tower.video_tower": "video_tower",
    "model.image_tower.image_tower": "image_tower",
    "model.mm_projector": "multi_modal_projector",
    "model": "language_model.model",
    "lm_head": "language_model.lm_head",
    "video_tower": "video_tower.vision_model",
    "image_tower": "image_tower.vision_model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
}


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def convert_video_llava_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_tokens(AddedToken("<video>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"

    image_processor = VideoLlavaImageProcessor.from_pretrained(vision_model_id)

    processor = VideoLlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    config = VideoLlavaConfig(text_config=text_config)
    config.pad_token_id = 32002

    with torch.device("meta"):
        model = VideoLlavaForConditionalGeneration(config)

    model_state_dict = set(model.state_dict().keys())

    # Pad to 64 for performance reasons
    pad_shape = 64
    state_dict_temp = "pytorch_model-0000{i}-of-00002.bin"
    for shard in range(1, 3):
        state_dict_path = hf_hub_download(old_state_dict_id, state_dict_temp.format(i=shard))
        state_dict = torch.load(state_dict_path, map_location="cpu")
        state_dict = convert_state_dict_to_hf(state_dict)
        model.load_state_dict(state_dict, strict=False, assign=True)
        model_state_dict -= set(state_dict.keys())

    if len(model_state_dict) > 0:
        raise RuntimeError(f"Missing keys in state dict: {model_state_dict}")

    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image and video token so we resize the model
    model.resize_token_embeddings(config.text_config.vocab_size + 3, pad_shape)
    model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
        dim=0,
    )
    model.language_model.lm_head.weight.data[32000:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
        dim=0,
    )

    model.push_to_hub(output_hub_path)
    processor.push_to_hub(output_hub_path)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_model_id",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_hub_path",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_state_dict_id",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    args = parser.parse_args()
    convert_video_llava_llama_to_hf(
        args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id
    )


if __name__ == "__main__":
    main()
