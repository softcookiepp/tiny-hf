# tiny-hf

This is a project that attempts to port Huggingface's Diffusers and
Transformers libraries to Tinygrad, a more cross-vendor compatible ML framework.

## Usage
- no instructions yet :c

## Development

The way the internals work is by providing a layer (tg_adapter) that helps
more easily convert pytorch code into tinygrad code.
The primary focus as of now is to make this layer more comprehensive enough that it
can be a drop-in replacement pytorch, at least for all the functionality used in diffusers and transformers.

Further on it may be a priority to convert all the stuff to tinygrad code, but this initial approach will be much easier to create and validate!
