import tinygrad
import gc

# memory before tensor creation
input("Note memory thingy")

# something large enough to be visible in system memory
tensor_elem_size = 1024*1024*512//4
a = tinygrad.Tensor.randn(tensor_elem_size, device = "CPU").realize()

input("Tensor has been created. Note python's memory consumption.")

for k in list(tinygrad.ops.buffers.keys() ):
	#tinygrad.ops.buffers[k].deallocate()
	buf = tinygrad.ops.buffers[k]
	buf._free(buf._buf, buf.options)
	#del tinygrad.ops.buffers[k]

# after this point, the memory does not change for me :c
input("note python's memory consumption again here")
