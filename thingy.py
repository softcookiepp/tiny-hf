import tinygrad

tensor_elem_size = 1024*1024*256//2
total_mem = 0

for i in range(1000):
  print(f"i = {i}")
  a = tinygrad.Tensor.randn(tensor_elem_size, device="CPU").realize()
  total_mem += a.lazydata.base.realized.nbytes
  print(f"total realized: {total_mem/(1024**3)} GB\n")
  del a
