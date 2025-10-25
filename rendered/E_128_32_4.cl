__kernel void E_128_32_4(__global float* data0_16384, __global float* data1_16384) {
  int gidx0 = get_group_id(0); /* 128 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  float4 val0 = (*((__global float4*)((data1_16384+alu0))));
  *((__global float4*)((data0_16384+alu0))) = val0;
}