__kernel void E_10_32_4n1(__global float* data0_1280, __global float* data1_1280) {
  int gidx0 = get_group_id(0); /* 10 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  float4 val0 = (*((__global float4*)((data1_1280+alu0))));
  *((__global float4*)((data0_1280+alu0))) = val0;
}