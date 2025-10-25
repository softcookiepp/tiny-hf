__kernel void E_5_8_4n1(__global float* data0_160, __global float* data1_160) {
  int gidx0 = get_group_id(0); /* 5 */
  int lidx0 = get_local_id(0); /* 8 */
  int alu0 = ((gidx0<<5)+(lidx0<<2));
  float4 val0 = (*((__global float4*)((data1_160+alu0))));
  *((__global float4*)((data0_160+alu0))) = val0;
}