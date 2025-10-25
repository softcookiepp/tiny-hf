__kernel void E_20480_32_4(__global float* data0_2621440, __global float* data1_2621440, __global float* data2_2621440) {
  int gidx0 = get_group_id(0); /* 20480 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  float4 val0 = (*((__global float4*)((data1_2621440+alu0))));
  float4 val1 = (*((__global float4*)((data2_2621440+alu0))));
  *((__global float4*)((data0_2621440+alu0))) = (float4)((val0.x+val1.x),(val0.y+val1.y),(val0.z+val1.z),(val0.w+val1.w));
}