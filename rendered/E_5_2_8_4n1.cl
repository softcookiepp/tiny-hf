__kernel void E_5_2_8_4n1(__global float* data0_320, __global float* data1_2, __global float* data2_160) {
  int gidx0 = get_group_id(0); /* 5 */
  int lidx0 = get_local_id(0); /* 2 */
  int lidx1 = get_local_id(1); /* 8 */
  float val0 = (*(data1_2+lidx0));
  int alu0 = ((gidx0<<5)+(lidx1<<2));
  float4 val1 = (*((__global float4*)((data2_160+alu0))));
  *((__global float4*)((data0_320+(alu0+(lidx0*160))))) = (float4)(sin((val0*val1.x)),sin((val0*val1.y)),sin((val0*val1.z)),sin((val0*val1.w)));
}