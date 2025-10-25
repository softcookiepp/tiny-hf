__kernel void E_8192_64_8_16_4(__global float* data0_268435456, __global float* data1_268435456, __global float* data2_65536, __global float* data3_65536) {
  int gidx0 = get_group_id(0); /* 64 */
  int gidx1 = get_group_id(1); /* 8192 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (lidx0+(gidx1<<3));
  float val0 = (*(data2_65536+alu0));
  float val1 = (*(data3_65536+alu0));
  int alu1 = ((gidx0<<6)+(lidx1<<2)+(gidx1<<15)+(lidx0<<12));
  float4 val2 = (*((__global float4*)((data1_268435456+alu1))));
  *((__global float4*)((data0_268435456+alu1))) = (float4)((exp2(((val2.x-val0)*1.4426950408889634))*val1),(exp2(((val2.y-val0)*1.4426950408889634))*val1),(exp2(((val2.z-val0)*1.4426950408889634))*val1),(exp2(((val2.w-val0)*1.4426950408889634))*val1));
}