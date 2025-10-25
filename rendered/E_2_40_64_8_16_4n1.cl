__kernel void E_2_40_64_8_16_4n1(__global float* data0_2621440, __global float* data1_2621440, __global float* data2_2621440, __global float* data3_64, __global float* data4_64, __global float* data5_320, __global float* data6_320) {
  int gidx0 = get_group_id(0); /* 64 */
  int gidx1 = get_group_id(1); /* 40 */
  int gidx2 = get_group_id(2); /* 2 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (lidx0+(gidx1<<3));
  float val0 = (*(data5_320+alu0));
  float val1 = (*(data6_320+alu0));
  int alu1 = ((gidx1<<15)+(lidx0<<12)+(gidx2*1310720)+(gidx0<<6)+(lidx1<<2));
  float4 val2 = (*((__global float4*)((data1_2621440+alu1))));
  float4 val3 = (*((__global float4*)((data2_2621440+alu1))));
  int alu2 = ((gidx2<<5)+((alu0*205)>>11));
  float val4 = (*(data3_64+alu2));
  float val5 = (*(data4_64+alu2));
  *((__global float4*)((data0_2621440+alu1))) = (float4)(((((val2.x+val3.x)-val4)*val5*val0)+val1),((((val2.y+val3.y)-val4)*val5*val0)+val1),((((val2.z+val3.z)-val4)*val5*val0)+val1),((((val2.w+val3.w)-val4)*val5*val0)+val1));
}