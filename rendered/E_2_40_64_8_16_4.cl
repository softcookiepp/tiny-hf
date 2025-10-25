__kernel void E_2_40_64_8_16_4(__global float* data0_2621440, __global float* data1_2621440, __global float* data2_64, __global float* data3_64, __global float* data4_320, __global float* data5_320) {
  int gidx0 = get_group_id(0); /* 64 */
  int gidx1 = get_group_id(1); /* 40 */
  int gidx2 = get_group_id(2); /* 2 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (lidx0+(gidx1<<3));
  float val0 = (*(data4_320+alu0));
  float val1 = (*(data5_320+alu0));
  int alu1 = ((gidx1<<15)+(lidx0<<12)+(gidx2*1310720)+(gidx0<<6)+(lidx1<<2));
  float4 val2 = (*((__global float4*)((data1_2621440+alu1))));
  int alu2 = ((gidx2<<5)+((alu0*205)>>11));
  float val3 = (*(data2_64+alu2));
  float val4 = (*(data3_64+alu2));
  float alu3 = (((val2.x-val3)*val4*val0)+val1);
  float alu4 = (((val2.y-val3)*val4*val0)+val1);
  float alu5 = (((val2.z-val3)*val4*val0)+val1);
  float alu6 = (((val2.w-val3)*val4*val0)+val1);
  *((__global float4*)((data0_2621440+alu1))) = (float4)((alu3*(1/(1.0+exp2((alu3*-1.4426950408889634))))),(alu4*(1/(1.0+exp2((alu4*-1.4426950408889634))))),(alu5*(1/(1.0+exp2((alu5*-1.4426950408889634))))),(alu6*(1/(1.0+exp2((alu6*-1.4426950408889634))))));
}