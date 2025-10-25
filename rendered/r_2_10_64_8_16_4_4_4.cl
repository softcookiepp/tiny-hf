__kernel void r_2_10_64_8_16_4_4_4(__global float* data0_2621440, __global float* data1_32768, __global float* data2_1280) {
  int gidx0 = get_group_id(0); /* 64 */
  int gidx1 = get_group_id(1); /* 10 */
  int gidx2 = get_group_id(2); /* 2 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = ((gidx1<<7)+(lidx0<<4));
  float4 val0 = (*((__global float4*)((data2_1280+alu0))));
  float4 val1 = (*((__global float4*)((data2_1280+(alu0+4)))));
  float4 val2 = (*((__global float4*)((data2_1280+(alu0+8)))));
  float4 val3 = (*((__global float4*)((data2_1280+(alu0+12)))));
  int alu1 = ((gidx0<<6)+(lidx1<<2));
  int alu2 = (alu1+(gidx2<<14));
  float4 val4 = (*((__global float4*)((data1_32768+alu2))));
  float4 val5 = (*((__global float4*)((data1_32768+(alu2+4096)))));
  float4 val6 = (*((__global float4*)((data1_32768+(alu2+8192)))));
  float4 val7 = (*((__global float4*)((data1_32768+(alu2+12288)))));
  int alu3 = ((gidx1<<17)+(lidx0<<14)+(gidx2*1310720)+alu1);
  *((__global float4*)((data0_2621440+alu3))) = (float4)(((val4.x*val0.x)+(val5.x*val0.y)+(val6.x*val0.z)+(val7.x*val0.w)),((val4.y*val0.x)+(val5.y*val0.y)+(val6.y*val0.z)+(val7.y*val0.w)),((val4.z*val0.x)+(val5.z*val0.y)+(val6.z*val0.z)+(val7.z*val0.w)),((val4.w*val0.x)+(val5.w*val0.y)+(val6.w*val0.z)+(val7.w*val0.w)));
  *((__global float4*)((data0_2621440+(alu3+4096)))) = (float4)(((val4.x*val1.x)+(val5.x*val1.y)+(val6.x*val1.z)+(val7.x*val1.w)),((val4.y*val1.x)+(val5.y*val1.y)+(val6.y*val1.z)+(val7.y*val1.w)),((val4.z*val1.x)+(val5.z*val1.y)+(val6.z*val1.z)+(val7.z*val1.w)),((val4.w*val1.x)+(val5.w*val1.y)+(val6.w*val1.z)+(val7.w*val1.w)));
  *((__global float4*)((data0_2621440+(alu3+8192)))) = (float4)(((val4.x*val2.x)+(val5.x*val2.y)+(val6.x*val2.z)+(val7.x*val2.w)),((val4.y*val2.x)+(val5.y*val2.y)+(val6.y*val2.z)+(val7.y*val2.w)),((val4.z*val2.x)+(val5.z*val2.y)+(val6.z*val2.z)+(val7.z*val2.w)),((val4.w*val2.x)+(val5.w*val2.y)+(val6.w*val2.z)+(val7.w*val2.w)));
  *((__global float4*)((data0_2621440+(alu3+12288)))) = (float4)(((val4.x*val3.x)+(val5.x*val3.y)+(val6.x*val3.z)+(val7.x*val3.w)),((val4.y*val3.x)+(val5.y*val3.y)+(val6.y*val3.z)+(val7.y*val3.w)),((val4.z*val3.x)+(val5.z*val3.y)+(val6.z*val3.z)+(val7.z*val3.w)),((val4.w*val3.x)+(val5.w*val3.y)+(val6.w*val3.z)+(val7.w*val3.w)));
}