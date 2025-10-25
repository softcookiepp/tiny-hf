__kernel void r_77_64_16_3_192_4(__global float* data0_236544, __global float* data1_59136, __global float* data2_2359296, __global float* data3_3072) {
  float acc0[3];
  int gidx0 = get_group_id(0); /* 64 */
  int gidx1 = get_group_id(1); /* 77 */
  int lidx0 = get_local_id(0); /* 16 */
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 192; ridx1003++) {
    int alu3 = (ridx1003<<2);
    int alu4 = ((gidx0*36864)+(lidx0*2304)+alu3);
    float4 val0 = (*((__global float4*)((data2_2359296+alu4))));
    float4 val1 = (*((__global float4*)((data2_2359296+(alu4+768)))));
    float4 val2 = (*((__global float4*)((data2_2359296+(alu4+1536)))));
    float4 val3 = (*((__global float4*)((data1_59136+((gidx1*768)+alu3)))));
    *(acc0+1) = ((*(acc0+1))+(val3.x*val1.x)+(val3.y*val1.y)+(val3.z*val1.z)+(val3.w*val1.w));
    *(acc0+2) = ((*(acc0+2))+(val3.x*val2.x)+(val3.y*val2.y)+(val3.z*val2.z)+(val3.w*val2.w));
    *(acc0+0) = ((*(acc0+0))+(val3.x*val0.x)+(val3.y*val0.y)+(val3.z*val0.z)+(val3.w*val0.w));
  }
  int alu9 = ((gidx0*48)+(lidx0*3));
  float val4 = (*(data3_3072+alu9));
  float val5 = (*(data3_3072+(alu9+1)));
  float val6 = (*(data3_3072+(alu9+2)));
  int alu10 = (alu9+(gidx1*3072));
  *(data0_236544+alu10) = ((*(acc0+0))+val4);
  *(data0_236544+(alu10+1)) = ((*(acc0+1))+val5);
  *(data0_236544+(alu10+2)) = ((*(acc0+2))+val6);
}