__kernel void r_77_77_4_3_16_4(__global float* data0_71148, __global float* data1_59136, __global float* data2_59136, __global float* data3_5929) {
  float acc0[3];
  int gidx0 = get_group_id(0); /* 77 */
  int gidx1 = get_group_id(1); /* 77 */
  int lidx0 = get_local_id(0); /* 4 */
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  int alu3 = (lidx0*192);
  for (int ridx1004 = 0; ridx1004 < 16; ridx1004++) {
    int alu4 = (ridx1004<<2);
    int alu5 = ((gidx0*768)+alu3+alu4);
    float4 val0 = (*((__global float4*)((data2_59136+alu5))));
    float4 val1 = (*((__global float4*)((data2_59136+(alu5+64)))));
    float4 val2 = (*((__global float4*)((data2_59136+(alu5+128)))));
    int alu6 = ((gidx1*768)+alu3+alu4);
    float4 val3 = (*((__global float4*)((data1_59136+alu6))));
    float4 val4 = (*((__global float4*)((data1_59136+(alu6+64)))));
    float4 val5 = (*((__global float4*)((data1_59136+(alu6+128)))));
    *(acc0+1) = ((*(acc0+1))+(val4.x*val1.x)+(val4.y*val1.y)+(val4.z*val1.z)+(val4.w*val1.w));
    *(acc0+2) = ((*(acc0+2))+(val5.x*val2.x)+(val5.y*val2.y)+(val5.z*val2.z)+(val5.w*val2.w));
    *(acc0+0) = ((*(acc0+0))+(val3.x*val0.x)+(val3.y*val0.y)+(val3.z*val0.z)+(val3.w*val0.w));
  }
  int alu11 = (gidx1*77);
  float val6 = (*(data3_5929+(gidx0+alu11)));
  int alu12 = (gidx0+alu11+(lidx0*17787));
  *(data0_71148+alu12) = (((*(acc0+0))*0.125)+val6);
  *(data0_71148+(alu12+5929)) = (((*(acc0+1))*0.125)+val6);
  *(data0_71148+(alu12+11858)) = (((*(acc0+2))*0.125)+val6);
}