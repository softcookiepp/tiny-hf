__kernel void r_128_32_4_40_4n1(__global float* data0_16384, __global float* data1_2621440, __global float* data2_2621440) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 128 */
  int lidx0 = get_local_id(0); /* 32 */
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 40; ridx1003++) {
    int alu4 = ((gidx0*20480)+(lidx0*640)+(ridx1003<<2));
    float4 val0 = (*((__global float4*)((data1_2621440+alu4))));
    float4 val1 = (*((__global float4*)((data2_2621440+alu4))));
    int alu5 = (alu4+160);
    float4 val2 = (*((__global float4*)((data1_2621440+alu5))));
    float4 val3 = (*((__global float4*)((data2_2621440+alu5))));
    int alu6 = (alu4+320);
    float4 val4 = (*((__global float4*)((data1_2621440+alu6))));
    float4 val5 = (*((__global float4*)((data2_2621440+alu6))));
    int alu7 = (alu4+480);
    float4 val6 = (*((__global float4*)((data1_2621440+alu7))));
    float4 val7 = (*((__global float4*)((data2_2621440+alu7))));
    *(acc0+0) = ((*(acc0+0))+val0.x+val1.x+val0.y+val1.y+val0.z+val1.z+val0.w+val1.w);
    *(acc0+1) = ((*(acc0+1))+val2.x+val3.x+val2.y+val3.y+val2.z+val3.z+val2.w+val3.w);
    *(acc0+2) = ((*(acc0+2))+val4.x+val5.x+val4.y+val5.y+val4.z+val5.z+val4.w+val5.w);
    *(acc0+3) = ((*(acc0+3))+val6.x+val7.x+val6.y+val7.y+val6.z+val7.z+val6.w+val7.w);
  }
  *((__global float4*)((data0_16384+((gidx0<<7)+(lidx0<<2))))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}