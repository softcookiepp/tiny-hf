__kernel void r_8_4_8_16_4_40_4n1(__global float* data0_16384, __global float* data1_2621440, __global float* data2_2621440, __global float* data3_64) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 4 */
  int gidx1 = get_group_id(1); /* 8 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  float val0 = (*(data3_64+(lidx0+(gidx1<<3))));
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 40; ridx1003++) {
    int alu4 = ((gidx0*10240)+(lidx1*640)+(gidx1*327680)+(lidx0*40960)+(ridx1003<<2));
    float4 val1 = (*((__global float4*)((data1_2621440+alu4))));
    float4 val2 = (*((__global float4*)((data2_2621440+alu4))));
    int alu5 = (alu4+160);
    float4 val3 = (*((__global float4*)((data1_2621440+alu5))));
    float4 val4 = (*((__global float4*)((data2_2621440+alu5))));
    int alu6 = (alu4+320);
    float4 val5 = (*((__global float4*)((data1_2621440+alu6))));
    float4 val6 = (*((__global float4*)((data2_2621440+alu6))));
    int alu7 = (alu4+480);
    float4 val7 = (*((__global float4*)((data1_2621440+alu7))));
    float4 val8 = (*((__global float4*)((data2_2621440+alu7))));
    float alu8 = ((val3.x+val4.x)-val0);
    float alu9 = ((val5.x+val6.x)-val0);
    float alu10 = ((val7.x+val8.x)-val0);
    float alu11 = ((val1.x+val2.x)-val0);
    float alu12 = ((val3.y+val4.y)-val0);
    float alu13 = ((val5.y+val6.y)-val0);
    float alu14 = ((val7.y+val8.y)-val0);
    float alu15 = ((val1.y+val2.y)-val0);
    float alu16 = ((val3.z+val4.z)-val0);
    float alu17 = ((val5.z+val6.z)-val0);
    float alu18 = ((val7.z+val8.z)-val0);
    float alu19 = ((val1.z+val2.z)-val0);
    float alu20 = ((val3.w+val4.w)-val0);
    *(acc0+1) = ((*(acc0+1))+(alu8*alu8)+(alu12*alu12)+(alu16*alu16)+(alu20*alu20));
    float alu22 = ((val5.w+val6.w)-val0);
    *(acc0+2) = ((*(acc0+2))+(alu9*alu9)+(alu13*alu13)+(alu17*alu17)+(alu22*alu22));
    float alu24 = ((val7.w+val8.w)-val0);
    *(acc0+3) = ((*(acc0+3))+(alu10*alu10)+(alu14*alu14)+(alu18*alu18)+(alu24*alu24));
    float alu26 = ((val1.w+val2.w)-val0);
    *(acc0+0) = ((*(acc0+0))+(alu11*alu11)+(alu15*alu15)+(alu19*alu19)+(alu26*alu26));
  }
  *((__global float4*)((data0_16384+((gidx0<<6)+(lidx1<<2)+(gidx1<<11)+(lidx0<<8))))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}