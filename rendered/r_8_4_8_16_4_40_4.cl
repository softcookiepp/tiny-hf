__kernel void r_8_4_8_16_4_40_4(__global float* data0_16384, __global float* data1_2621440, __global float* data2_64) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 4 */
  int gidx1 = get_group_id(1); /* 8 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  float val0 = (*(data2_64+(lidx0+(gidx1<<3))));
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 40; ridx1003++) {
    int alu4 = ((gidx0*10240)+(lidx1*640)+(gidx1*327680)+(lidx0*40960)+(ridx1003<<2));
    float4 val1 = (*((__global float4*)((data1_2621440+alu4))));
    float4 val2 = (*((__global float4*)((data1_2621440+(alu4+160)))));
    float4 val3 = (*((__global float4*)((data1_2621440+(alu4+320)))));
    float4 val4 = (*((__global float4*)((data1_2621440+(alu4+480)))));
    float alu5 = (val2.x-val0);
    float alu6 = (val3.x-val0);
    float alu7 = (val4.x-val0);
    float alu8 = (val1.x-val0);
    float alu9 = (val2.y-val0);
    float alu10 = (val3.y-val0);
    float alu11 = (val4.y-val0);
    float alu12 = (val1.y-val0);
    float alu13 = (val2.z-val0);
    float alu14 = (val3.z-val0);
    float alu15 = (val4.z-val0);
    float alu16 = (val1.z-val0);
    float alu17 = (val2.w-val0);
    *(acc0+1) = ((*(acc0+1))+(alu5*alu5)+(alu9*alu9)+(alu13*alu13)+(alu17*alu17));
    float alu19 = (val3.w-val0);
    *(acc0+2) = ((*(acc0+2))+(alu6*alu6)+(alu10*alu10)+(alu14*alu14)+(alu19*alu19));
    float alu21 = (val4.w-val0);
    *(acc0+3) = ((*(acc0+3))+(alu7*alu7)+(alu11*alu11)+(alu15*alu15)+(alu21*alu21));
    float alu23 = (val1.w-val0);
    *(acc0+0) = ((*(acc0+0))+(alu8*alu8)+(alu12*alu12)+(alu16*alu16)+(alu23*alu23));
  }
  *((__global float4*)((data0_16384+((gidx0<<6)+(lidx1<<2)+(gidx1<<11)+(lidx0<<8))))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}