__kernel void r_20_2_16_4_80_4(__global float* data0_2560, __global float* data1_640, __global float* data2_409600, __global float* data3_1280) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 20 */
  int lidx0 = get_local_id(0); /* 2 */
  int lidx1 = get_local_id(1); /* 16 */
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1002 = 0; ridx1002 < 80; ridx1002++) {
    int alu4 = (ridx1002<<2);
    int alu5 = ((gidx0*20480)+(lidx1*1280)+alu4);
    float4 val0 = (*((__global float4*)((data2_409600+alu5))));
    float4 val1 = (*((__global float4*)((data2_409600+(alu5+320)))));
    float4 val2 = (*((__global float4*)((data2_409600+(alu5+640)))));
    float4 val3 = (*((__global float4*)((data2_409600+(alu5+960)))));
    float4 val4 = (*((__global float4*)((data1_640+((lidx0*320)+alu4)))));
    *(acc0+1) = ((*(acc0+1))+(val4.x*val1.x)+(val4.y*val1.y)+(val4.z*val1.z)+(val4.w*val1.w));
    *(acc0+2) = ((*(acc0+2))+(val4.x*val2.x)+(val4.y*val2.y)+(val4.z*val2.z)+(val4.w*val2.w));
    *(acc0+3) = ((*(acc0+3))+(val4.x*val3.x)+(val4.y*val3.y)+(val4.z*val3.z)+(val4.w*val3.w));
    *(acc0+0) = ((*(acc0+0))+(val4.x*val0.x)+(val4.y*val0.y)+(val4.z*val0.z)+(val4.w*val0.w));
  }
  int alu11 = ((gidx0<<6)+(lidx1<<2));
  float4 val5 = (*((__global float4*)((data3_1280+alu11))));
  float alu12 = ((*(acc0+0))+val5.x);
  float alu13 = ((*(acc0+1))+val5.y);
  float alu14 = ((*(acc0+2))+val5.z);
  float alu15 = ((*(acc0+3))+val5.w);
  *((__global float4*)((data0_2560+(alu11+(lidx0*1280))))) = (float4)((alu12*(1/(1.0+exp2((alu12*-1.4426950408889634))))),(alu13*(1/(1.0+exp2((alu13*-1.4426950408889634))))),(alu14*(1/(1.0+exp2((alu14*-1.4426950408889634))))),(alu15*(1/(1.0+exp2((alu15*-1.4426950408889634))))));
}