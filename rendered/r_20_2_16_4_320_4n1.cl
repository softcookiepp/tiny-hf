__kernel void r_20_2_16_4_320_4n1(__global float* data0_2560, __global float* data1_2560, __global float* data2_1638400, __global float* data3_1280) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 20 */
  int lidx0 = get_local_id(0); /* 2 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (lidx0*1280);
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 320; ridx1003++) {
    int alu5 = (ridx1003<<2);
    int alu6 = ((gidx0*81920)+(lidx1*5120)+alu5);
    float4 val0 = (*((__global float4*)((data2_1638400+alu6))));
    float4 val1 = (*((__global float4*)((data2_1638400+(alu6+1280)))));
    float4 val2 = (*((__global float4*)((data2_1638400+(alu6+2560)))));
    float4 val3 = (*((__global float4*)((data2_1638400+(alu6+3840)))));
    float4 val4 = (*((__global float4*)((data1_2560+(alu0+alu5)))));
    *(acc0+1) = ((*(acc0+1))+(val4.x*val1.x)+(val4.y*val1.y)+(val4.z*val1.z)+(val4.w*val1.w));
    *(acc0+2) = ((*(acc0+2))+(val4.x*val2.x)+(val4.y*val2.y)+(val4.z*val2.z)+(val4.w*val2.w));
    *(acc0+3) = ((*(acc0+3))+(val4.x*val3.x)+(val4.y*val3.y)+(val4.z*val3.z)+(val4.w*val3.w));
    *(acc0+0) = ((*(acc0+0))+(val4.x*val0.x)+(val4.y*val0.y)+(val4.z*val0.z)+(val4.w*val0.w));
  }
  int alu12 = ((gidx0<<6)+(lidx1<<2));
  float4 val5 = (*((__global float4*)((data3_1280+alu12))));
  *((__global float4*)((data0_2560+(alu12+alu0)))) = (float4)(((*(acc0+0))+val5.x),((*(acc0+1))+val5.y),((*(acc0+2))+val5.z),((*(acc0+3))+val5.w));
}