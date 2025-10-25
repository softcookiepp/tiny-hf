__kernel void r_128_32_4_40_4(__global float* data0_16384, __global float* data1_2621440) {
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
    float4 val1 = (*((__global float4*)((data1_2621440+(alu4+160)))));
    float4 val2 = (*((__global float4*)((data1_2621440+(alu4+320)))));
    float4 val3 = (*((__global float4*)((data1_2621440+(alu4+480)))));
    *(acc0+0) = ((*(acc0+0))+val0.x+val0.y+val0.z+val0.w);
    *(acc0+1) = ((*(acc0+1))+val1.x+val1.y+val1.z+val1.w);
    *(acc0+2) = ((*(acc0+2))+val2.x+val2.y+val2.z+val2.w);
    *(acc0+3) = ((*(acc0+3))+val3.x+val3.y+val3.z+val3.w);
  }
  *((__global float4*)((data0_16384+((gidx0<<7)+(lidx0<<2))))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}