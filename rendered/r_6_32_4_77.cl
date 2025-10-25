__kernel void r_6_32_4_77(__global float* data0_768, __global int* data1_1, __global int* data2_77, __global float* data3_59136) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 6 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  int val0 = (*(data1_1+0));
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 77; ridx1003++) {
    int val1 = (*(data2_77+ridx1003));
    float4 val2 = (*((__global float4*)((data3_59136+(alu0+(ridx1003*768))))));
    bool alu5 = (val0!=val1);
    float alu6 = (alu5?0.0:val2.x);
    *(acc0+0) = ((*(acc0+0))+alu6);
    float alu8 = (alu5?0.0:val2.y);
    *(acc0+1) = ((*(acc0+1))+alu8);
    float alu10 = (alu5?0.0:val2.z);
    *(acc0+2) = ((*(acc0+2))+alu10);
    float alu12 = (alu5?0.0:val2.w);
    *(acc0+3) = ((*(acc0+3))+alu12);
  }
  *((__global float4*)((data0_768+alu0))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}