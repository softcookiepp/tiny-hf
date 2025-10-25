__kernel void r_512_32_4_1024_4n1(__global float* data0_65536, __global float* data1_268435456, __global float* data2_65536) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 512 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  float4 val0 = (*((__global float4*)((data2_65536+alu0))));
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 1024; ridx1003++) {
    int alu5 = ((gidx0<<19)+(lidx0<<14)+(ridx1003<<2));
    float4 val1 = (*((__global float4*)((data1_268435456+alu5))));
    float4 val2 = (*((__global float4*)((data1_268435456+(alu5+4096)))));
    float4 val3 = (*((__global float4*)((data1_268435456+(alu5+8192)))));
    float4 val4 = (*((__global float4*)((data1_268435456+(alu5+12288)))));
    *(acc0+1) = ((*(acc0+1))+exp2(((val2.x-val0.y)*1.4426950408889634))+exp2(((val2.y-val0.y)*1.4426950408889634))+exp2(((val2.z-val0.y)*1.4426950408889634))+exp2(((val2.w-val0.y)*1.4426950408889634)));
    *(acc0+2) = ((*(acc0+2))+exp2(((val3.x-val0.z)*1.4426950408889634))+exp2(((val3.y-val0.z)*1.4426950408889634))+exp2(((val3.z-val0.z)*1.4426950408889634))+exp2(((val3.w-val0.z)*1.4426950408889634)));
    *(acc0+3) = ((*(acc0+3))+exp2(((val4.x-val0.w)*1.4426950408889634))+exp2(((val4.y-val0.w)*1.4426950408889634))+exp2(((val4.z-val0.w)*1.4426950408889634))+exp2(((val4.w-val0.w)*1.4426950408889634)));
    *(acc0+0) = ((*(acc0+0))+exp2(((val1.x-val0.x)*1.4426950408889634))+exp2(((val1.y-val0.x)*1.4426950408889634))+exp2(((val1.z-val0.x)*1.4426950408889634))+exp2(((val1.w-val0.x)*1.4426950408889634)));
  }
  *((__global float4*)((data0_65536+alu0))) = (float4)((1/(*(acc0+0))),(1/(*(acc0+1))),(1/(*(acc0+2))),(1/(*(acc0+3))));
}