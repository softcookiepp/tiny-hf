__kernel void r_77_3_4_77n1(__global float* data0_924, __global float* data1_71148, __global float* data2_924) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 77 */
  int lidx0 = get_local_id(0); /* 3 */
  int alu0 = ((gidx0*12)+(lidx0<<2));
  float4 val0 = (*((__global float4*)((data2_924+alu0))));
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 77; ridx1003++) {
    int alu5 = ((gidx0*924)+(lidx0*308)+ridx1003);
    float val1 = (*(data1_71148+alu5));
    float val2 = (*(data1_71148+(alu5+77)));
    float val3 = (*(data1_71148+(alu5+154)));
    float val4 = (*(data1_71148+(alu5+231)));
    *(acc0+1) = ((*(acc0+1))+exp2(((val2-val0.y)*1.4426950408889634)));
    *(acc0+2) = ((*(acc0+2))+exp2(((val3-val0.z)*1.4426950408889634)));
    *(acc0+3) = ((*(acc0+3))+exp2(((val4-val0.w)*1.4426950408889634)));
    *(acc0+0) = ((*(acc0+0))+exp2(((val1-val0.x)*1.4426950408889634)));
  }
  *((__global float4*)((data0_924+alu0))) = (float4)((1/(*(acc0+0))),(1/(*(acc0+1))),(1/(*(acc0+2))),(1/(*(acc0+3))));
}