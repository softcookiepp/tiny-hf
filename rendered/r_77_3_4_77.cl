__kernel void r_77_3_4_77(__global float* data0_924, __global float* data1_71148) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 77 */
  int lidx0 = get_local_id(0); /* 3 */
  *(acc0+0) = -INFINITY;
  *(acc0+1) = -INFINITY;
  *(acc0+2) = -INFINITY;
  *(acc0+3) = -INFINITY;
  for (int ridx1003 = 0; ridx1003 < 77; ridx1003++) {
    int alu4 = ((gidx0*924)+(lidx0*308)+ridx1003);
    float val0 = (*(data1_71148+alu4));
    float val1 = (*(data1_71148+(alu4+77)));
    float val2 = (*(data1_71148+(alu4+154)));
    float val3 = (*(data1_71148+(alu4+231)));
    float alu5 = (((*(acc0+0))<val0)?val0:(*(acc0+0)));
    *(acc0+0) = alu5;
    float alu7 = (((*(acc0+1))<val1)?val1:(*(acc0+1)));
    *(acc0+1) = alu7;
    float alu9 = (((*(acc0+2))<val2)?val2:(*(acc0+2)));
    *(acc0+2) = alu9;
    float alu11 = (((*(acc0+3))<val3)?val3:(*(acc0+3)));
    *(acc0+3) = alu11;
  }
  *((__global float4*)((data0_924+((gidx0*12)+(lidx0<<2))))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}