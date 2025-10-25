__kernel void r_3_77_4_16_4_77(__global float* data0_59136, __global float* data1_71148, __global float* data2_59136) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 77 */
  int gidx1 = get_group_id(1); /* 3 */
  int lidx0 = get_local_id(0); /* 4 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (lidx1<<2);
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  *(acc0+3) = 0.0;
  for (int ridx1004 = 0; ridx1004 < 77; ridx1004++) {
    float val0 = (*(data1_71148+((gidx1*23716)+(lidx0*5929)+(gidx0*77)+ridx1004)));
    float4 val1 = (*((__global float4*)((data2_59136+((gidx1<<8)+(lidx0<<6)+alu0+(ridx1004*768))))));
    *(acc0+0) = ((*(acc0+0))+(val0*val1.x));
    *(acc0+1) = ((*(acc0+1))+(val0*val1.y));
    *(acc0+2) = ((*(acc0+2))+(val0*val1.z));
    *(acc0+3) = ((*(acc0+3))+(val0*val1.w));
  }
  *((__global float4*)((data0_59136+((gidx1*19712)+(lidx0*4928)+(gidx0<<6)+alu0)))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}