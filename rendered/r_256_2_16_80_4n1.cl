__kernel void r_256_2_16_80_4n1(__global float* data0_8192, __global float* data1_2621440, __global float* data2_8192) {
  float acc0[1];
  int gidx0 = get_group_id(0); /* 256 */
  int lidx0 = get_local_id(0); /* 2 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (lidx1+(gidx0<<4));
  int alu1 = (alu0+(lidx0<<12));
  *(acc0+0) = 0.0;
  float val0 = (*(data2_8192+alu1));
  for (int ridx1002 = 0; ridx1002 < 80; ridx1002++) {
    int alu3 = (alu0+(lidx0*1310720)+(ridx1002<<14));
    float val1 = (*(data1_2621440+alu3));
    float val2 = (*(data1_2621440+(alu3+4096)));
    float val3 = (*(data1_2621440+(alu3+8192)));
    float val4 = (*(data1_2621440+(alu3+12288)));
    float alu4 = (val1-val0);
    float alu5 = (val2-val0);
    float alu6 = (val3-val0);
    float alu7 = (val4-val0);
    *(acc0+0) = ((*(acc0+0))+(alu4*alu4)+(alu5*alu5)+(alu6*alu6)+(alu7*alu7));
  }
  *(data0_8192+alu1) = (1/sqrt((((*(acc0+0))*0.003125)+1e-05)));
}