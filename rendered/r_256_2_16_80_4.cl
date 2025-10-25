__kernel void r_256_2_16_80_4(__global float* data0_8192, __global float* data1_2621440) {
  float acc0[1];
  int gidx0 = get_group_id(0); /* 256 */
  int lidx0 = get_local_id(0); /* 2 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (lidx1+(gidx0<<4));
  *(acc0+0) = 0.0;
  for (int ridx1002 = 0; ridx1002 < 80; ridx1002++) {
    int alu2 = (alu0+(lidx0*1310720)+(ridx1002<<14));
    float val0 = (*(data1_2621440+alu2));
    float val1 = (*(data1_2621440+(alu2+4096)));
    float val2 = (*(data1_2621440+(alu2+8192)));
    float val3 = (*(data1_2621440+(alu2+12288)));
    *(acc0+0) = ((*(acc0+0))+val0+val1+val2+val3);
  }
  *(data0_8192+(alu0+(lidx0<<12))) = ((*(acc0+0))*0.003125);
}