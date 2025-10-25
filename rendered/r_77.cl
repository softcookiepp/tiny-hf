__kernel void r_77(__global long* data0_1, __global long* data1_77) {
  long acc0[1];
  *(acc0+0) = -9223372036854775808;
  for (int ridx1000 = 0; ridx1000 < 77; ridx1000++) {
    long val0 = (*(data1_77+ridx1000));
    long alu1 = (((*(acc0+0))<val0)?val0:(*(acc0+0)));
    *(acc0+0) = alu1;
  }
  *(data0_1+0) = (*(acc0+0));
}