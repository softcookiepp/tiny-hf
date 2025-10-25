__kernel void r_77n1(__global int* data0_1, __global long* data1_77, __global long* data2_1) {
  int acc0[1];
  *(acc0+0) = -2147483648;
  long val0 = (*(data2_1+0));
  for (int ridx1000 = 0; ridx1000 < 77; ridx1000++) {
    long val1 = (*(data1_77+ridx1000));
    int alu1 = (((int)((val1==val0)))*(77-ridx1000));
    int alu2 = (((*(acc0+0))<alu1)?alu1:(*(acc0+0)));
    *(acc0+0) = alu2;
  }
  int alu5 = ((-(*(acc0+0))<-77)?(154-(*(acc0+0))):(77-(*(acc0+0))));
  *(data0_1+0) = alu5;
}