__kernel void En5(__global long* data0_1, __global long* data1_1) {
  long val0 = (*(data1_1+0));
  *(data0_1+0) = val0;
}