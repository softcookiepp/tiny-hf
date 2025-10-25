__kernel void E_2n1(__global long* data0_2, __global long* data1_1) {
  int lidx0 = get_local_id(0); /* 2 */
  long val0 = (*(data1_1+0));
  *(data0_2+lidx0) = val0;
}