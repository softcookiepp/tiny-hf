__kernel void E_2n2(__global float* data0_2, __global long* data1_2) {
  int lidx0 = get_local_id(0); /* 2 */
  long val0 = (*(data1_2+lidx0));
  *(data0_2+lidx0) = ((float)(((long)(((float)(val0))))));
}