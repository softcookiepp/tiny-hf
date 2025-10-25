__kernel void E_2(__global int* data0_2) {
  int lidx0 = get_local_id(0); /* 2 */
  *(data0_2+lidx0) = ((int)((((float)((lidx0+1)))+-1.0)));
}