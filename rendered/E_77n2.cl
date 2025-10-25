__kernel void E_77n2(__global long* data0_77, __global long* data1_77) {
  int gidx0 = get_group_id(0); /* 77 */
  long val0 = (*(data1_77+gidx0));
  *(data0_77+gidx0) = val0;
}