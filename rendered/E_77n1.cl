__kernel void E_77n1(__global int* data0_77) {
  int gidx0 = get_group_id(0); /* 77 */
  *(data0_77+gidx0) = gidx0;
}