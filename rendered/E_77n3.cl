__kernel void E_77n3(__global int* data0_77, __global int* data1_77) {
  int gidx0 = get_group_id(0); /* 77 */
  int val0 = (*(data1_77+gidx0));
  *(data0_77+gidx0) = (val0+1);
}