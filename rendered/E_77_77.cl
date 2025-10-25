__kernel void E_77_77(__global float* data0_5929, __global int* data1_77, __global int* data2_77) {
  int gidx0 = get_group_id(0); /* 77 */
  int gidx1 = get_group_id(1); /* 77 */
  int val0 = (*(data1_77+gidx0));
  int val1 = (*(data2_77+gidx1));
  float alu0 = ((val0<val1)?0.0:-INFINITY);
  *(data0_5929+(gidx0+(gidx1*77))) = alu0;
}