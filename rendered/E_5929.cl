__kernel void E_5929(__global float* data0_5929, __global float* data1_5929) {
  int gidx0 = get_group_id(0); /* 5929 */
  float val0 = (*(data1_5929+gidx0));
  *(data0_5929+gidx0) = val0;
}