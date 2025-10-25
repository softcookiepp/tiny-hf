__kernel void E_1848_32_2(__global float* data0_118272, __global float* data1_59136, __global float* data2_59136) {
  int gidx0 = get_group_id(0); /* 1848 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = (lidx0+(gidx0<<5));
  float val0 = (*(data1_59136+alu0));
  float val1 = (*(data2_59136+alu0));
  *(data0_118272+alu0) = val0;
  *(data0_118272+(alu0+59136)) = val1;
}