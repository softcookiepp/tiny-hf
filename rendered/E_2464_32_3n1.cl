__kernel void E_2464_32_3n1(__global float* data0_236544, __global float* data1_236544, __global float* data2_236544) {
  int gidx0 = get_group_id(0); /* 2464 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0*96)+(lidx0*3));
  float val0 = (*(data1_236544+alu0));
  float val1 = (*(data2_236544+alu0));
  int alu1 = (alu0+1);
  float val2 = (*(data1_236544+alu1));
  float val3 = (*(data2_236544+alu1));
  int alu2 = (alu0+2);
  float val4 = (*(data1_236544+alu2));
  float val5 = (*(data2_236544+alu2));
  *(data0_236544+alu1) = (val2*val3);
  *(data0_236544+alu2) = (val4*val5);
  *(data0_236544+alu0) = (val0*val1);
}