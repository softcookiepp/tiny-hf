__kernel void E_2464_32_3(__global float* data0_236544, __global float* data1_236544) {
  int gidx0 = get_group_id(0); /* 2464 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0*96)+(lidx0*3));
  float val0 = (*(data1_236544+alu0));
  int alu1 = (alu0+1);
  float val1 = (*(data1_236544+alu1));
  int alu2 = (alu0+2);
  float val2 = (*(data1_236544+alu2));
  *(data0_236544+alu1) = (1/(1.0+exp2((val1*-2.4554669595930156))));
  *(data0_236544+alu2) = (1/(1.0+exp2((val2*-2.4554669595930156))));
  *(data0_236544+alu0) = (1/(1.0+exp2((val0*-2.4554669595930156))));
}