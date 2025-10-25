__kernel void E_10_32_4(__global int* data0_1280) {
  int gidx0 = get_group_id(0); /* 10 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = ((gidx0<<7)+(lidx0<<2));
  int alu1 = (alu0+1);
  int alu2 = (alu0+2);
  int alu3 = (alu0+3);
  *(data0_1280+alu0) = ((int)((((float)(alu1))+-1.0)));
  *(data0_1280+alu1) = ((int)((((float)(alu2))+-1.0)));
  *(data0_1280+alu2) = ((int)((((float)(alu3))+-1.0)));
  *(data0_1280+alu3) = ((int)((((float)((alu0+4)))+-1.0)));
}