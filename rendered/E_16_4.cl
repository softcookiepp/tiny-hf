__kernel void E_16_4(__global int* data0_64) {
  int lidx0 = get_local_id(0); /* 16 */
  int alu0 = (lidx0<<2);
  int alu1 = (alu0+1);
  int alu2 = (alu0+2);
  int alu3 = (alu0+3);
  *(data0_64+alu0) = ((int)((0.5*(((float)(alu1))+-1.0))));
  *(data0_64+alu1) = ((int)((0.5*(((float)(alu2))+-1.0))));
  *(data0_64+alu2) = ((int)((0.5*(((float)(alu3))+-1.0))));
  *(data0_64+alu3) = ((int)((0.5*(((float)((alu0+4)))+-1.0))));
}