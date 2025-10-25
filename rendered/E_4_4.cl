__kernel void E_4_4(__global int* data0_16) {
  int lidx0 = get_local_id(0); /* 4 */
  int alu0 = (lidx0<<2);
  int alu1 = (alu0+1);
  int alu2 = (alu0+2);
  int alu3 = (alu0+3);
  *(data0_16+alu0) = ((int)((0.5*(((float)(alu1))+-1.0))));
  *(data0_16+alu1) = ((int)((0.5*(((float)(alu2))+-1.0))));
  *(data0_16+alu2) = ((int)((0.5*(((float)(alu3))+-1.0))));
  *(data0_16+alu3) = ((int)((0.5*(((float)((alu0+4)))+-1.0))));
}