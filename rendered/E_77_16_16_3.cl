__kernel void E_77_16_16_3(__global float* data0_59136, __global long* data1_77, __global float* data2_59136) {
  int gidx0 = get_group_id(0); /* 16 */
  int gidx1 = get_group_id(1); /* 77 */
  int lidx0 = get_local_id(0); /* 16 */
  long val0 = (*(data1_77+gidx1));
  int alu0 = ((gidx0*48)+(lidx0*3));
  int alu1 = (alu0+(((int)(val0))*768));
  bool alu2 = ((-1<val0)&(val0<77));
  float val1 = (alu2?*(data2_59136+(alu1+1)):0.0);
  float val2 = (alu2?*(data2_59136+(alu1+2)):0.0);
  float val3 = (alu2?*(data2_59136+alu1):0.0);
  int alu3 = (alu0+(gidx1*768));
  *(data0_59136+alu3) = val3;
  *(data0_59136+(alu3+1)) = val1;
  *(data0_59136+(alu3+2)) = val2;
}