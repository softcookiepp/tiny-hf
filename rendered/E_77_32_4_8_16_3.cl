__kernel void E_77_32_4_8_16_3(__global float* data0_3784704, __global long* data1_77, __global float* data2_37945344) {
  int gidx0 = get_group_id(0); /* 4 */
  int gidx1 = get_group_id(1); /* 32 */
  int gidx2 = get_group_id(2); /* 77 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  long val0 = (*(data1_77+gidx2));
  int alu0 = ((gidx0*12352)+(lidx1*772));
  int alu1 = ((gidx0*-9486336)+(lidx1*-592896)+(((int)(val0))*768)+(gidx0*9486336)+(lidx1*592896)+(gidx1*24)+(lidx0*3));
  bool alu2 = (((val0<((long)(alu0)))!=1)&(val0<((long)((alu0+772)))));
  float val1 = (alu2?*(data2_37945344+(alu1+1)):0.0);
  float val2 = (alu2?*(data2_37945344+(alu1+2)):0.0);
  float val3 = (alu2?*(data2_37945344+alu1):0.0);
  int alu3 = (lidx1+(gidx0<<4)+(gidx1*1536)+(lidx0*192)+(gidx2*49152));
  *(data0_3784704+alu3) = val3;
  *(data0_3784704+(alu3+64)) = val1;
  *(data0_3784704+(alu3+128)) = val2;
}