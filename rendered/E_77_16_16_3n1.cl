__kernel void E_77_16_16_3n1(__global float* data0_59136, __global float* data1_59136, __global float* data2_77, __global float* data3_77, __global float* data4_768, __global float* data5_768) {
  int gidx0 = get_group_id(0); /* 16 */
  int gidx1 = get_group_id(1); /* 77 */
  int lidx0 = get_local_id(0); /* 16 */
  float val0 = (*(data2_77+gidx1));
  float val1 = (*(data3_77+gidx1));
  int alu0 = ((gidx0*48)+(lidx0*3));
  float val2 = (*(data4_768+alu0));
  float val3 = (*(data5_768+alu0));
  int alu1 = (alu0+(gidx1*768));
  float val4 = (*(data1_59136+alu1));
  int alu2 = (alu1+1);
  float val5 = (*(data1_59136+alu2));
  int alu3 = (alu1+2);
  float val6 = (*(data1_59136+alu3));
  int alu4 = (alu0+1);
  float val7 = (*(data4_768+alu4));
  float val8 = (*(data5_768+alu4));
  int alu5 = (alu0+2);
  float val9 = (*(data4_768+alu5));
  float val10 = (*(data5_768+alu5));
  *(data0_59136+alu2) = (((val5-val0)*val1*val7)+val8);
  *(data0_59136+alu3) = (((val6-val0)*val1*val9)+val10);
  *(data0_59136+alu1) = (((val4-val0)*val1*val2)+val3);
}