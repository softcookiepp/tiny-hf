__kernel void E_77_16_16_3n2(__global float* data0_59136, __global float* data1_59136, __global float* data2_59136, __global float* data3_77, __global float* data4_77, __global float* data5_768, __global float* data6_768) {
  int gidx0 = get_group_id(0); /* 16 */
  int gidx1 = get_group_id(1); /* 77 */
  int lidx0 = get_local_id(0); /* 16 */
  float val0 = (*(data3_77+gidx1));
  float val1 = (*(data4_77+gidx1));
  int alu0 = ((gidx0*48)+(lidx0*3));
  float val2 = (*(data5_768+alu0));
  float val3 = (*(data6_768+alu0));
  int alu1 = (alu0+(gidx1*768));
  float val4 = (*(data1_59136+alu1));
  float val5 = (*(data2_59136+alu1));
  int alu2 = (alu1+1);
  float val6 = (*(data1_59136+alu2));
  float val7 = (*(data2_59136+alu2));
  int alu3 = (alu1+2);
  float val8 = (*(data1_59136+alu3));
  float val9 = (*(data2_59136+alu3));
  int alu4 = (alu0+1);
  float val10 = (*(data5_768+alu4));
  float val11 = (*(data6_768+alu4));
  int alu5 = (alu0+2);
  float val12 = (*(data5_768+alu5));
  float val13 = (*(data6_768+alu5));
  *(data0_59136+alu2) = ((((val6+val7)-val0)*val1*val10)+val11);
  *(data0_59136+alu3) = ((((val8+val9)-val0)*val1*val12)+val13);
  *(data0_59136+alu1) = ((((val4+val5)-val0)*val1*val2)+val3);
}