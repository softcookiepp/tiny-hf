__kernel void E_77_77_4_3(__global float* data0_71148, __global float* data1_71148, __global float* data2_924, __global float* data3_924) {
  int gidx0 = get_group_id(0); /* 77 */
  int gidx1 = get_group_id(1); /* 77 */
  int lidx0 = get_local_id(0); /* 4 */
  int alu0 = ((gidx1*12)+(lidx0*3));
  float val0 = (*(data2_924+alu0));
  float val1 = (*(data3_924+alu0));
  int alu1 = (alu0+1);
  float val2 = (*(data2_924+alu1));
  float val3 = (*(data3_924+alu1));
  int alu2 = (alu0+2);
  float val4 = (*(data2_924+alu2));
  float val5 = (*(data3_924+alu2));
  int alu3 = (gidx0+(gidx1*924)+(lidx0*231));
  float val6 = (*(data1_71148+alu3));
  int alu4 = (alu3+77);
  float val7 = (*(data1_71148+alu4));
  int alu5 = (alu3+154);
  float val8 = (*(data1_71148+alu5));
  *(data0_71148+alu3) = (exp2(((val6-val0)*1.4426950408889634))*val1);
  *(data0_71148+alu4) = (exp2(((val7-val2)*1.4426950408889634))*val3);
  *(data0_71148+alu5) = (exp2(((val8-val4)*1.4426950408889634))*val5);
}