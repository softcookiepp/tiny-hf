__kernel void r_77_16_48n40(__global float* data0_77, __global float* data1_59136, __global float* data2_59136, __global float* data3_59136, __global float* data4_59136, __global float* data5_59136, __global float* data6_59136, __global float* data7_59136, __global float* data8_59136, __global float* data9_59136, __global float* data10_59136, __global float* data11_59136, __global float* data12_59136, __global float* data13_59136, __global float* data14_59136, __global float* data15_59136, __global float* data16_59136, __global float* data17_59136, __global float* data18_59136, __global float* data19_59136, __global float* data20_59136, __global float* data21_59136) {
  __attribute__ ((aligned (16))) __local float temp0[16];
  float acc0[1];
  float acc1[1];
  int gidx0 = get_group_id(0); /* 77 */
  int lidx0 = get_local_id(0); /* 16 */
  *(acc1+0) = 0.0;
  *(acc0+0) = 0.0;
  for (int ridx1002 = 0; ridx1002 < 48; ridx1002++) {
    int alu2 = ((lidx0*48)+ridx1002+(gidx0*768));
    float val0 = (*(data1_59136+alu2));
    float val1 = (*(data2_59136+alu2));
    float val2 = (*(data3_59136+alu2));
    float val3 = (*(data4_59136+alu2));
    float val4 = (*(data5_59136+alu2));
    float val5 = (*(data6_59136+alu2));
    float val6 = (*(data7_59136+alu2));
    float val7 = (*(data8_59136+alu2));
    float val8 = (*(data9_59136+alu2));
    float val9 = (*(data10_59136+alu2));
    float val10 = (*(data11_59136+alu2));
    float val11 = (*(data12_59136+alu2));
    float val12 = (*(data13_59136+alu2));
    float val13 = (*(data14_59136+alu2));
    float val14 = (*(data15_59136+alu2));
    float val15 = (*(data16_59136+alu2));
    float val16 = (*(data17_59136+alu2));
    float val17 = (*(data18_59136+alu2));
    float val18 = (*(data19_59136+alu2));
    float val19 = (*(data20_59136+alu2));
    float val20 = (*(data21_59136+alu2));
    *(acc0+0) = ((*(acc0+0))+val0+val1+val2+val3+val4+val5+val6+val7+val8+val9+val10+val11+val12+val13+val14+val15+val16+val17+val18+val19+val20);
  }
  *(temp0+lidx0) = (*(acc0+0));
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((((bool)(lidx0))!=1)) {
    for (int ridx1103 = 0; ridx1103 < 16; ridx1103++) {
      float val21 = (*(temp0+ridx1103));
      *(acc1+0) = ((*(acc1+0))+val21);
    }
    *(data0_77+gidx0) = ((*(acc1+0))*0.0013020833333333333);
  }
}