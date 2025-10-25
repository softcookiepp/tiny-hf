__kernel void r_77_16_48n31(__global float* data0_77, __global float* data1_59136, __global float* data2_59136, __global float* data3_59136, __global float* data4_59136, __global float* data5_59136, __global float* data6_59136, __global float* data7_59136, __global float* data8_59136, __global float* data9_59136, __global float* data10_59136, __global float* data11_59136, __global float* data12_59136, __global float* data13_59136, __global float* data14_59136, __global float* data15_59136, __global float* data16_59136, __global float* data17_77) {
  __attribute__ ((aligned (16))) __local float temp0[16];
  float acc0[1];
  float acc1[1];
  int gidx0 = get_group_id(0); /* 77 */
  int lidx0 = get_local_id(0); /* 16 */
  *(acc1+0) = 0.0;
  *(acc0+0) = 0.0;
  float val0 = (*(data17_77+gidx0));
  for (int ridx1002 = 0; ridx1002 < 48; ridx1002++) {
    int alu2 = ((lidx0*48)+ridx1002+(gidx0*768));
    float val1 = (*(data1_59136+alu2));
    float val2 = (*(data2_59136+alu2));
    float val3 = (*(data3_59136+alu2));
    float val4 = (*(data4_59136+alu2));
    float val5 = (*(data5_59136+alu2));
    float val6 = (*(data6_59136+alu2));
    float val7 = (*(data7_59136+alu2));
    float val8 = (*(data8_59136+alu2));
    float val9 = (*(data9_59136+alu2));
    float val10 = (*(data10_59136+alu2));
    float val11 = (*(data11_59136+alu2));
    float val12 = (*(data12_59136+alu2));
    float val13 = (*(data13_59136+alu2));
    float val14 = (*(data14_59136+alu2));
    float val15 = (*(data15_59136+alu2));
    float val16 = (*(data16_59136+alu2));
    float alu3 = ((val1+val2+val3+val4+val5+val6+val7+val8+val9+val10+val11+val12+val13+val14+val15+val16)-val0);
    *(acc0+0) = ((*(acc0+0))+(alu3*alu3));
  }
  *(temp0+lidx0) = (*(acc0+0));
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((((bool)(lidx0))!=1)) {
    for (int ridx1103 = 0; ridx1103 < 16; ridx1103++) {
      float val17 = (*(temp0+ridx1103));
      *(acc1+0) = ((*(acc1+0))+val17);
    }
    *(data0_77+gidx0) = (1/sqrt((((*(acc1+0))*0.0013020833333333333)+1e-05)));
  }
}