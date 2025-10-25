__kernel void r_77_16_48n18(__global float* data0_77, __global float* data1_59136, __global float* data2_59136, __global float* data3_59136, __global float* data4_59136, __global float* data5_59136, __global float* data6_59136, __global float* data7_59136, __global float* data8_59136, __global float* data9_59136, __global float* data10_59136) {
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
    *(acc0+0) = ((*(acc0+0))+val0+val1+val2+val3+val4+val5+val6+val7+val8+val9);
  }
  *(temp0+lidx0) = (*(acc0+0));
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((((bool)(lidx0))!=1)) {
    for (int ridx1103 = 0; ridx1103 < 16; ridx1103++) {
      float val10 = (*(temp0+ridx1103));
      *(acc1+0) = ((*(acc1+0))+val10);
    }
    *(data0_77+gidx0) = ((*(acc1+0))*0.0013020833333333333);
  }
}