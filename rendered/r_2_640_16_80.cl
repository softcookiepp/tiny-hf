__kernel void r_2_640_16_80(__global float* data0_1280, __global float* data1_2560, __global float* data2_819200, __global float* data3_640) {
  __attribute__ ((aligned (16))) __local float temp0[16];
  float acc0[1];
  float acc1[1];
  int gidx0 = get_group_id(0); /* 640 */
  int gidx1 = get_group_id(1); /* 2 */
  int lidx0 = get_local_id(0); /* 16 */
  *(acc1+0) = 0.0;
  float val0 = (*(data3_640+gidx0));
  *(acc0+0) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 80; ridx1003++) {
    int alu2 = ((lidx0*80)+ridx1003);
    float val1 = (*(data2_819200+(alu2+(gidx0*1280))));
    float val2 = (*(data1_2560+(alu2+(gidx1*1280))));
    *(acc0+0) = ((*(acc0+0))+(val2*val1));
  }
  *(temp0+lidx0) = (*(acc0+0));
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((((bool)(lidx0))!=1)) {
    for (int ridx1104 = 0; ridx1104 < 16; ridx1104++) {
      float val3 = (*(temp0+ridx1104));
      *(acc1+0) = ((*(acc1+0))+val3);
    }
    *(data0_1280+(gidx0+(gidx1*640))) = ((*(acc1+0))+val0);
  }
}