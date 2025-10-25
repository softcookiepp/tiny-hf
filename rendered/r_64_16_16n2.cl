__kernel void r_64_16_16n2(__global float* data0_64, __global float* data1_16384) {
  __attribute__ ((aligned (16))) __local float temp0[16];
  float acc0[1];
  float acc1[1];
  int gidx0 = get_group_id(0); /* 64 */
  int lidx0 = get_local_id(0); /* 16 */
  *(acc1+0) = 0.0;
  *(acc0+0) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 16; ridx1003++) {
    float val0 = (*(data1_16384+((lidx0<<4)+ridx1003+(gidx0<<8))));
    *(acc0+0) = ((*(acc0+0))+val0);
  }
  *(temp0+lidx0) = (*(acc0+0));
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((((bool)(lidx0))!=1)) {
    for (int ridx1104 = 0; ridx1104 < 16; ridx1104++) {
      float val1 = (*(temp0+ridx1104));
      *(acc1+0) = ((*(acc1+0))+val1);
    }
    *(data0_64+gidx0) = (1/sqrt((((*(acc1+0))*2.44140625e-05)+1e-06)));
  }
}