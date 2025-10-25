__kernel void E_512_32_2(__global float* data0_32768, __global float* data1_16384) {
  int gidx0 = get_group_id(0); /* 512 */
  int lidx0 = get_local_id(0); /* 32 */
  int alu0 = (lidx0+(gidx0<<5));
  float val0 = (*(data1_16384+alu0));
  *(data0_32768+alu0) = val0;
  *(data0_32768+(alu0+16384)) = val0;
}