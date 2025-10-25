__kernel void En3(__global uint* data0_1) {
  uint val0 = (*(data0_1+0));
  *(data0_1+0) = (val0+8);
}