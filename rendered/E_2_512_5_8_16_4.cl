__kernel void E_2_512_5_8_16_4(__global float* data0_2621440, __global float* data1_2621440) {
  int gidx0 = get_group_id(0); /* 5 */
  int gidx1 = get_group_id(1); /* 512 */
  int gidx2 = get_group_id(2); /* 2 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (gidx2*1310720);
  int alu1 = (lidx0+(gidx1<<3)+alu0+(gidx0<<18)+(lidx1<<14));
  float val0 = (*(data1_2621440+alu1));
  float val1 = (*(data1_2621440+(alu1+4096)));
  float val2 = (*(data1_2621440+(alu1+8192)));
  float val3 = (*(data1_2621440+(alu1+12288)));
  *((__global float4*)((data0_2621440+((gidx1*2560)+(lidx0*320)+alu0+(gidx0<<6)+(lidx1<<2))))) = (float4)(val0,val1,val2,val3);
}