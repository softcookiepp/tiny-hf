__kernel void E_5_2_16_4(__global float* data0_640, __global float* data1_320, __global float* data2_320) {
  int gidx0 = get_group_id(0); /* 5 */
  int lidx0 = get_local_id(0); /* 2 */
  int lidx1 = get_local_id(1); /* 16 */
  float4 cast0 = (float4)(0.0,0.0,0.0,0.0);
  int alu0 = (lidx1+(gidx0<<4));
  int alu1 = ((gidx0<<6)+(lidx1<<2));
  int alu2 = (alu1+(lidx0*160));
  float4 val0 = ((alu0<40)?*((__global float4*)((data1_320+alu2))):cast0);
  float4 val1 = ((39<alu0)?*((__global float4*)((data2_320+(alu2+-160)))):cast0);
  *((__global float4*)((data0_640+(alu1+(lidx0*320))))) = (float4)((val0.x+val1.x),(val0.y+val1.y),(val0.z+val1.z),(val0.w+val1.w));
}