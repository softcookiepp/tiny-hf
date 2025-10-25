__kernel void E_5_8_4(__global float* data0_160) {
  int gidx0 = get_group_id(0); /* 5 */
  int lidx0 = get_local_id(0); /* 8 */
  int alu0 = ((gidx0<<5)+(lidx0<<2));
  *((__global float4*)((data0_160+alu0))) = (float4)(exp2(((((float)((alu0+1)))+-1.0)*-0.08304820237218408)),exp2(((((float)((alu0+2)))+-1.0)*-0.08304820237218408)),exp2(((((float)((alu0+3)))+-1.0)*-0.08304820237218408)),exp2(((((float)((alu0+4)))+-1.0)*-0.08304820237218408)));
}