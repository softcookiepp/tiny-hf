__kernel void r_616_32_3_16_4(__global float* data0_59136, __global float* data1_3784704, __global float* data2_59136) {
  float acc0[3];
  int gidx0 = get_group_id(0); /* 616 */
  int lidx0 = get_local_id(0); /* 32 */
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  for (int ridx1004 = 0; ridx1004 < 16; ridx1004++) {
    int alu3 = ((gidx0*6144)+(lidx0*192)+(ridx1004<<2));
    float4 val0 = (*((__global float4*)((data1_3784704+alu3))));
    float4 val1 = (*((__global float4*)((data1_3784704+(alu3+64)))));
    float4 val2 = (*((__global float4*)((data1_3784704+(alu3+128)))));
    *(acc0+0) = ((*(acc0+0))+val0.x+val0.y+val0.z+val0.w);
    *(acc0+1) = ((*(acc0+1))+val1.x+val1.y+val1.z+val1.w);
    *(acc0+2) = ((*(acc0+2))+val2.x+val2.y+val2.z+val2.w);
  }
  int alu8 = ((gidx0*96)+(lidx0*3));
  float val3 = (*(data2_59136+alu8));
  int alu9 = (alu8+1);
  float val4 = (*(data2_59136+alu9));
  int alu10 = (alu8+2);
  float val5 = (*(data2_59136+alu10));
  *(data0_59136+alu8) = ((*(acc0+0))+val3);
  *(data0_59136+alu9) = ((*(acc0+1))+val4);
  *(data0_59136+alu10) = ((*(acc0+2))+val5);
}