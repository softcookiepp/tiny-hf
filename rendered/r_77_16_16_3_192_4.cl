__kernel void r_77_16_16_3_192_4(__global float* data0_59136, __global float* data1_59136, __global float* data2_589824, __global float* data3_768) {
  float acc0[3];
  int gidx0 = get_group_id(0); /* 16 */
  int gidx1 = get_group_id(1); /* 77 */
  int lidx0 = get_local_id(0); /* 16 */
  int alu0 = (gidx1*768);
  *(acc0+0) = 0.0;
  *(acc0+1) = 0.0;
  *(acc0+2) = 0.0;
  for (int ridx1003 = 0; ridx1003 < 192; ridx1003++) {
    int alu4 = (ridx1003<<2);
    int alu5 = ((gidx0*36864)+(lidx0*2304)+alu4);
    float4 val0 = (*((__global float4*)((data2_589824+alu5))));
    float4 val1 = (*((__global float4*)((data2_589824+(alu5+768)))));
    float4 val2 = (*((__global float4*)((data2_589824+(alu5+1536)))));
    float4 val3 = (*((__global float4*)((data1_59136+(alu0+alu4)))));
    *(acc0+1) = ((*(acc0+1))+(val3.x*val1.x)+(val3.y*val1.y)+(val3.z*val1.z)+(val3.w*val1.w));
    *(acc0+2) = ((*(acc0+2))+(val3.x*val2.x)+(val3.y*val2.y)+(val3.z*val2.z)+(val3.w*val2.w));
    *(acc0+0) = ((*(acc0+0))+(val3.x*val0.x)+(val3.y*val0.y)+(val3.z*val0.z)+(val3.w*val0.w));
  }
  int alu10 = ((gidx0*48)+(lidx0*3));
  float val4 = (*(data3_768+alu10));
  float val5 = (*(data3_768+(alu10+1)));
  float val6 = (*(data3_768+(alu10+2)));
  int alu11 = (alu10+alu0);
  *(data0_59136+alu11) = ((*(acc0+0))+val4);
  *(data0_59136+(alu11+1)) = ((*(acc0+1))+val5);
  *(data0_59136+(alu11+2)) = ((*(acc0+2))+val6);
}