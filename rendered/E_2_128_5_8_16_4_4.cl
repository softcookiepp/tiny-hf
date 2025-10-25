__kernel void E_2_128_5_8_16_4_4(__global float* data0_2621440, __global float* data1_2621440, __global float* data2_8192, __global float* data3_8192, __global float* data4_320, __global float* data5_320) {
  int gidx0 = get_group_id(0); /* 5 */
  int gidx1 = get_group_id(1); /* 128 */
  int gidx2 = get_group_id(2); /* 2 */
  int lidx0 = get_local_id(0); /* 8 */
  int lidx1 = get_local_id(1); /* 16 */
  int alu0 = (gidx2*1310720);
  int alu1 = ((gidx1<<5)+(lidx0<<2));
  int alu2 = (alu1+(gidx2<<12));
  float4 val0 = (*((__global float4*)((data2_8192+alu2))));
  float4 val1 = (*((__global float4*)((data3_8192+alu2))));
  int alu3 = ((gidx0<<6)+(lidx1<<2));
  float4 val2 = (*((__global float4*)((data4_320+alu3))));
  float4 val3 = (*((__global float4*)((data5_320+alu3))));
  int alu4 = (alu1+alu0+(gidx0<<18)+(lidx1<<14));
  float4 val4 = (*((__global float4*)((data1_2621440+alu4))));
  float4 val5 = (*((__global float4*)((data1_2621440+(alu4+4096)))));
  float4 val6 = (*((__global float4*)((data1_2621440+(alu4+8192)))));
  float4 val7 = (*((__global float4*)((data1_2621440+(alu4+12288)))));
  int alu5 = ((gidx1*10240)+(lidx0*1280)+alu0+alu3);
  *((__global float4*)((data0_2621440+alu5))) = (float4)((((val4.x-val0.x)*val1.x*val2.x)+val3.x),(((val5.x-val0.x)*val1.x*val2.y)+val3.y),(((val6.x-val0.x)*val1.x*val2.z)+val3.z),(((val7.x-val0.x)*val1.x*val2.w)+val3.w));
  *((__global float4*)((data0_2621440+(alu5+320)))) = (float4)((((val4.y-val0.y)*val1.y*val2.x)+val3.x),(((val5.y-val0.y)*val1.y*val2.y)+val3.y),(((val6.y-val0.y)*val1.y*val2.z)+val3.z),(((val7.y-val0.y)*val1.y*val2.w)+val3.w));
  *((__global float4*)((data0_2621440+(alu5+640)))) = (float4)((((val4.z-val0.z)*val1.z*val2.x)+val3.x),(((val5.z-val0.z)*val1.z*val2.y)+val3.y),(((val6.z-val0.z)*val1.z*val2.z)+val3.z),(((val7.z-val0.z)*val1.z*val2.w)+val3.w));
  *((__global float4*)((data0_2621440+(alu5+960)))) = (float4)((((val4.w-val0.w)*val1.w*val2.x)+val3.x),(((val5.w-val0.w)*val1.w*val2.y)+val3.y),(((val6.w-val0.w)*val1.w*val2.z)+val3.z),(((val7.w-val0.w)*val1.w*val2.w)+val3.w));
}