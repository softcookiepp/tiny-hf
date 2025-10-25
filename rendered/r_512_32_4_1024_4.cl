__kernel void r_512_32_4_1024_4(__global float* data0_65536, __global float* data1_268435456) {
  float acc0[4];
  int gidx0 = get_group_id(0); /* 512 */
  int lidx0 = get_local_id(0); /* 32 */
  *(acc0+0) = -INFINITY;
  *(acc0+1) = -INFINITY;
  *(acc0+2) = -INFINITY;
  *(acc0+3) = -INFINITY;
  for (int ridx1003 = 0; ridx1003 < 1024; ridx1003++) {
    int alu4 = ((gidx0<<19)+(lidx0<<14)+(ridx1003<<2));
    float4 val0 = (*((__global float4*)((data1_268435456+alu4))));
    float4 val1 = (*((__global float4*)((data1_268435456+(alu4+4096)))));
    float4 val2 = (*((__global float4*)((data1_268435456+(alu4+8192)))));
    float4 val3 = (*((__global float4*)((data1_268435456+(alu4+12288)))));
    float alu5 = (((*(acc0+0))<val0.x)?val0.x:(*(acc0+0)));
    float alu6 = (((*(acc0+1))<val1.x)?val1.x:(*(acc0+1)));
    float alu7 = (((*(acc0+2))<val2.x)?val2.x:(*(acc0+2)));
    float alu8 = (((*(acc0+3))<val3.x)?val3.x:(*(acc0+3)));
    float alu9 = ((alu5<val0.y)?val0.y:alu5);
    float alu10 = ((alu6<val1.y)?val1.y:alu6);
    float alu11 = ((alu7<val2.y)?val2.y:alu7);
    float alu12 = ((alu8<val3.y)?val3.y:alu8);
    float alu13 = ((alu9<val0.z)?val0.z:alu9);
    float alu14 = ((alu10<val1.z)?val1.z:alu10);
    float alu15 = ((alu11<val2.z)?val2.z:alu11);
    float alu16 = ((alu12<val3.z)?val3.z:alu12);
    float alu17 = ((alu13<val0.w)?val0.w:alu13);
    *(acc0+0) = alu17;
    float alu19 = ((alu14<val1.w)?val1.w:alu14);
    *(acc0+1) = alu19;
    float alu21 = ((alu15<val2.w)?val2.w:alu15);
    *(acc0+2) = alu21;
    float alu23 = ((alu16<val3.w)?val3.w:alu16);
    *(acc0+3) = alu23;
  }
  *((__global float4*)((data0_65536+((gidx0<<7)+(lidx0<<2))))) = (float4)((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
}