__kernel void E_4n20(__global float* data0_4, __global float* data1_8) {
  float4 val0 = (*((__global float4*)((data1_8+0))));
  float4 val1 = (*((__global float4*)((data1_8+4))));
  *((__global float4*)((data0_4+0))) = (float4)(sin(((float)(((uchar)(((int)((sin((1.5707963267948966+(val0.x*-6.283185307179586)))*sqrt((log2((1.0-val1.x))*-1.3862943611198906)))))))))),sin(((float)(((uchar)(((int)((sin((1.5707963267948966+(val0.y*-6.283185307179586)))*sqrt((log2((1.0-val1.y))*-1.3862943611198906)))))))))),sin(((float)(((uchar)(((int)((sin((1.5707963267948966+(val0.z*-6.283185307179586)))*sqrt((log2((1.0-val1.z))*-1.3862943611198906)))))))))),sin(((float)(((uchar)(((int)((sin((1.5707963267948966+(val0.w*-6.283185307179586)))*sqrt((log2((1.0-val1.w))*-1.3862943611198906)))))))))));
}