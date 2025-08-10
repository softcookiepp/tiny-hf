fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> temp0: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3:array<f32>;
@compute @workgroup_size(16) fn r_2_640_16_80(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 640 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 16 */
  var val0 = data3[gidx0];
  var alu0 = (lidx0*80);
  var acc0 = 0.0f;
  for (var ridx3 = 0; ridx3 < 80; ridx3++) {
    var val1 = data2[((gidx0*1280)+alu0+ridx3)];
    var val2 = data1[((gidx1*1280)+alu0+ridx3)];
    acc0 = (acc0+(val2*val1));
  }
  temp0[lidx0] = acc0;
  workgroupBarrier();
  if (((bool(lidx0))!=true)) {
    var acc1 = 0.0f;
    for (var ridx1002 = 0; ridx1002 < 16; ridx1002++) {
      var val3 = temp0[ridx1002];
      acc1 = (acc1+val3);
    }
    data0[(gidx0+(gidx1*640))] = (acc1+val0);
  }
}