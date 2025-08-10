#version 450
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform _11_10
{
    float _m0;
} _10;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float _m0[];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    float _m0[];
} _18;

layout(set = 0, binding = 3, std430) buffer _22_21
{
    float _m0[];
} _21;

layout(set = 0, binding = 4, std430) buffer _25_24
{
    float _m0[];
} _24;

shared float _13[16];

void main()
{
    float _74 = 0.0;
    int _69 = 0;
    float _63 = 0.0;
    int _56 = 0;
    int _75 = 0;
    float _70 = 0.0;
    int _66 = 0;
    int _59 = 0;
    float _76 = 0.0;
    float _72 = 0.0;
    float _68 = 0.0;
    int _61 = 0;
    uvec2 _111 = uvec2(4294967295u);
    uvec2 _165 = uvec2(4294967295u);
    if (all(equal(gl_LocalInvocationID, uvec3(0u))))
    {
        _13 = float[](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    barrier();
    _56 = int(gl_WorkGroupID.x);
    _59 = int(gl_WorkGroupID.y);
    _61 = int(gl_LocalInvocationID.x);
    _63 = _24._m0[_56];
    _66 = _61 * 80;
    for (;;)
    {
        if (all(equal(uvec2(0u), _111)))
        {
            break;
        }
        _111 -= uvec2(uint(_111.y == 0u), 1u);
        if (!(_69 < 80))
        {
            break;
        }
        _70 = _21._m0[((_56 * 1280) + _66) + _69];
        _72 = _18._m0[((_59 * 1280) + _66) + _69];
        _68 += (_72 * _70);
        _69++;
        continue;
    }
    _13[_61] = _68;
    barrier();
    if ((_61 != 0) != true)
    {
        for (;;)
        {
            if (all(equal(uvec2(0u), _165)))
            {
                break;
            }
            _165 -= uvec2(uint(_165.y == 0u), 1u);
            if (!(_75 < 16))
            {
                break;
            }
            _76 = _13[_75];
            _74 += _76;
            _75++;
            continue;
        }
        _15._m0[_56 + (_59 * 640)] = _74 + _63;
        return;
    }
    else
    {
        return;
    }
}

