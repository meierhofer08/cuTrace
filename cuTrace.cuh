//
// Created by MM on 23.11.2023.
//

#ifndef CUTRACE_CUTRACE_CUH
#define CUTRACE_CUTRACE_CUH

#include <vector_types.h>

struct Sphere {
    float4 center; // center.xyz, radius
    float4 emission; // emission.xyz, 0
    float4 material; // material.rgb, refltype
};

#define REFL_DIFF 1
#define REFL_SPEC 2
#define REFL_FRAC 3

struct Ray { float3 o; float3 d; };

#define BOX_HX	2.6
#define BOX_HY	2
#define BOX_HZ	2.8

void prepareData(Sphere* spheres, int spheresSize);
float4* runTracer(int resx, int resy, int spp);

#endif //CUTRACE_CUTRACE_CUH
