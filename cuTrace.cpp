//
// Created by MM on 23.11.2023.
//
#include <cstdio>
#include <cstdlib>
#include <string>
#include <chrono>

#include "cuTrace.cuh"

Sphere spheres[] = {  // center.xyz, radius  |  emmission.xyz  |  color.rgb, refltype
        -1e2 - BOX_HX, 0, 0, 1e2,       0, 0, 0,  .55, .90, .20,  REFL_DIFF, // Left (DIFFUSE)
        1e2  + BOX_HX, 0, 0, 1e2,       0, 0, 0,  .75, .15, .75,  REFL_DIFF, // Right
        0, 1e2 + BOX_HY, 0, 1e2,        0, 0, 0, .75, .75, .75,  REFL_DIFF, // Top
        0,-1e2 - BOX_HY, 0, 1e2,        0, 0, 0,  .75, .75, .75,  REFL_DIFF, // Bottom
        0, 0, -1e2 - BOX_HZ, 1e2,       0, 0, 0,  .75, .75, .75,  REFL_DIFF, // Back
        0, 0, 1e2 + 3*BOX_HZ, 1e2,      0, 0, 0,  0, 0, 0,        REFL_DIFF, // Front
        -1.3, -BOX_HY + 0.8, -1.3, 0.8, 0, 0, 0,  .999,.999,.999, REFL_SPEC, // REFLECTIVE
        1.3, -BOX_HY + 0.8, -0.2, 0.8,  0, 0, 0,  .999,.999,.999, REFL_FRAC, // REFRACTIVE
        0, BOX_HY + 9.96, 0, 10,        14.7,15,13.5,  0, 0, 0,   REFL_DIFF, // Light
};

int main(int argc, char** argv) {
    std::printf("Hi\n");

    //-- parse arguments
    int spp = argc>1 ? atoi(argv[1]) : 4000;    // samples per pixel
    int resy = argc>2 ? atoi(argv[2]) : 600;
    int scene = argc>3 ? atoi(argv[3]) : 1;  // scene id (distinguish between different assignment outputs)
    int resx = resy*3/2;	                    // image resolution

    // run tracer
    auto tstart = std::chrono::system_clock::now();		                    // take start time
    prepareData(spheres, sizeof(spheres) / sizeof(Sphere));
    float3* pixels = runTracer(resx, resy, spp);
    auto tend = std::chrono::system_clock::now();

    // write output
    FILE *file = fopen((std::string("scene") + std::to_string(scene) + ".ppm").c_str(), "w");
    fprintf(file, "P3\n# spp: %d\n", spp);
    fprintf(file, "# time: %.2f s\n", std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart).count());
    fprintf(file, "%d %d %d\n", resx, resy, 255);
    for (int i = 0; i < resx * resy;i++)
        fprintf(file, "%d %d %d ", int(pixels[i].x), int(pixels[i].y), int(pixels[i].z));

    return 0;
}