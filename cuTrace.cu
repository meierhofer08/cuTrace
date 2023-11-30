//
// Created by MM on 23.11.2023.
//

#include <cstdio>
#include "cuda_runtime_api.h"

#include "cuTrace.cuh"

// -------------- VECTOR FUNCTIONS --------------
// float2

__device__ float2 operator+(const float2 a, const float2 b) {
    return {a.x + b.x, a.y + b.y};
}

__device__ float2 operator+(const float s, const float2 f) {
    return {f.x + s, f.y + s};
}

__device__ float2 operator-(const float2 a, const float2 b) {
    return {a.x - b.x, a.y - b.y};
}

__device__ float2 operator-(const float2 f, const float s) {
    return {f.x - s, f.y - s};
}

__device__ float2 operator*(const float2 a, const float2 b) {
    return {a.x * b.x, a.y * b.y};
}

__device__ float2 operator*(const float s, const float2 f) {
    return {f.x * s, f.y * s};
}

__device__ float2 operator*(const float2 f, const float s) {
    return {f.x * s, f.y * s};
}

__device__ float2 operator/(const float2 a, const float2 b) {
    return {a.x / b.x, a.y / b.y};
}

// float3

__device__ float3 operator+(const float3 a, const float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ float3 operator+(const float s, const float3 f) {
    return {f.x + s, f.y + s, f.z + s};
}

__device__ float3 operator-(const float3 a, const float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ float3 operator-(const float3 f, const float s) {
    return {f.x + s, f.y + s, f.z - s};
}

__device__ float3 operator*(const float3 a, const float3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ float3 operator*(const float s, const float3 f) {
    return {f.x * s, f.y * s, f.z * s};
}

__device__ float3 operator*(const float3 f, const float s) {
    return {f.x * s, f.y * s, f.z * s};
}

__device__ float3 operator/(const float3 a, const float3 b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ float3 operator/(const float3 f, const float s) {
    return {f.x / s, f.y / s, f.z / s};
}

__device__ bool operator==(const float3& a, const float3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device__ float3 make_float3(const float4& f4) {
    return {f4.x, f4.y, f4.z};
}

__device__ float dot(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3 vec) {
    float len = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
    return {vec.x / len, vec.y / len, vec.z / len};
}

__device__ float3 cross(const float3 a, const float3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ float3 reflect(const float3 vec, const float3 normal) {
    return vec - normal * 2 * dot(normal, vec);
}

__device__ uint3 operator*(const uint3 u, const unsigned s) {
    return {u.x * s, u.y * s, u.z * s};
}

__device__ float3 operator*(const uint3 u, const float s) {
    return {(float) u.x * s, (float) u.y * s, (float) u.z * s};
}

// -------------- END VECTOR FUNCTIONS --------------

Sphere* g_spheres = nullptr;
int allSpheresSize = 0;

#define MAX_SPHERES 30
__device__ const float PI = 3.141592653589793f;
#define cam Ray{float3{0, 0.52, 7.4}, normalize(float3{0, -0.06, -1})}
#define cx normalize(cross(cam.d, abs(cam.d.y) < 0.9 ? float3{0, 1, 0} : float3{0, 0, 1}))
#define cy cross(cx, cam.d)
#define sdim float2{0.036, 0.024};    // sensor size (36 x 24 mm)


__device__ float3 rand01(uint3 seed) {                   // pseudo-random number generator
    for (int i = 3; i-- > 0;) {
        uint3 oldSeed = seed;
        seed.x = ((oldSeed.x >> 8U) ^ oldSeed.y);
        seed.y = ((oldSeed.y >> 8U) ^ oldSeed.z);
        seed.z = ((oldSeed.z >> 8U) ^ oldSeed.x);
        seed = seed * 1103515245U;
    }
    return seed * (float) (1.0 / float(0xffffffffU));
}

__device__ float clamp(float value, float min, float max) {
    if (value < min) {
        return min;
    } else if (value > max) {
        return max;
    } else {
        return value;
    }
}

__global__ void runCuTracePass(Sphere* spheres, const unsigned spheresSize, float4* radiance, unsigned pass, int spp, unsigned resx, unsigned resy) {
    unsigned uid = blockIdx.x * blockDim.x + threadIdx.x;
    if (uid >= resx * resy) {
        return;
    }
    unsigned yPos = uid / resx;
    unsigned xPos = uid % resx;
    float2 fpix{(float) xPos, (float) yPos};
    float2 fimgdim{(float) resx, (float) resy};

    __shared__ float4 sf_spheres[MAX_SPHERES * 3];
    auto* s_spheres = (Sphere*) sf_spheres;
    unsigned idInBlock = uid % blockDim.x;
    auto* f_spheres = (float4*) spheres;
    if (idInBlock < spheresSize * 3) {
        sf_spheres[idInBlock] = f_spheres[idInBlock];
    }
    __syncthreads();

    //-- sample sensor
    float3 rnd2 = 2 * rand01(uint3{xPos, yPos, pass});   // vvv tent filter sample
    float2 tent{rnd2.x < 1 ? sqrt(rnd2.x) - 1 : 1 - sqrt(2 - rnd2.x),
                rnd2.y < 1 ? sqrt(rnd2.y) - 1 : 1 - sqrt(2 - rnd2.y)};
    float2 sens =
            ((fpix + 0.5 * (0.5 + float2{(float) ((pass / 2) % 2), (float) (pass % 2)} + tent)) / fimgdim - 0.5) *
    sdim;
    float3 spos = cam.o + cx * sens.x + cy * sens.y, lc =
            cam.o + cam.d * 0.035;           // sample on 3d sensor plane
    float3 accrad{0, 0, 0}, accmat{1, 1, 1};
    Ray r{lc, normalize(lc - spos)};          // construct ray
    Sphere obj{{},
               {},
               {}};

    //-- loop over ray bounces
    for (int depth = 0, maxDepth = 12; depth < maxDepth; depth++) {
        float d, inf = 1e20, t = inf, eps = 1e-4;   // intersect ray with scene
        for (int i = spheresSize; i-- > 0;) {
            Sphere& s = s_spheres[i];                  // perform intersection test
            float3 oc = s.center - r.o;      // Solve t^2*d.d + 2*t*(o-s).d + (o-s).(o-s)-r^2 = 0
            float b = dot(oc, r.d), det = b * b - dot(oc, oc) + s.radius * s.radius;
            if (det < 0) continue; else det = sqrt(det);
            d = (d = b - det) > eps ? d : ((d = b + det) > eps ? d : inf);
            if (d < t) {
                t = d;
                obj = s;
            }
        }
        if (t < inf) {// object hit
            float3 x = r.o + r.d * t, n = normalize(x - obj.center), nl =
                    dot(n, r.d) < 0 ? n : n * (-1);
            float p = max(max(obj.material.x, obj.material.y), obj.material.z);  // max reflectance
            accrad = accrad + accmat * obj.emission;
            accmat = accmat * obj.material;
            float3 rdir = reflect(r.d, n), rnd = rand01(uint3{xPos, yPos, pass * maxDepth + depth});
            if (depth > 5) {
                if (rnd.z >= p) break;  // Russian Roulette ray termination
                else accmat = accmat / p;       // Energy compensation of surviving rays
            }
            //-- Ideal DIFFUSE reflection
            if (obj.refltype == REFL_DIFF) {
                float r1 = 2 * PI * rnd.x, r2 = rnd.y, r2s = sqrt(r2);  // cosine-weighted importance sampling
                float3 w = nl, u = normalize(
                        (cross(abs(w.x) > 0.1 ? float3{0, 1, 0} : float3{1, 0, 0}, w))), v = cross(
                        w, u);
                r = Ray{x, normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2))};
            }
                //-- Ideal SPECULAR reflection
            else if (obj.refltype == REFL_SPEC) {
                r = Ray{x, rdir};
            }
                //-- Ideal dielectric REFRACTION
            else if (obj.refltype == REFL_FRAC) {
                bool into = n == nl;
                float cos2t, nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.d, nl);
                if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) >= 0) {  // Fresnel reflection/refraction
                    float3 tdir = normalize(r.d * nnt - n * ((into ? 1.f : -1.f) * (ddn * nnt + sqrt(cos2t))));
                    float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : dot(tdir, n));
                    float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = 0.25 + 0.5 * Re, RP =
                            Re / P, TP =
                            Tr / (1 - P);
                    r = Ray{x, rnd.x < P ? rdir : tdir};    // pick reflection with probability P
                    accmat = accmat * (rnd.x < P ? RP : TP);         // energy compensation
                } else r = Ray{x, rdir};                    // Total internal reflection
            }
        }
    }

    float4 accRadiance{0, 0, 0, 0};
    if (pass > 0) {
        accRadiance = radiance[uid];
    }
    accRadiance.x += accrad.x / spp;
    accRadiance.y += accrad.y / spp;
    accRadiance.z += accrad.z / spp;

    accRadiance.x = pow(clamp(accRadiance.x, 0, 1), 0.45) * 255 + 0.5;
    accRadiance.y = pow(clamp(accRadiance.y, 0, 1), 0.45) * 255 + 0.5;
    accRadiance.z = pow(clamp(accRadiance.z, 0, 1), 0.45) * 255 + 0.5;
    radiance[uid] = accRadiance;
}

__global__ void
runCuTraceOffset(Sphere* spheres, const unsigned spheresSize, float4* radiance, int spp, unsigned resx, unsigned resy, unsigned batchSize,
                 unsigned offset) {
    unsigned uid = blockIdx.x * blockDim.x + threadIdx.x;
    if (uid >= batchSize) {
        return;
    }
    uid += offset;
    if (uid >= resx * resy) {
        return;
    }
    unsigned yPos = uid / resx;
    unsigned xPos = uid % resx;
    float2 fpix{(float) xPos, (float) yPos};
    float2 fimgdim{(float) resx, (float) resy};
    float4 accRadiance{0, 0, 0, 0};

    __shared__ Sphere s_spheres[MAX_SPHERES];
    unsigned idInBlock = uid % blockDim.x;
    if (idInBlock < spheresSize) {
        s_spheres[idInBlock] = spheres[idInBlock];
    }
    __syncthreads();

    //-- sample sensor
    for (unsigned pass = 0; pass < spp; pass++) {
        float3 rnd2 = 2 * rand01(uint3{xPos, yPos, pass});   // vvv tent filter sample
        float2 tent{rnd2.x < 1 ? sqrt(rnd2.x) - 1 : 1 - sqrt(2 - rnd2.x),
                    rnd2.y < 1 ? sqrt(rnd2.y) - 1 : 1 - sqrt(2 - rnd2.y)};
        float2 sens =
                ((fpix + 0.5 * (0.5 + float2{(float) ((pass / 2) % 2), (float) (pass % 2)} + tent)) / fimgdim - 0.5) *
        sdim;
        float3 spos = cam.o + cx * sens.x + cy * sens.y, lc =
                cam.o + cam.d * 0.035;           // sample on 3d sensor plane
        float3 accrad{0, 0, 0}, accmat{1, 1, 1};
        Ray r{lc, normalize(lc - spos)};          // construct ray
        Sphere obj{{},
                   {},
                   {}};

        //-- loop over ray bounces
        for (int depth = 0, maxDepth = 12; depth < maxDepth; depth++) {
            float d, inf = 1e20, t = inf, eps = 1e-4;   // intersect ray with scene
            for (int i = spheresSize; i-- > 0;) {
                Sphere& s = s_spheres[i];                  // perform intersection test
                float3 oc = s.center - r.o;      // Solve t^2*d.d + 2*t*(o-s).d + (o-s).(o-s)-r^2 = 0
                float b = dot(oc, r.d), det = b * b - dot(oc, oc) + s.radius * s.radius;
                if (det < 0) continue; else det = sqrt(det);
                d = (d = b - det) > eps ? d : ((d = b + det) > eps ? d : inf);
                if (d < t) {
                    t = d;
                    obj = s;
                }
            }
            if (t < inf) {// object hit
                float3 x = r.o + r.d * t, n = normalize(x - obj.center), nl =
                        dot(n, r.d) < 0 ? n : n * (-1);
                float p = max(max(obj.material.x, obj.material.y), obj.material.z);  // max reflectance
                accrad = accrad + accmat * obj.emission;
                accmat = accmat * obj.material;
                float3 rdir = reflect(r.d, n), rnd = rand01(uint3{xPos, yPos, pass * maxDepth + depth});
                if (depth > 5) {
                    if (rnd.z >= p) break;  // Russian Roulette ray termination
                    else accmat = accmat / p;       // Energy compensation of surviving rays
                }
                //-- Ideal DIFFUSE reflection
                if (obj.refltype == REFL_DIFF) {
                    float r1 = 2 * PI * rnd.x, r2 = rnd.y, r2s = sqrt(r2);  // cosine-weighted importance sampling
                    float3 w = nl, u = normalize(
                            (cross(abs(w.x) > 0.1 ? float3{0, 1, 0} : float3{1, 0, 0}, w))), v = cross(
                            w, u);
                    r = Ray{x, normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2))};
                }
                    //-- Ideal SPECULAR reflection
                else if (obj.refltype == REFL_SPEC) {
                    r = Ray{x, rdir};
                }
                    //-- Ideal dielectric REFRACTION
                else if (obj.refltype == REFL_FRAC) {
                    bool into = n == nl;
                    float cos2t, nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = dot(r.d, nl);
                    if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) >= 0) {  // Fresnel reflection/refraction
                        float3 tdir = normalize(r.d * nnt - n * ((into ? 1.f : -1.f) * (ddn * nnt + sqrt(cos2t))));
                        float a = nt - nc, b = nt + nc, R0 = a * a / (b * b), c = 1 - (into ? -ddn : dot(tdir, n));
                        float Re = R0 + (1 - R0) * c * c * c * c * c, Tr = 1 - Re, P = 0.25 + 0.5 * Re, RP =
                                Re / P, TP =
                                Tr / (1 - P);
                        r = Ray{x, rnd.x < P ? rdir : tdir};    // pick reflection with probability P
                        accmat = accmat * (rnd.x < P ? RP : TP);         // energy compensation
                    } else r = Ray{x, rdir};                    // Total internal reflection
                }
            }
        }

        accRadiance.x += accrad.x / spp;
        accRadiance.y += accrad.y / spp;
        accRadiance.z += accrad.z / spp;
    }

    accRadiance.x = pow(clamp(accRadiance.x, 0, 1), 0.45) * 255 + 0.5;
    accRadiance.y = pow(clamp(accRadiance.y, 0, 1), 0.45) * 255 + 0.5;
    accRadiance.z = pow(clamp(accRadiance.z, 0, 1), 0.45) * 255 + 0.5;
    radiance[uid] = accRadiance;
}

__global__ void showNoise(Sphere* spheres, float4* radiance, unsigned pass, int spp, int resx, int resy) {
    unsigned uid = blockIdx.x * blockDim.x + threadIdx.x;
    if (uid > resx * resy) {
        return;
    }
    unsigned y = uid / resx;
    unsigned x = uid % resx;

    // output noise
    float3 randomRadiance = rand01({x, y, pass});
    float4 pixRadiance{randomRadiance.x, randomRadiance.y, randomRadiance.z, 0};
    if (pass == spp - 1) {
        pixRadiance.x = pow(clamp(pixRadiance.x, 0, 1), 0.45) * 255 + 0.5;
        pixRadiance.y = pow(clamp(pixRadiance.y, 0, 1), 0.45) * 255 + 0.5;
        pixRadiance.z = pow(clamp(pixRadiance.z, 0, 1), 0.45) * 255 + 0.5;
    }
    radiance[uid] = pixRadiance;
}

__global__ void printSphereInfo(Sphere* spheres, const unsigned spheresSize, int pass, int spp) {
    unsigned uid = blockIdx.x * blockDim.x + threadIdx.x;
    if (uid >= spheresSize) {
        return;
    }
    Sphere* sphere = spheres + uid;

    std::printf("----\n"
                "Info about sphere %d:\n"
                "Sphere properties: Center (%f, %f, %f) Radius %f\n"
                "Sphere emission (%f, %f, %f)\n"
                "Sphere color (%f, %f, %f) Reflection type %f\n"
                "----\n\n",
                uid, sphere->center.x, sphere->center.y, sphere->center.z, sphere->radius,
                sphere->emission.x, sphere->emission.y, sphere->emission.z,
                sphere->material.x, sphere->material.y, sphere->material.z, sphere->refltype);
}

void prepareData(Sphere* spheres, int spheresSize) {
    std::printf("Preparing data...\n");
    cudaMalloc(&g_spheres, spheresSize * sizeof(Sphere));
    cudaMemcpy(g_spheres, spheres, spheresSize * sizeof(Sphere), cudaMemcpyHostToDevice);
    allSpheresSize = spheresSize;
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
            else printf("Unknown device type\n");
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void runTracerPass(int resx, int resy, int spp, float4* g_radiance) {
    const unsigned blockSize = 128;
    for (unsigned pass = 0; pass < spp; pass++) {
        runCuTracePass<<<resx * resy / blockSize, blockSize>>>(g_spheres, allSpheresSize, g_radiance, pass, spp, resx, resy);
        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error: %s", cudaGetErrorString(err));
            return;
        }
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", spp, 100.0 * (pass+1) / spp);
    }
}

void runTracerOffset(int resx, int resy, int spp, float4* g_radiance) {
    const unsigned blockSize = 32;
    const unsigned coreOverfill = blockSize / 32;
    const unsigned batchFactor = coreOverfill * 64;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    unsigned cores = getSPcores(props);

    unsigned batchSize = cores * batchFactor;
    for (unsigned offset = 0; offset < resx * resy; offset += batchSize) {
        runCuTraceOffset<<<batchSize / blockSize, blockSize>>>(g_spheres, allSpheresSize, g_radiance, spp, resx, resy, batchSize,
                                                           offset);
        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("Error: %s", cudaGetErrorString(err));
            return;
        }
        fprintf(stderr, "\rRendering (%d spp) %5.2f%%", spp, 100.0 * ((float)(offset + batchSize) / (resx * resy)));
    }
}

float4* runTracer(int resx, int resy, int spp) {
    std::printf("Running tracer...\n");
    float4* g_radiance;
    cudaMalloc(&g_radiance, resx * resy * sizeof(float4));

    runTracerOffset(resx, resy, spp, g_radiance);
    auto* radiance = (float4*) malloc(resx * resy * sizeof(float4));
    cudaMemcpy(radiance, g_radiance, resx * resy * sizeof(float4), cudaMemcpyDeviceToHost);

    return radiance;
}
