#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hitable.h"

// Function to compute a random point in a unit sphere
__device__ vec3 random_in_unit_sphere() {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state)) - vec3(1, 1, 1);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
};

class lambertian : public material {
public:
    __host__ __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere();
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }
    vec3 albedo;
};

class metal : public material {
public:
    __host__ __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    vec3 albedo;
    float fuzz;
};

#endif
