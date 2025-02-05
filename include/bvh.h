#ifndef BVH_H
#define BVH_H

#include "hitable.h"
#include "ray.h"
#include <algorithm> 

// Simple Axis-Aligned Bounding Box (AABB)
class aabb {
public:
    __host__ __device__ aabb() {}
    __host__ __device__ aabb(const vec3& min, const vec3& max) : _min(min), _max(max) {}

    __device__ bool hit(const ray& r, float tmin, float tmax) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (_min[a] - r.origin()[a]) * invD;
            float t1 = (_max[a] - r.origin()[a]) * invD;
            if (invD < 0.0f)
                std::swap(t0, t1);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin)
                return false;
        }
        return true;
    }

    vec3 _min;
    vec3 _max;
};

// Function to combine two AABBs
__device__ aabb surrounding_box(aabb box0, aabb box1) {
    vec3 small(fminf(box0._min.x(), box1._min.x()),
               fminf(box0._min.y(), box1._min.y()),
               fminf(box0._min.z(), box1._min.z()));
    vec3 big(fmaxf(box0._max.x(), box1._max.x()),
             fmaxf(box0._max.y(), box1._max.y()),
             fmaxf(box0._max.z(), box1._max.z()));
    return aabb(small, big);
}

// BVH Node
class bvh_node : public hitable {
public:
    __host__ __device__ bvh_node() {}
    __host__ __device__ bvh_node(hitable** l, int n);
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(aabb& box) const;

    hitable* left;
    hitable* right;
    aabb box;
};

__device__ bool bvh_node::bounding_box(aabb& b) const {
    b = box;
    return true;
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (box.hit(r, t_min, t_max)) {
        hit_record left_rec, right_rec;
        bool hit_left = left->hit(r, t_min, t_max, left_rec);
        bool hit_right = right->hit(r, t_min, t_max, right_rec);
        if (hit_left && hit_
