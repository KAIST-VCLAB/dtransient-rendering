#pragma once
#ifndef INTEGRATOR_H__
#define INTEGRATOR_H__

#include "ptr.h"
#include "fwd.h"

struct Scene;

enum RenderMode{
    MEMORY_DUPLE,
    MEMORY_LOCK
};

struct RenderOptions {
    RenderOptions(uint64_t seed, int num_samples, int max_bounces, int num_samples_primary_edge, int num_samples_secondary_edge, bool quiet, int mode = 0, float ddistCoeff = 0.0f) 
        : seed(seed), num_samples(num_samples), max_bounces(max_bounces)
        , num_samples_primary_edge(num_samples_primary_edge), num_samples_secondary_edge(num_samples_secondary_edge)
        , quiet(quiet), mode(RenderMode(mode)), ddistCoeff(ddistCoeff) 
    {
        num_samples_secondary_edge_direct = num_samples_secondary_edge;
        num_samples_secondary_edge_indirect = num_samples_secondary_edge;
        grad_threshold = 1e8f;
        primary_delta = 1e-4f; 
    }

    RenderOptions(const RenderOptions &options)
        : seed(options.seed), num_samples(options.num_samples), max_bounces(options.max_bounces)
        , num_samples_primary_edge(options.num_samples_primary_edge), num_samples_secondary_edge(options.num_samples_secondary_edge)
        , quiet(options.quiet), mode(options.mode), ddistCoeff(options.ddistCoeff)
        , num_samples_secondary_edge_direct(options.num_samples_secondary_edge_direct), num_samples_secondary_edge_indirect(options.num_samples_secondary_edge_indirect)
        , grad_threshold(options.grad_threshold), primary_delta(options.primary_delta) {} 

    uint64_t seed;
    int num_samples;
    int max_bounces;
    int num_samples_primary_edge;       // Camera ray
    int num_samples_secondary_edge;     // Secondary (i.e., reflected/scattered) rays
    bool quiet;
    RenderMode mode;                           
    Float ddistCoeff;

    // For path-space differentiable rendering
    int num_samples_secondary_edge_direct;
    int num_samples_secondary_edge_indirect;
    float grad_threshold;
    Float primary_delta; 
};

struct Integrator {
    virtual ~Integrator() {}
    virtual void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient = ptr<float>()) const = 0;
};

#endif //INTEGRATOR_H__
