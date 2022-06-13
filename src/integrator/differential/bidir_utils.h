#pragma once
#ifndef BIDIRECTIONAL_PATH_TRACER_UTILS_H__
#define BIDIRECTIONAL_PATH_TRACER_UTILS_H__

#include "intersection.h"
#include "intersectionAD.h"

namespace bidir {
    struct PathNode {
        Intersection its;
        Spectrum throughput;
        Vector wo;
        Float pdf0, pdf1;

        mutable Intersection its1;
        Float G1;
        Float w;
        Float opd; // accumulated optical path distance
    };
    struct PathNodeAD {
        IntersectionAD itsAD;
        Intersection its;

        SpectrumAD throughput;
        FloatAD J;
        VectorAD wo;
        Float pdf0, pdf1;

        mutable Intersection its1;
        Float G1;
        Float w;
        FloatAD opd; // accumulated optical path distance
    };

    inline Float mis_ratio(const Float &pdf0, const Float &pdf1) {
        Float ret = pdf0/pdf1;
        ret *= ret;
        return ret;
    }

    int buildPath(const Scene &scene, RndSampler *sampler, int max_depth, bool importance, PathNode *path);
    int buildPathAD(const Scene &scene, RndSampler *sampler, int max_depth, bool importance, PathNodeAD *path);

    void preprocessPath(int pathLength, bool fix_first, PathNode *path);
    void preprocessPathAD(int pathLength, bool fix_first, PathNodeAD *path);

    Spectrum evalSegment(const Scene &scene, const Intersection &its0, const Intersection &its1, bool useEmission = false);
    SpectrumAD evalSegmentAD(const Scene &scene, const IntersectionAD &its0, const IntersectionAD &its1, bool useEmission = false);

    int radiance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces,
                  PathNode *camPath, PathNode *lightPath, Spectrum *ret, std::tuple<Spectrum, Float, int> *ret_trans = nullptr);

    std::pair<int, int> weightedImportance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces, int pix_id,
                           PathNode* camPath, PathNode* lightPath, const Spectrum *weight,
                           std::pair<int, Spectrum>* ret, std::tuple<int, Spectrum, Float, int>* ret_trans = nullptr);

} //namespace bidir

#endif //BIDIRECTIONAL_PATH_TRACER_UTILS_H__
