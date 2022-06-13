#pragma once
#ifndef PSDR_BDPTADBB_H
#define PSDR_BDPTADBB_H

#include "bdptAD.h"

struct BidirectionalPathTracerADBiBd: BidirectionalPathTracerAD {
    int radiance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces,
                 Spectrum *ret, std::tuple<Spectrum, Float, int> *ret_trans = nullptr) const;
    std::pair<int, int> weightedImportance(const Scene& scene, RndSampler* sampler, const Intersection& its, int max_bounces,const Spectrum *weight,
                                           std::pair<int, Spectrum>* ret, std::tuple<int, Spectrum, Float, int>* ret_trans = nullptr) const;
};

#endif //PSDR_BDPTADBB_H

