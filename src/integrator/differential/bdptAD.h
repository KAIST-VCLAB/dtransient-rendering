#pragma once
#ifndef BIDIRECTIONAL_PATH_TRACER_AD_H__
#define BIDIRECTIONAL_PATH_TRACER_AD_H__

#include "integratorADps.h"
#include "bidir_utils.h"

struct BidirectionalPathTracerAD : IntegratorAD_PathSpace {

    std::pair<Spectrum,int> pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y,
                        std::tuple<Spectrum, Float, int> *ret_trans = nullptr) const;

    SpectrumAD pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y, ptr<float> temp_hist = ptr<float>()) const;

    void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient = ptr<float>()) const;
    std::string getName() const { return "bdptAD"; }

    mutable bidir::PathNodeAD m_pathAD[2*BDPT_MAX_THREADS][BDPT_MAX_PATH_LENGTH];
    mutable bidir::PathNode m_path[2*BDPT_MAX_THREADS][BDPT_MAX_PATH_LENGTH];
    mutable Spectrum m_rad[BDPT_MAX_THREADS][BDPT_MAX_PATH_LENGTH];
    mutable Float m_pathDist[BDPT_MAX_THREADS][BDPT_MAX_PATH_LENGTH];

    mutable long long rejectedSamples;
    mutable long long outtimeSamples; // number of samples outside of sensing time interval

};

#endif //BIDIRECTIONAL_PATH_TRACER_AD_H__
