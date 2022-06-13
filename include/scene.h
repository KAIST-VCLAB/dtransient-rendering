#pragma once
#ifndef SCENE_H__
#define SCENE_H__

#include "ptr.h"
#include "utils.h"
#include "ray.h"
#include "rayAD.h"
#include "intersection.h"
#include "intersectionAD.h"
#include "cameratransient.h"
#include "emitter.h"
#include "shape.h"
#include "bsdf.h"
#include "medium.h"
#include "sampler.h"
#include "phase.h"
#include "pmf.h"
#include "edge_manager.h"
#include "../src/edge_manager/pathspace.h"
#include <vector>
#include <array>
#include <memory>
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>

struct Scene {
    Scene(const CameraTransient &camera, 
          const std::vector<const Shape*> &shapes,
          const std::vector<const BSDF*> &bsdfs,
          const std::vector<const Emitter*> &area_lights,
          const std::vector<const PhaseFunction*> &phases,
          const std::vector<const Medium*> &media,
          bool use_hierarchy = true);
    ~Scene();

    void initEdges(const Eigen::Array<Float, -1, 1> &shapeWeights);

    // For pyBind
    void initEdgesPy(ptr<float> shapeWeights) {
        initEdges(Eigen::Map<Eigen::Array<float, -1, 1> >(shapeWeights.get(), shape_list.size(), 1).cast<Float>());
    }

    void initPathSpaceEdges(const Eigen::Array<Float, -1, 1> &shapeWeights,
                            const Vector3i &direct_dims, const Eigen::Array<Float, -1, 1> &direct_data,
                            const Vector3i &indirect_dims, const Eigen::Array<Float, -1, 1> &indirect_data,
                            bool verbose = false);

    // For pyBind
    void initPathSpaceEdgesPy1_1(const std::vector<int> &dims, ptr<float> data) {
        assert(dims.size() == 3);
        initPathSpaceEdges(
            Eigen::Array<Float, -1, 1>::Ones(shape_list.size()),
            Vector3i(dims[0], dims[1], dims[2]), Eigen::Map<Eigen::Array<float, -1, 1> >(data.get(), dims[0]*dims[1]*dims[2], 1).cast<Float>(),
            Vector3i(dims[0], dims[1], dims[2]), Eigen::Map<Eigen::Array<float, -1, 1> >(data.get(), dims[0]*dims[1]*dims[2], 1).cast<Float>(),
            false
        );
    }

    void initPathSpaceEdgesPy1_2(const std::vector<int> &direct_dims, ptr<float> direct_data,
                                 const std::vector<int> &indirect_dims, ptr<float> indirect_data) {
        assert(direct_dims.size() == 3 && indirect_dims.size() == 3);
        initPathSpaceEdges(
            Eigen::Array<Float, -1, 1>::Ones(shape_list.size()),
            Vector3i(direct_dims[0], direct_dims[1], direct_dims[2]),
            Eigen::Map<Eigen::Array<float, -1, 1> >(direct_data.get(), direct_dims[0]*direct_dims[1]*direct_dims[2], 1).cast<Float>(),
            Vector3i(indirect_dims[0], indirect_dims[1], indirect_dims[2]),
            Eigen::Map<Eigen::Array<float, -1, 1> >(indirect_data.get(), indirect_dims[0]*indirect_dims[1]*indirect_dims[2], 1).cast<Float>(),
            false
        );
    }

    void initPathSpaceEdgesPy2_1(ptr<float> shapeWeights, const std::vector<int> &dims, ptr<float> data) {
        assert(dims.size() == 3);
        initPathSpaceEdges(
            Eigen::Map<Eigen::Array<float, -1, 1> >(shapeWeights.get(), shape_list.size(), 1).cast<Float>(),
            Vector3i(dims[0], dims[1], dims[2]), Eigen::Map<Eigen::Array<float, -1, 1> >(data.get(), dims[0]*dims[1]*dims[2], 1).cast<Float>(),
            Vector3i(dims[0], dims[1], dims[2]), Eigen::Map<Eigen::Array<float, -1, 1> >(data.get(), dims[0]*dims[1]*dims[2], 1).cast<Float>(),
            false
        );
    }

    void initPathSpaceEdgesPy2_2(ptr<float> shapeWeights,
                               const std::vector<int> &direct_dims, ptr<float> direct_data,
                               const std::vector<int> &indirect_dims, ptr<float> indirect_data) {
        assert(direct_dims.size() == 3 && indirect_dims.size() == 3);
        initPathSpaceEdges(
            Eigen::Map<Eigen::Array<float, -1, 1> >(shapeWeights.get(), shape_list.size(), 1).cast<Float>(),
            Vector3i(direct_dims[0], direct_dims[1], direct_dims[2]),
            Eigen::Map<Eigen::Array<float, -1, 1> >(direct_data.get(), direct_dims[0]*direct_dims[1]*direct_dims[2], 1).cast<Float>(),
            Vector3i(indirect_dims[0], indirect_dims[1], indirect_dims[2]),
            Eigen::Map<Eigen::Array<float, -1, 1> >(indirect_data.get(), indirect_dims[0]*indirect_dims[1]*indirect_dims[2], 1).cast<Float>(),
            false
        );
    }
    CameraTransient camera;
    std::vector<const Shape*> shape_list;
    std::vector<const BSDF*> bsdf_list;
    std::vector<const Emitter*> emitter_list;
    std::vector<const PhaseFunction*> phase_list;
    std::vector<const Medium*> medium_list;

    // Embree handles
    RTCDevice embree_device;
    RTCScene embree_scene;

    // Light sampling
    int num_lights;
    DiscreteDistribution light_distrb;

    // Edge sampling related
    EdgeManager *ptr_edgeManager;
    PathSpaceEdgeManager *ptr_psEdgeManager;
    bool use_hierarchy;

    // Point sampling
    DiscreteDistribution shape_distrb;
    inline Float getArea() const { return shape_distrb.getSum(); }

    // Simple visibility test (IGNORING null interfaces!)
    bool isVisible(const Vector &p, bool pOnSurface, const Vector &q, bool qOnSurface) const;

    // Path Tracer
    bool rayIntersect(const Ray &ray, bool onSurface, Intersection& its, Float* p_opt_path_dist = nullptr) const;
    bool rayIntersectAD(const RayAD &ray, bool onSurface, IntersectionAD& its, FloatAD* p_opt_path_dist = nullptr) const;

    Spectrum sampleEmitterDirect(const Intersection &its, const Vector2 &rnd_light, RndSampler* sampler, Vector& wo, Float &pdf, Vector *pos_emitter = nullptr) const;
    SpectrumAD sampleEmitterDirectAD(const IntersectionAD &its, const Vector2 &rnd_light, RndSampler* sampler, VectorAD& wo, Float &pdf, Float *psInfo = NULL) const;

    // Volume Path Tracer
    Spectrum rayIntersectAndLookForEmitter(const Ray &ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                           Intersection &its, Float& pdf_nee) const;

    SpectrumAD rayIntersectAndLookForEmitterAD(const RayAD &ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                               IntersectionAD &its, Float& pdf_nee, IntersectionAD *itsFar = nullptr) const;

    Spectrum sampleAttenuatedEmitterDirect(const Intersection& its, const Vector2 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                           Vector& wo, Float& pdf, bool flag = false) const; // wo in local space

    SpectrumAD sampleAttenuatedEmitterDirectAD(const IntersectionAD& its, const Vector2 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                               VectorAD& wo, Float& pdf) const; // wo in local space

    Spectrum sampleAttenuatedEmitterDirect(const Vector &pscatter, const Vector2 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                           Vector& wo, Float& pdf) const; // wo in *world* space

    SpectrumAD sampleAttenuatedEmitterDirectAD(const VectorAD &pscatter, const Vector2 &rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                               VectorAD& wo, Float& pdf) const; // wo in *world* space

    Float evalTransmittance(const Ray& ray, bool onSurface, const Medium* ptr_medium, Float remaining, RndSampler* sampler, int max_interactions) const;
    FloatAD evalTransmittanceAD(const RayAD& ray, bool onSurface, const Medium* ptr_medium, FloatAD remaining, RndSampler* sampler, int max_interactions) const;

    Float pdfEmitterSample(const Intersection& its) const;
    Float pdfEmitterSample(const IntersectionAD& its) const;

    inline const Edge* sampleEdge(const Vector& p, const Frame* ptr_frame, Float& rnd, int& shape_id, Float& pdf) const {
        return ptr_edgeManager->sampleSecondaryEdge(p, ptr_frame, rnd, shape_id, pdf);
    }

    inline const Edge* sampleEdge(const Vector& p, const Frame* ptr_frame, Float rnd, int& shape_id, Float &t, Float& pdf) const {
        t = rnd;
        const Edge* ret = sampleEdge(p, ptr_frame, t, shape_id, pdf);
        if (ret != nullptr)
            pdf /= ret->length;
        return ret;
    }

    Spectrum sampleEmitterPosition(const Vector2 &rnd_light, Intersection& its, Float *pdf = nullptr) const;
    SpectrumAD sampleEmitterPosition(const Vector2 &rnd_light, IntersectionAD& its, FloatAD& J, Float *pdf = nullptr) const;

    Float sampleAttenuatedSensorDirect(const Intersection& its, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const;
    Float sampleAttenuatedSensorDirect(const Vector& p, const Medium* ptr_med, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const;


    Vector2i samplePosition(const Vector2 &rnd2, PositionSamplingRecord &pRec) const;
    Vector2i samplePositionAD(const Vector2 &rnd2, PositionSamplingRecordAD &pRec) const;

    void getPoint(const Intersection &its, VectorAD &x, VectorAD &n, FloatAD &J) const;
    void getPoint(const Intersection &its, const VectorAD &p, IntersectionAD& its_AD, FloatAD &J) const;
    void getPoint(const Intersection &its, IntersectionAD& its_AD, FloatAD &J) const;
    void getPointAD(const IntersectionAD &its, VectorAD &x, VectorAD &n, FloatAD &J) const;

    const Edge& sampleEdgeRay(const Vector &rnd3, int &shape_id, RayAD &ray, Float &pdf) const;
    const Edge& sampleEdgeRayDirect(const Vector &rnd3, int &shape_id, RayAD &ray, Float &pdf) const;
};
#endif
