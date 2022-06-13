#pragma once
#ifndef INTEGRATOR_AD_PATHSPACE_H__
#define INTEGRATOR_AD_PATHSPACE_H__

#include "intersection.h"
#include "integratorAD.h"
#include "nanoflann.hpp"
#include <vector>

#define BDPT_MAX_THREADS 48
#define BDPT_MAX_PATH_LENGTH 50 
#define USE_BOUNDARY_NEE

struct Ray;
struct RayAD;
struct Edge;

struct EdgeEvaluationRecord {
    Intersection its1, its2;

    SpectrumAD value0;
    int idx_pixel;

    FloatAD value1;
};

struct GuidingOptions {
    int type;                       // [Type 0] Direct Guiding [Type 1] Indirect Guiding (KNN) [Type 2] Indirect Guiding (Radius Search)
    std::vector<int> params;

    size_t num_cam_path;
    size_t num_light_path;
    float search_radius;            // Only necessary when using [Type 2] guiding
    bool quiet;

    GuidingOptions(int type, const std::vector<int> &params):
        type(type), params(params), num_cam_path(0), num_light_path(0), search_radius(0.0), quiet(false) {};
    GuidingOptions(int type, const std::vector<int> &params, size_t num_cam_path, size_t num_light_path):
        type(type), params(params),
        num_cam_path(num_cam_path), num_light_path(num_light_path), search_radius(0.0), quiet(false) {};
    GuidingOptions(int type, const std::vector<int> &params, size_t num_cam_path, size_t num_light_path, float search_radius):
        type(type), params(params),
        num_cam_path(num_cam_path), num_light_path(num_light_path), search_radius(search_radius), quiet(false) {};
};

struct MapNode {
    Intersection its;
    Spectrum val;           // radiance/importance
    int depth;              // To seperate direct from indirect (if needed)
};

template <typename T>
struct PointCloud {
    struct Point
    {
        T  x,y,z;
    };

    std::vector<Point>  pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

template <typename T>
using KDtree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<T, PointCloud<T> >, PointCloud<T>, 3>;

struct IntegratorAD_PathSpace : IntegratorAD {
    mutable long long m_taskId[BDPT_MAX_THREADS];
    mutable omp_lock_t* m_pixelLock; // OpenMP lock for each pixel (height and weight, but not length of the sequence)
    mutable int m_num_pixelLock = -1; // -1 for RenderMode::MEMORY_DUPLE, number of pixels for RenderMode::MEMORY_LOCK

    void init_pixel_lock(int a_num_pixelLock) const;
    void destroy_pixel_lock() const;
    int set_pixel_lock(int idx_pixel, int idx_thread=0) const;
    void unset_pixel_lock(int idx_pixel) const;

    /********************
     *  For direct edges
     ********************/
    virtual std::pair<int,int> evalEdgeDirect(const Scene &scene, int shape_id, const Edge &rEdge, const RayAD &edgeRay,
                               RndSampler *sampler, int max_bounces, EdgeEvaluationRecord &eRec,
                               std::pair<int, Spectrum>* record, std::tuple<int, Spectrum, Float, int>* record_trans = nullptr) const;

    virtual void preprocessDirect(const Scene &scene, const std::vector<int> &params, int max_bounces, ptr<float> data, bool quiet) const;
    virtual void renderEdgesDirect(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient = ptr<float>()) const;
    /********************
     *  For indirect edges
     ********************/

    virtual void evalEdge(const Scene &scene, int shape_id, const Edge &rEdge, const RayAD &edgeRay, RndSampler *sampler, EdgeEvaluationRecord &eRec) const;

    virtual void buildPhotonMap(const Scene &scene, const GuidingOptions& opts, int max_bounces,
                                std::vector<MapNode> &rad_nodes, std::vector<MapNode> &imp_nodes) const;

    virtual int queryPhotonMap(const KDtree<Float> &indices, const GuidingOptions& opts, const Float* query_point,
                               size_t* matched_indices, Float& matched_dist_sqr, bool type) const;

    virtual void preprocessIndirect(const Scene &scene, const GuidingOptions& opts, int max_bounces,
                                    const std::vector<MapNode> &rad_nodes, const KDtree<Float> &rad_indices,
                                    const std::vector<MapNode> &imp_nodes, const KDtree<Float> &imp_indices,
                                    ptr<float> data, bool quiet) const;

    virtual void preprocess(const Scene &scene, int max_bounces, const GuidingOptions& opts, ptr<float> data) const;
    virtual int radiance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces,
                         Spectrum *ret, std::tuple<Spectrum, Float, int> *ret_trans = nullptr) const;

    virtual std::pair<int, int> weightedImportance(const Scene& scene, RndSampler* sampler, const Intersection& its, int max_depth, const Spectrum *weight,
                                   std::pair<int, Spectrum>* ret, std::tuple<int, Spectrum, Float, int>* ret_trans = nullptr) const;

    virtual void traceRayFromEdgeSegement(const Scene &scene, const EdgeEvaluationRecord& eRec, Float edgePdf, int max_depth, RndSampler *sampler,
                                          std::vector<Spectrum> &image, std::vector<Spectrum>* p_transient = nullptr) const;

    /********************
     *  Main functions
     ********************/

    virtual void renderEdges(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient = ptr<float>()) const;
    virtual void render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient = ptr<float>()) const;
};

#endif //INTEGRATOR_AD_PATHSPACE_H__
