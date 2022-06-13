#include "integratorADps.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"
#include "math_func.h"
#include "bidir_utils.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>

#define NUM_NEAREST_NEIGHBOR 20
#define MAX_NEAREST_NEIGHBOR 200
#define BSDF_CLAMPING_MAX 1e4
#define BSDF_CLAMPING_MIN 1e-4
// #define VERBOSE


static bidir::PathNode paths[2*BDPT_MAX_THREADS][BDPT_MAX_PATH_LENGTH];

void IntegratorAD_PathSpace::init_pixel_lock(int a_num_pixelLock) const{
    m_num_pixelLock = a_num_pixelLock;
    m_pixelLock = new omp_lock_t[m_num_pixelLock];
    for (int i = 0; i < m_num_pixelLock; i++)
        omp_init_lock(m_pixelLock + i);
}

void IntegratorAD_PathSpace::destroy_pixel_lock() const{
    for (int i = 0; i < m_num_pixelLock; i++)
        omp_destroy_lock(m_pixelLock + i);
    delete[] m_pixelLock;
}

int IntegratorAD_PathSpace::set_pixel_lock(int idx_pixel, int idx_thread) const{
    if (m_num_pixelLock == -1){
        // RenderMode::MEMORY_DUPLE
        return idx_thread;
    } else{
        // RenderMode::MEMORY_LOCK
        assert(idx_pixel >= 0);
        assert(idx_pixel < m_num_pixelLock);
        omp_set_lock(m_pixelLock + idx_pixel);
        return 0;
    }    
}

void IntegratorAD_PathSpace::unset_pixel_lock(int idx_pixel) const{
    if (m_num_pixelLock != -1){
        // RenderMode::MEMORY_LOCK
        assert(idx_pixel >= 0);
        assert(idx_pixel < m_num_pixelLock);
        omp_unset_lock(m_pixelLock + idx_pixel);
    }
}

std::pair<int,int> IntegratorAD_PathSpace::evalEdgeDirect(const Scene &scene, int shape_id, const Edge &rEdge, const RayAD &edgeRay, RndSampler *sampler, int max_bounces,
                                           EdgeEvaluationRecord &eRec, std::pair<int, Spectrum>* record, std::tuple<int, Spectrum, Float, int>* record_trans) const
{
#ifndef USE_BOUNDARY_NEE
    std::cerr << "Without next-event estimation (NEE), evalEdgeDirect() should not be used." << std::endl;
    assert(false);
#endif
    Intersection &its1 = eRec.its1, &its2 = eRec.its2;
    int &idx_pixel = eRec.idx_pixel; std::pair<int,int> ret; 

    eRec.value0.zero();
    eRec.value1.zero();
    idx_pixel = -1;

    const Shape &shape = *scene.shape_list[shape_id];
    Ray _edgeRay = edgeRay.toRay();
    const Vector &d1 = _edgeRay.dir;
    if ( scene.rayIntersect(_edgeRay, true, its2) && scene.rayIntersect(_edgeRay.flipped(), true, its1) ) {
        const Vector2i ind0(shape_id, rEdge.f0), ind1(shape_id, rEdge.f1);
        if ( its1.indices != ind0 && its1.indices != ind1 && its2.indices != ind0 && its2.indices != ind1 && its2.isEmitter() ) {
            const Float gn1d1 = its1.geoFrame.n.dot(d1), sn1d1 = its1.shFrame.n.dot(d1),
                        gn2d1 = its2.geoFrame.n.dot(-d1), sn2d1 = its2.shFrame.n.dot(-d1);
            assert(std::abs(its1.wi.z() - sn1d1) < Epsilon && std::abs(its2.wi.z() - sn2d1) < Epsilon);

            bool valid1 = (its1.ptr_bsdf->isTransmissive() && math::signum(gn1d1)*math::signum(sn1d1) > 0.5f) || (!its1.ptr_bsdf->isTransmissive() && gn1d1 > Epsilon && sn1d1 > Epsilon),
                 valid2 = (its2.ptr_bsdf->isTransmissive() && math::signum(gn2d1)*math::signum(sn2d1) > 0.5f) || (!its2.ptr_bsdf->isTransmissive() && gn2d1 > Epsilon && sn2d1 > Epsilon);
            if ( valid1 && valid2 ) {
                // const Shape &shape1 = *scene.shape_list[its1.indices[0]]; const Vector3i &f1 = shape1.getIndices(its1.indices[1]);
                const Shape &shape2 = *scene.shape_list[its2.indices[0]]; const Vector3i &f2 = shape2.getIndices(its2.indices[1]);
                const VectorAD &v0 = shape2.getVertexAD(f2[0]), &v1 = shape2.getVertexAD(f2[1]), &v2 = shape2.getVertexAD(f2[2]);
                Float baseValue = 0.0f;
                Vector n;
                {
                    Float dist = (its2.p - its1.p).norm();
                    Float cos2 = std::abs(gn2d1);
                    Vector e = (shape.getVertex(rEdge.v0) - shape.getVertex(rEdge.v1)).normalized().cross(d1);
                    Float sinphi = e.norm();
                    Vector proj = e.cross(its2.geoFrame.n).normalized();
                    Float sinphi2 = d1.cross(proj).norm();
                    n = its2.geoFrame.n.cross(proj).normalized();

                    Float deltaV;
                    Vector e1 = shape.getVertex(rEdge.v2) - shape.getVertex(rEdge.v0);
                    deltaV = math::signum(e.dot(e1))*math::signum(e.dot(n));

                    if ( sinphi > Epsilon && sinphi2 > Epsilon )
                        baseValue = deltaV*(its1.t/dist)*(sinphi/sinphi2)*cos2;
                }

                if ( std::abs(baseValue) > Epsilon ) {
                    VectorAD u2;

                    /// Direct: camera - its1 - its2 (emitter)
                    {
                        Vector2 pix_uv;
                        Vector d0;
                        Float sensor_val = scene.sampleAttenuatedSensorDirect(its1, sampler, 0, pix_uv, d0);
                        if ( sensor_val > Epsilon ) {
                            RayAD cameraRay = scene.camera.samplePrimaryRayAD(pix_uv[0], pix_uv[1]);
                            IntersectionAD its;
                            if ( scene.rayIntersectAD(cameraRay, false, its) && (its.p.val - its1.p).norm() < ShadowEpsilon ) {
                                bool valid = rayIntersectTriangleAD(v0, v1, v2, RayAD(its.p, edgeRay.org - its.p), u2);
                                if (valid) {
                                    Vector d0_local = its1.toLocal(d0);
                                    Spectrum value0 = its1.evalBSDF(d0_local, EBSDFMode::EImportanceWithCorrection) * sensor_val * baseValue * its2.Le(-d1);
                                    idx_pixel = scene.camera.getPixelIndex(pix_uv);
                                    for ( int j = 0; j < nder; ++j )
                                        eRec.value0.grad(j) = value0*n.dot(u2.grad(j));
                                }
                            }
                        }
                    }

                    /// Indirect: camera - p+ - its1 - its2 (emitter)
                    if ( max_bounces > 1 ) {
                        VectorAD x1, n1;
                        FloatAD J1;
                        scene.getPoint(its1, x1, n1, J1);
                        bool valid = rayIntersectTriangleAD(v0, v1, v2, RayAD(x1, edgeRay.org - x1), u2);
                        if (valid) {
                            for ( int j = 0; j < nder; ++j )
                                eRec.value1.grad(j) = baseValue*n.dot(u2.grad(j));
                            ret = weightedImportance(scene, sampler, its1, max_bounces - 1,
                                                                               nullptr, record, record_trans); 
                        }
                    }
                }
            }
        }
        // Check invalid paths 
        for ( int j = 0; j < nder; ++j ) {
            if ( !std::isfinite(eRec.value0.grad(j)[0]) || !std::isfinite(eRec.value0.grad(j)[1]) || !std::isfinite(eRec.value0.grad(j)[2]) ) {
                omp_set_lock(&messageLock);
                std::cerr << std::fixed << std::setprecision(2)
                          << "\n[WARN] Invalid gradient: [" << eRec.value0.grad(j).transpose() << "]" << std::endl;
                omp_unset_lock(&messageLock);
                eRec.value0.grad(j).setZero();
            }

            if ( !std::isfinite(eRec.value1.grad(j)) ) {
                omp_set_lock(&messageLock);
                std::cerr << std::fixed << std::setprecision(2)
                          << "\n[WARN] Invalid gradient: [" << eRec.value1.grad(j) << "]" << std::endl;
                omp_unset_lock(&messageLock);
                eRec.value1.grad(j) = 0.0f;
            }
        }
    }
    return ret;
}

void IntegratorAD_PathSpace::evalEdge(const Scene &scene, int shape_id, const Edge &rEdge, const RayAD &edgeRay, RndSampler *sampler, EdgeEvaluationRecord &eRec) const {
    Intersection &its1 = eRec.its1, &its2 = eRec.its2;
    int &idx_pixel = eRec.idx_pixel;

    eRec.value0.zero();
    eRec.value1.zero();
    idx_pixel = -1;

    const Shape &shape = *scene.shape_list[shape_id];

    Ray _edgeRay = edgeRay.toRay();
    const Vector &d1 = _edgeRay.dir;
    if ( scene.rayIntersect(_edgeRay, true, its2) && scene.rayIntersect(_edgeRay.flipped(), true, its1) ) {
        const Vector2i ind0(shape_id, rEdge.f0), ind1(shape_id, rEdge.f1);
        if ( its1.indices != ind0 && its1.indices != ind1 && its2.indices != ind0 && its2.indices != ind1 ) {
            const Float gn1d1 = its1.geoFrame.n.dot(d1), sn1d1 = its1.shFrame.n.dot(d1),
                        gn2d1 = its2.geoFrame.n.dot(-d1), sn2d1 = its2.shFrame.n.dot(-d1);
            assert(std::abs(its1.wi.z() - sn1d1) < Epsilon && std::abs(its2.wi.z() - sn2d1) < Epsilon);

            bool valid1 = (its1.ptr_bsdf->isTransmissive() && math::signum(gn1d1)*math::signum(sn1d1) > 0.5f) || (!its1.ptr_bsdf->isTransmissive() && gn1d1 > Epsilon && sn1d1 > Epsilon),
                 valid2 = (its2.ptr_bsdf->isTransmissive() && math::signum(gn2d1)*math::signum(sn2d1) > 0.5f) || (!its2.ptr_bsdf->isTransmissive() && gn2d1 > Epsilon && sn2d1 > Epsilon);
            if ( valid1 && valid2 ) {
                // const Shape &shape1 = *scene.shape_list[its1.indices[0]]; const Vector3i &f1 = shape1.getIndices(its1.indices[1]);
                const Shape &shape2 = *scene.shape_list[its2.indices[0]]; const Vector3i &f2 = shape2.getIndices(its2.indices[1]);
                const VectorAD &v0 = shape2.getVertexAD(f2[0]), &v1 = shape2.getVertexAD(f2[1]), &v2 = shape2.getVertexAD(f2[2]);

                Float baseValue = 0.0f;
                Vector n;
                {
                    Float dist = (its2.p - its1.p).norm();
                    Float cos2 = std::abs(gn2d1);
                    Vector e = (shape.getVertex(rEdge.v0) - shape.getVertex(rEdge.v1)).normalized().cross(d1);
                    Float sinphi = e.norm();
                    Vector proj = e.cross(its2.geoFrame.n).normalized();
                    Float sinphi2 = d1.cross(proj).norm();
                    n = its2.geoFrame.n.cross(proj).normalized();

                    Float deltaV;
                    Vector e1 = shape.getVertex(rEdge.v2) - shape.getVertex(rEdge.v0);
                    deltaV = math::signum(e.dot(e1))*math::signum(e.dot(n));

                    if ( sinphi > Epsilon && sinphi2 > Epsilon )
                        baseValue = deltaV*(its1.t/dist)*(sinphi/sinphi2)*cos2;
                }

                if ( std::abs(baseValue) > Epsilon ) {
                    VectorAD u2;

                    // Direct
                    {
                        Vector2 pix_uv;
                        Vector d0;
                        Float sensor_val = scene.sampleAttenuatedSensorDirect(its1, sampler, 0, pix_uv, d0);
                        if ( sensor_val > Epsilon ) {
                            RayAD cameraRay = scene.camera.samplePrimaryRayAD(pix_uv[0], pix_uv[1]);
                            IntersectionAD its;
                            if ( scene.rayIntersectAD(cameraRay, false, its) && (its.p.val - its1.p).norm() < ShadowEpsilon ) {
                                bool valid = rayIntersectTriangleAD(v0, v1, v2, RayAD(its.p, edgeRay.org - its.p), u2);
                                if (valid) {
                                    Vector d0_local = its1.toLocal(d0);
                                    // Float correction = std::abs((its1.wi.z()*d0.dot(its1.geoFrame.n))/(d0_local.z()*d1.dot(its1.geoFrame.n)));
                                    Spectrum value0 = its1.evalBSDF(d0_local, EBSDFMode::EImportanceWithCorrection)*sensor_val*baseValue;
                                    idx_pixel = scene.camera.getPixelIndex(pix_uv);
                                    for ( int j = 0; j < nder; ++j )
                                        eRec.value0.grad(j) = value0*n.dot(u2.grad(j));
                                }
                            }
                        }
                    }

                    // Indirect
                    {
                        VectorAD x1, n1;
                        FloatAD J1;
                        scene.getPoint(its1, x1, n1, J1);
                        bool valid = rayIntersectTriangleAD(v0, v1, v2, RayAD(x1, edgeRay.org - x1), u2);
                        if (valid) {
                            for ( int j = 0; j < nder; ++j ) {
                                eRec.value1.grad(j) = baseValue*n.dot(u2.grad(j));
                            }
                        }
                    }
                }
            }
        }

        for ( int j = 0; j < nder; ++j ) {
            if ( !std::isfinite(eRec.value0.grad(j)[0]) || !std::isfinite(eRec.value0.grad(j)[1]) || !std::isfinite(eRec.value0.grad(j)[2]) ) {
                omp_set_lock(&messageLock);
                std::cerr << std::fixed << std::setprecision(2)
                          << "\n[WARN] Invalid gradient: [" << eRec.value0.grad(j).transpose() << "]" << std::endl;
                omp_unset_lock(&messageLock);
                eRec.value0.grad(j).setZero();
            }

            if ( !std::isfinite(eRec.value1.grad(j)) ) {
                omp_set_lock(&messageLock);
                std::cerr << std::fixed << std::setprecision(2)
                          << "\n[WARN] Invalid gradient: [" << eRec.value1.grad(j) << "]" << std::endl;
                omp_unset_lock(&messageLock);
                eRec.value1.grad(j) = 0.0f;
            }
        }
    }
}

void IntegratorAD_PathSpace::preprocessDirect(const Scene &scene, const std::vector<int> &params, int max_bounces, ptr<float> data, bool quiet) const {
#ifndef USE_BOUNDARY_NEE
    std::cerr << "Without next-event estimation (NEE), preprocessDirect() should not be used." << std::endl;
    assert(false);
#endif
    const int nworker = omp_get_num_procs();
    std::vector<RndSampler> samplers;
    for ( int i = 0; i < nworker; ++i ) samplers.push_back(RndSampler(13, i));
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
    for ( int omp_i = 0; omp_i < params[0]*params[1]; ++omp_i ) {
        std::pair<int, Spectrum> importance[BDPT_MAX_PATH_LENGTH + 1];
        EdgeEvaluationRecord eRec;
        const int tid = omp_get_thread_num();
        RndSampler &sampler = samplers[tid];

        const int i = omp_i/params[1], j = omp_i % params[1];
        for ( int k = 0; k < params[2]; ++k ) {
            Float res = 0.0f;
            for ( int t = 0; t < params[3]; ++t ) {
                Vector rnd = sampler.next3D();
                rnd[0] = (rnd[0] + i)/static_cast<Float>(params[0]);
                rnd[1] = (rnd[1] + j)/static_cast<Float>(params[1]);
                rnd[2] = (rnd[2] + k)/static_cast<Float>(params[2]);

                int shape_id;
                RayAD edgeRay;
                Float edgePdf, value = 0.0f;
                const Edge &rEdge = scene.sampleEdgeRayDirect(rnd, shape_id, edgeRay, edgePdf);
                if ( shape_id >= 0 ) {
                    int num_indirect_path = std::get<0>(evalEdgeDirect(scene, shape_id, rEdge, edgeRay, &sampler, max_bounces, eRec, importance));
                    if ( eRec.idx_pixel >= 0) {
                        Float val = eRec.value0.der.abs().maxCoeff()/edgePdf;
                        if ( std::isfinite(val) ) value += val;
                    }

                    if ( num_indirect_path > 0) {
                        for (int m = 0; m < num_indirect_path; m++) {
                            assert(eRec.its2.isEmitter());
                            Float val = (eRec.value1.der.abs().maxCoeff() * eRec.its2.Le(-edgeRay.dir.val) * importance[m].second).maxCoeff()/edgePdf;
                            if ( std::isfinite(val) ) value += val;
                        }
                    }
                }
                res += value;
            }
            Float avg = res/static_cast<Float>(params[3]);
            data[static_cast<long long>(omp_i)*params[2] + k] = static_cast<float>(avg);
        }

        if ( !quiet ) {
            omp_set_lock(&messageLock);
            progressIndicator(static_cast<Float>(omp_i)/(params[0]*params[1]));
            omp_unset_lock(&messageLock);
        }
    }
}

void IntegratorAD_PathSpace::buildPhotonMap(const Scene &scene, const GuidingOptions& opts, int max_bounces,
                                            std::vector<MapNode> &rad_nodes, std::vector<MapNode> &imp_nodes) const
{
    const int nworker = omp_get_num_procs();
    std::vector<RndSampler> samplers;
    for ( int i = 0; i < nworker; ++i ) samplers.push_back(RndSampler(17, i));

    std::vector< std::vector<MapNode> > rad_nodes_per_thread(nworker);
    std::vector< std::vector<MapNode> > imp_nodes_per_thread(nworker);
    for (int i = 0; i < nworker; i++) {
        rad_nodes_per_thread[i].reserve(opts.num_cam_path/nworker * max_bounces);
        imp_nodes_per_thread[i].reserve(opts.num_light_path/nworker * max_bounces);
    }

    const Camera &camera = scene.camera;
    const CropRectangle &rect = camera.rect;
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
    for (size_t omp_i = 0; omp_i < opts.num_cam_path; omp_i++) {
        const int tid = omp_get_thread_num();
        bidir::PathNode *cameraPath = paths[2*tid];
        RndSampler &sampler = samplers[tid];

        Float x = rect.isValid() ? rect.offset_x + sampler.next1D() * rect.crop_width
                                 : sampler.next1D() * camera.width;
        Float y = rect.isValid() ? rect.offset_y + sampler.next1D() * rect.crop_height
                                 : sampler.next1D() * camera.height;
        Ray cameraRay = camera.samplePrimaryRay(x, y);
        int cameraPathLen;
        if ( scene.rayIntersect(cameraRay, false, cameraPath[0].its) ) {
            cameraPath[0].throughput = Spectrum(1.0f);
            cameraPathLen = 1;
            if ( max_bounces > 0 )
                cameraPathLen = bidir::buildPath(scene, &sampler, max_bounces, false, cameraPath);
        } else
            cameraPathLen = 0;

        for (int i = 0; i < cameraPathLen; i++) {
            assert(i <= max_bounces);
            rad_nodes_per_thread[tid].push_back( MapNode{cameraPath[i].its, cameraPath[i].throughput, i} );
        }
    }
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
    for (size_t omp_i = 0; omp_i < opts.num_light_path; omp_i++) {
        const int tid = omp_get_thread_num();
        bidir::PathNode *lightPath = paths[2*tid+1];
        RndSampler &sampler = samplers[tid];
        lightPath[0].throughput = scene.sampleEmitterPosition(sampler.next2D(), lightPath[0].its, &lightPath[0].pdf0);
        int lightPathLen = 1;
        if ( max_bounces > 0 ) {
            Vector wo;
            Float tmp = lightPath[0].its.ptr_emitter->sampleDirection(sampler.next2D(), wo, &lightPath[1].pdf0);
            wo = lightPath[0].its.geoFrame.toWorld(wo);
            if ( scene.rayIntersect(Ray(lightPath[0].its.p, wo), true, lightPath[1].its) ) {
                lightPath[0].wo = wo;
                lightPath[1].throughput = lightPath[0].throughput*tmp;
                if ( max_bounces > 1 )
                    lightPathLen = bidir::buildPath(scene, &sampler, max_bounces, true, &lightPath[1]) + 1;
                else
                    lightPathLen = 2;
            }
        }
#ifdef USE_BOUNDARY_NEE
        int lightPathStart = 1;
#else
        int lightPathStart = 0;
#endif
        for (int i = lightPathStart; i < lightPathLen; i++) {
            assert(i <= max_bounces);
            imp_nodes_per_thread[tid].push_back( MapNode{lightPath[i].its, lightPath[i].throughput, i} );
        }
    }

    size_t sz_rad = 0, sz_imp = 0;
    for (int i = 0; i < nworker; i++) {
        sz_rad += rad_nodes_per_thread[i].size();
        sz_imp += imp_nodes_per_thread[i].size();
    }
    rad_nodes.reserve(sz_rad);
    imp_nodes.reserve(sz_imp);
    for (int i = 0; i < nworker; i++) {
        rad_nodes.insert(rad_nodes.end(), rad_nodes_per_thread[i].begin(), rad_nodes_per_thread[i].end());
        imp_nodes.insert(imp_nodes.end(), imp_nodes_per_thread[i].begin(), imp_nodes_per_thread[i].end());
    }
}

int IntegratorAD_PathSpace::queryPhotonMap(const KDtree<Float> &indices, const GuidingOptions& opts, const Float* query_point,
                                           size_t* matched_indices, Float& matched_dist_sqr, bool type) const {
    assert( opts.type == 1 || opts.type == 2 );
    int num_matched = 0;
    Float dist_sqr[NUM_NEAREST_NEIGHBOR];
    if (opts.type == 1) {
        num_matched = indices.knnSearch(query_point, NUM_NEAREST_NEIGHBOR, matched_indices, dist_sqr);
        assert(num_matched > 0);
        matched_dist_sqr = dist_sqr[num_matched - 1];
        if (matched_dist_sqr <  opts.search_radius) {
#ifdef VERBOSE
            omp_set_lock(&messageLock);
            if (type)
                std::cout << "[INFO] RadianceMap: " << "r = " << matched_dist_sqr << " < " << opts.search_radius << std::endl;
            else
                std::cout << "[INFO] ImportanceMap: " << "r = " << matched_dist_sqr << " < " << opts.search_radius << std::endl;
            omp_unset_lock(&messageLock);
#endif
            std::vector<std::pair<size_t, Float>> rsearch_result;
            nanoflann::SearchParams search_params;
            num_matched = indices.radiusSearch(query_point, NUM_NEAREST_NEIGHBOR, rsearch_result, search_params);
            matched_dist_sqr = opts.search_radius;
            if (num_matched > MAX_NEAREST_NEIGHBOR) {
                omp_set_lock(&messageLock);
                std::cout << "[INFO] #matched = " << num_matched << " > " << MAX_NEAREST_NEIGHBOR << (type ? "(RadianceMap)" : "(ImportanceMap)") << std::endl;
                omp_unset_lock(&messageLock);
                num_matched = MAX_NEAREST_NEIGHBOR;
            }

            for (int i = 0; i < num_matched; i++)
                matched_indices[i] = rsearch_result[i].first;
        }
    } else {
        nanoflann::SearchParams search_params;
        std::vector<std::pair<size_t, Float>> rsearch_result;
        num_matched = indices.radiusSearch(query_point, opts.search_radius, rsearch_result, search_params);
        if (num_matched == 0) {
#ifdef VERBOSE
            omp_set_lock(&messageLock);
            if (type)
                std::cout << "[INFO] RadianceMap: No photon is found within dist(d2) " << opts.search_radius << std::endl;
            else
                std::cout << "[INFO] ImportanceMap: No photon is found within dist(d2) " << opts.search_radius << std::endl;
            omp_unset_lock(&messageLock);
#endif
            num_matched = indices.knnSearch(query_point, NUM_NEAREST_NEIGHBOR, matched_indices, dist_sqr);
            assert(num_matched > 0);
            matched_dist_sqr = dist_sqr[num_matched - 1];
        } else {
            if (num_matched > MAX_NEAREST_NEIGHBOR) {
                omp_set_lock(&messageLock);
                std::cout << "[INFO] #matched = " << num_matched << " > " << MAX_NEAREST_NEIGHBOR << std::endl;
                omp_unset_lock(&messageLock);
                num_matched = MAX_NEAREST_NEIGHBOR;
            }

            matched_dist_sqr = opts.search_radius;
            for (int i = 0; i < num_matched; i++)
                matched_indices[i] = rsearch_result[i].first;
        }
    }
    return num_matched;
}

void IntegratorAD_PathSpace::preprocessIndirect(const Scene &scene, const GuidingOptions& opts, int max_bounces,
                                                const std::vector<MapNode> &rad_nodes, const KDtree<Float> &rad_indices,
                                                const std::vector<MapNode> &imp_nodes, const KDtree<Float> &imp_indices,
                                                ptr<float> data, bool quiet) const
{
    const int nworker = omp_get_num_procs();
    std::vector<RndSampler> samplers;
    for ( int i = 0; i < nworker; ++i ) samplers.push_back(RndSampler(13, i));
    const std::vector<int> &params = opts.params;

#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
    for ( int omp_i = 0; omp_i < params[0]*params[1]; ++omp_i ) {
        size_t matched_indices[MAX_NEAREST_NEIGHBOR];
        const int tid = omp_get_thread_num();
        RndSampler &sampler = samplers[tid];

        const int i = omp_i/params[1], j = omp_i % params[1];
        for ( int k = 0; k < params[2]; ++k ) {
            Float res = 0.0f;
            for ( int t = 0; t < params[3]; ++t ) {
                Vector rnd = sampler.next3D();
                rnd[0] = (rnd[0] + i)/static_cast<Float>(params[0]);
                rnd[1] = (rnd[1] + j)/static_cast<Float>(params[1]);
                rnd[2] = (rnd[2] + k)/static_cast<Float>(params[2]);

                int shape_id;
                RayAD edgeRay;
                Float edgePdf, value1 = 0.0f;
                const Edge &rEdge = scene.sampleEdgeRay(rnd, shape_id, edgeRay, edgePdf);
                if ( shape_id >= 0 ) {
                    EdgeEvaluationRecord eRec;
                    evalEdge(scene, shape_id, rEdge, edgeRay, &sampler, eRec);
                    value1 = eRec.value1.der.abs().maxCoeff()/edgePdf;
                    if (value1 > 0.0f) {
                        Float pt_rad[3] = {eRec.its1.p[0], eRec.its1.p[1], eRec.its1.p[2]};
                        Float pt_imp[3] = {eRec.its2.p[0], eRec.its2.p[1], eRec.its2.p[2]};
                        Float matched_r2_rad, matched_r2_imp;
                        int num_matched = queryPhotonMap(rad_indices, opts, pt_rad, matched_indices, matched_r2_rad, true);
                        std::vector<Spectrum> radiance(max_bounces, Spectrum::Zero());
                        for (int m = 0; m < num_matched; m++) {
                            const MapNode& node = rad_nodes[matched_indices[m]];
                            assert(node.depth < max_bounces);
                            Float bsdf_val = node.its.evalBSDF(node.its.toLocal(edgeRay.dir.val)).maxCoeff();
                            if (bsdf_val < BSDF_CLAMPING_MIN) bsdf_val = BSDF_CLAMPING_MIN;
                            if (bsdf_val > BSDF_CLAMPING_MAX) bsdf_val = BSDF_CLAMPING_MAX;
                            radiance[node.depth] += node.val * bsdf_val;
                        }
                        num_matched = queryPhotonMap(imp_indices, opts, pt_imp, matched_indices, matched_r2_imp, false);
                        std::vector<Spectrum> importance(max_bounces, Spectrum::Zero());
                        for (int m = 0; m < num_matched; m++) {
                            const MapNode& node = imp_nodes[matched_indices[m]];
                            assert(node.depth < max_bounces);
#ifdef USE_BOUNDARY_NEE
                            assert(node.depth > 0);
#endif
                            // Float bsdf_val = node.its.evalBSDF(node.its.toLocal(-edgeRay.dir.val), EBSDFMode::EImportanceWithCorrection).maxCoeff();
                            // if (bsdf_val < BSDF_CLAMPING_MIN) bsdf_val = BSDF_CLAMPING_MIN;
                            // if (bsdf_val > BSDF_CLAMPING_MAX) bsdf_val = BSDF_CLAMPING_MAX;
                            if (node.depth == 0) {
                                importance[node.depth] += node.val.maxCoeff() * node.its.ptr_emitter->evalDirection(node.its.geoFrame.n, -edgeRay.dir.val);
                            }
                            else {
                                Float bsdf_val = node.its.evalBSDF(node.its.toLocal(-edgeRay.dir.val), EBSDFMode::EImportanceWithCorrection).maxCoeff();
                                if (bsdf_val < BSDF_CLAMPING_MIN) bsdf_val = BSDF_CLAMPING_MIN;
                                if (bsdf_val > BSDF_CLAMPING_MAX) bsdf_val = BSDF_CLAMPING_MAX;
                                importance[node.depth] += node.val * bsdf_val;
                            }
                        }

                        Spectrum value2 = Spectrum::Zero();
#ifdef USE_BOUNDARY_NEE
                        int impStart = 1;
#else
                        int impStart = 0;
#endif
                        for (int m = 0; m < max_bounces; m++) {
                            for (int n = impStart; n < max_bounces-m; n++)
                                value2 += radiance[m] * importance[n];
                        }

                        value1 *= value2.maxCoeff() / (matched_r2_rad * matched_r2_imp);
                        assert(std::isfinite(value1));
                    }
                }
                res += value1;
            }

            Float avg = res/static_cast<Float>(params[3]);
            data[static_cast<long long>(omp_i)*params[2] + k] = static_cast<float>(avg);
        }

        if ( !quiet ) {
            omp_set_lock(&messageLock);
            progressIndicator(static_cast<Float>(omp_i)/(params[0]*params[1]));
            omp_unset_lock(&messageLock);
        }
    }
}

void IntegratorAD_PathSpace::preprocess(const Scene &scene, int max_bounces, const GuidingOptions& opts, ptr<float> data) const {
    using namespace std::chrono;
    const std::vector<int> &params = opts.params;

    assert(opts.type < 4);
    std::string guiding_type[4] = {"Direct Guiding", "Indirect Guiding (KNN)",
                                   "Indirect Guiding (Radius Search)", "Indirect Guiding (Old)"};

    assert(params.size() == 4);
    if ( !opts.quiet )
        std::cout << "[INFO] Preprocessing for " << guiding_type[opts.type]
                  << " at (" << params[0] << " x " << params[1] << " x " << params[2] << ") ... " << std::endl;

    auto _start = high_resolution_clock::now();
    if ( opts.type == 0) {
        preprocessDirect(scene, params, max_bounces, data, opts.quiet);
    } else {
#ifdef USE_BOUNDARY_NEE
        if (max_bounces < 2) {
            if ( !opts.quiet )
                std::cout << "[INFO] max_bounces < 2, no indirect component. Guiding cancelled." << std::endl;
            return;
        }
#endif
        if (opts.type == 3) {
            // Old Indirect guiding (unbiased)
            const int nworker = omp_get_num_procs();
            std::vector<RndSampler> samplers;
            for ( int i = 0; i < nworker; ++i ) samplers.push_back(RndSampler(13, i));
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
            for ( int omp_i = 0; omp_i < params[0]*params[1]; ++omp_i ) {
                const int tid = omp_get_thread_num();
                RndSampler &sampler = samplers[tid];
                const int i = omp_i/params[1], j = omp_i % params[1];
                for ( int k = 0; k < params[2]; ++k ) {
                    Float res = 0.0f;
                    for ( int t = 0; t < params[3]; ++t ) {
                        Vector rnd = sampler.next3D();
                        rnd[0] = (rnd[0] + i)/static_cast<Float>(params[0]);
                        rnd[1] = (rnd[1] + j)/static_cast<Float>(params[1]);
                        rnd[2] = (rnd[2] + k)/static_cast<Float>(params[2]);

                        int shape_id;
                        RayAD edgeRay;
                        Float edgePdf, value = 0.0f;
                        const Edge &rEdge = scene.sampleEdgeRay(rnd, shape_id, edgeRay, edgePdf);
                        if ( shape_id >= 0 ) {
                            EdgeEvaluationRecord eRec;
                            evalEdge(scene, shape_id, rEdge, edgeRay, &sampler, eRec);
                            value = eRec.value1.der.abs().maxCoeff()/edgePdf;
                        }
                        res += value;
                    }

                    Float avg = res/static_cast<Float>(params[3]);
                    data[static_cast<long long>(omp_i)*params[2] + k] = static_cast<float>(avg);
                }

                if ( !opts.quiet ) {
                    omp_set_lock(&messageLock);
                    progressIndicator(static_cast<Float>(omp_i)/(params[0]*params[1]));
                    omp_unset_lock(&messageLock);
                }
            }
        } else {
            if ( !opts.quiet )
                std::cout << "[INFO] #camPath = " << opts.num_cam_path << ", #lightPath = " << opts.num_light_path << std::endl;
            std::vector<MapNode> rad_nodes, imp_nodes;
            buildPhotonMap(scene, opts, max_bounces-1, rad_nodes, imp_nodes);
            if ( !opts.quiet )
                std::cout << "[INFO] #rad_nodes = " << rad_nodes.size() << ", #imp_nodes = " << imp_nodes.size() << std::endl;

            // Build up the point KDtree for query
            PointCloud<Float> rad_cloud;
            rad_cloud.pts.resize(rad_nodes.size());
            for (size_t i = 0; i < rad_nodes.size(); i++) {
                rad_cloud.pts[i].x = rad_nodes[i].its.p[0];
                rad_cloud.pts[i].y = rad_nodes[i].its.p[1];
                rad_cloud.pts[i].z = rad_nodes[i].its.p[2];
            }
            KDtree<Float> rad_indices(3, rad_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            rad_indices.buildIndex();

            PointCloud<Float> imp_cloud;
            imp_cloud.pts.resize(imp_nodes.size());
            for (size_t i = 0; i < imp_nodes.size(); i++) {
                imp_cloud.pts[i].x = imp_nodes[i].its.p[0];
                imp_cloud.pts[i].y = imp_nodes[i].its.p[1];
                imp_cloud.pts[i].z = imp_nodes[i].its.p[2];
            }
            KDtree<Float> imp_indices(3, imp_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
            imp_indices.buildIndex();
            // Indirect Guiding
            preprocessIndirect(scene, opts, max_bounces, rad_nodes, rad_indices, imp_nodes, imp_indices, data, opts.quiet);
        }
    }
    if ( !opts.quiet )
        std::cout << "\nDone in " << duration_cast<seconds>(high_resolution_clock::now() - _start).count() << " seconds." << std::endl;
}

void IntegratorAD_PathSpace::renderEdgesDirect(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient) const {
#ifndef USE_BOUNDARY_NEE
    std::cerr << "Without next-event estimation (NEE), renderEdgesDirect() should not be used." << std::endl;
    assert(false);
#endif
    const CameraTransient &camera = scene.camera;
    int num_pixels = camera.getNumPixels();
    const int nworker = omp_get_num_procs();
    int num_duple_img = options.mode == MEMORY_LOCK ? 1 : nworker; 
    std::vector<std::vector<Spectrum> > image_per_thread(num_duple_img); 
    for (int i = 0; i < num_duple_img; i++) image_per_thread[i].resize(nder*num_pixels, Spectrum(0.0f)); 

    std::vector<std::vector<Spectrum> > transient_per_thread(num_duple_img); 
    const int duration = camera.duration;
    if ( !rendered_transient.is_null() ) {
        assert(camera.valid_transient());
        for (int i = 0; i < num_duple_img; i++) 
            transient_per_thread[i].resize(nder * num_pixels * duration, Spectrum(0.0f));
    }

    constexpr int num_samples_per_block = 128;
    long long num_samples = static_cast<long long>(options.num_samples_secondary_edge_direct)*num_pixels;
    const long long num_block = static_cast<long long>(std::ceil(static_cast<Float>(num_samples)/num_samples_per_block));
    num_samples = num_block*num_samples_per_block;
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
    for ( long long index_block = 0; index_block < num_block; ++index_block ) {
        for ( int omp_i = 0; omp_i < num_samples_per_block; ++omp_i ) {
            const int tid = omp_get_thread_num();
            RndSampler sampler(options.seed, m_taskId[tid] = index_block*num_samples_per_block + omp_i);
            int shape_id;
            RayAD edgeRay;
            Float edgePdf;
            const Edge &rEdge = scene.sampleEdgeRayDirect(sampler.next3D(), shape_id, edgeRay, edgePdf);
            if ( shape_id >= 0 ) {
                std::pair<int, Spectrum> importance[BDPT_MAX_PATH_LENGTH + 1];
                EdgeEvaluationRecord eRec;
                std::tuple<int, Spectrum, Float, int> importance_trans[(BDPT_MAX_PATH_LENGTH-3)*(BDPT_MAX_PATH_LENGTH-2)/2 - 1];
                int num_indirect_path, num_indpath_trans;
                std::tie(num_indirect_path, num_indpath_trans)
                    = evalEdgeDirect(scene, shape_id, rEdge, edgeRay, &sampler, options.max_bounces, eRec, importance, importance_trans);
                // importance_pathDist will be initialized and store path distances from camera to its0 for each path.

                /// one-bounce (See evalEdgeDirect)
                if ( eRec.idx_pixel >= 0 ) {
                    int idx_duple = set_pixel_lock(eRec.idx_pixel, tid); 
                    for ( int j = 0; j < nder; ++j )
                        image_per_thread[idx_duple][j*num_pixels + eRec.idx_pixel] += eRec.value0.grad(j)/edgePdf; 
                    if ( !rendered_transient.is_null() ) {
                        Float pathTime = (eRec.its1.getOpdFrom(camera.cpos.val) + eRec.its1.getOpdFrom(eRec.its2.p))*INV_C;
                        int i_bin_start, i_bin_end;
                        camera.bin_range(pathTime, i_bin_start, i_bin_end);
                        camera.clip_bin_index(i_bin_start, i_bin_end);
                        for ( int j = 0; j < nder; j++)
                            for (int i_bin = i_bin_start; i_bin <= i_bin_end; i_bin++)
                                transient_per_thread[idx_duple][(j*num_pixels + eRec.idx_pixel)*duration + i_bin] += eRec.value0.grad(j)/edgePdf*camera.eval_tsens(pathTime, i_bin);
                    }
                    unset_pixel_lock(eRec.idx_pixel); 
                }

                /// multi-bounce (See evalEdgeDirect)
                if ( num_indirect_path > 0 ) {
                    Spectrum light_val = eRec.its2.Le(-edgeRay.dir.val);
                    for (int k = 0; k < num_indirect_path; k++) {
                        int idx_pixel = importance[k].first; 
                        int idx_duple = set_pixel_lock(idx_pixel, tid); 
                        for ( int j = 0; j < nder; ++j )
                            image_per_thread[idx_duple][j*num_pixels + idx_pixel] += 
                                eRec.value1.grad(j)*importance[k].second*light_val/edgePdf;
                        unset_pixel_lock(idx_pixel); 
                    }
                    if ( !rendered_transient.is_null() ) {
                        for (int k = 0; k < num_indpath_trans; k++){
                            int idx_pixel; Spectrum value; Float pathDist;
                            std::tie(idx_pixel, value, pathDist, std::ignore) = importance_trans[k];
                            int idx_duple = set_pixel_lock(idx_pixel, tid);
                            int i_bin_start, i_bin_end;
                            // importance_pathDist stores path distances from camera to its0 for each path.
                            Float pathTime = (pathDist + eRec.its1.getOpdFrom(eRec.its2.p)) * INV_C;
                            camera.bin_range(pathTime, i_bin_start, i_bin_end);
                            camera.clip_bin_index(i_bin_start, i_bin_end);

                            for ( int j = 0; j < nder; ++j ) {
                                int offset = j * num_pixels + idx_pixel;
                                for (int i_bin = i_bin_start; i_bin <= i_bin_end; i_bin++)
                                    transient_per_thread[idx_duple][offset*duration + i_bin] +=
                                        eRec.value1.grad(j) * value * light_val / edgePdf * camera.eval_tsens(pathTime, i_bin);
                            }
                            unset_pixel_lock(idx_pixel);
                        }
                    }
                }
            }
        }

        if ( !options.quiet ) {
            omp_set_lock(&messageLock);
            progressIndicator(Float(index_block + 1)/num_block);
            omp_unset_lock(&messageLock);
        }
    }
    if ( !options.quiet ) std::cout << std::endl;

    for ( int i = 0; i < num_duple_img; ++i ) 
        for ( int j = 0; j < nder; ++j )
            for ( int idx_pixel = 0; idx_pixel < num_pixels; ++idx_pixel ) {
                int offset1 = ((j + 1)*num_pixels + idx_pixel)*3,
                    offset2 = j*num_pixels + idx_pixel;
                rendered_image[offset1    ] += image_per_thread[i][offset2][0]/static_cast<Float>(num_samples);
                rendered_image[offset1 + 1] += image_per_thread[i][offset2][1]/static_cast<Float>(num_samples);
                rendered_image[offset1 + 2] += image_per_thread[i][offset2][2]/static_cast<Float>(num_samples);
            }
    if ( !rendered_transient.is_null() ) {
        assert(camera.valid_transient());
        for ( int i = 0; i < num_duple_img; ++i ) 
            for ( int j = 0; j < nder; ++j )
                for ( int idx_pixel = 0; idx_pixel < num_pixels; ++idx_pixel ) {
                    int offset1 = ((j + 1)*num_pixels + idx_pixel)*3,
                        offset2 = j*num_pixels + idx_pixel;
                    for (int i_bin = 0; i_bin < duration; i_bin++) {
                        int offset3 = offset1*duration + i_bin*3,
                            offset4 = offset2*duration + i_bin;
                        rendered_transient[offset3    ] += transient_per_thread[i][offset4][0] / static_cast<Float>(num_samples);
                        rendered_transient[offset3 + 1] += transient_per_thread[i][offset4][1] / static_cast<Float>(num_samples);
                        rendered_transient[offset3 + 2] += transient_per_thread[i][offset4][2] / static_cast<Float>(num_samples);
                    }
                }
    }
}

int IntegratorAD_PathSpace::radiance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces, Spectrum *ret, std::tuple<Spectrum, Float, int> *ret_trans) const {
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// x0=its, x1, ..., x(d-1), xd
    /// |       |        |
    /// e0      e1       e(d-1)
    ///     where d <= max_bounces, ei is emitter for direct component
    ///     (WARN: the ray sampled at xd does not intersect with the scene, then terminate even if d < max_bounces)
    ///
    /// @output:
    /// For 0 <= i <= d:
    ///     ret[i] += radiance(emitted from xi + e(i-1) -> x0=its.wi)   (WARN: without initialization by zero!!)
    ///              (== radiance by paths consisting of i segments and i+1 vertices)
    /// For 0 <= i < len(ret_trans) (<= (mb+1)(mb+2)/2 - 1)
    ///     ret_trans[i] = (value, pathDist, depth) (with initialization)
    /// @return: len(ret_trans)
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Called by IntegratorAD_PathSpace::traceRayFromEdgeSegement, BidirectionalPathTracerAD::pixelColor
    Intersection _its(its);
    if ( _its.isEmitter() ) ret[0] += _its.Le(_its.toWorld(_its.wi));

    Spectrum throughput(1.0f);
    int num_trans = 0;
    Float pathDist = 0.0;
    if ( ret_trans != nullptr ){
        assert(scene.camera.valid_transient());
        ret_trans[num_trans] = {ret[0], pathDist, 0};
        num_trans++;
    }
    for ( int d_emitter = 1; d_emitter <= max_bounces; d_emitter++ ) {
        // 1. Direct illumination
        /// _its == x[d_emitter-1]
        Float pdf_nee;
        Vector wo;
        Vector pos_emitter;
        Spectrum value = scene.sampleEmitterDirect(_its, sampler->next2D(), sampler, wo, pdf_nee, &pos_emitter);
        if ( !value.isZero(Epsilon) ) {
            Spectrum bsdf_val = _its.evalBSDF(wo);
            Float bsdf_pdf = _its.pdfBSDF(wo);
            Float mis_weight = pdf_nee/(pdf_nee + bsdf_pdf);
            Spectrum value_direct = throughput*value*bsdf_val*mis_weight;
            ret[d_emitter] += value_direct;
            if (ret_trans){
                ret_trans[num_trans] = {value_direct, pathDist + (_its.p-pos_emitter).norm(), d_emitter};
                num_trans++;
            }
        }

        // 2. Indirect illumination
        Float bsdf_pdf, bsdf_eta;
        Spectrum bsdf_weight = _its.sampleBSDF(sampler->next3D(), wo, bsdf_pdf, bsdf_eta);
        if ( bsdf_weight.isZero(Epsilon) ) break;
        wo = _its.toWorld(wo);

        Ray ray_emitter(_its.p, wo);
        if ( !scene.rayIntersect(ray_emitter, true, _its) ) break;
        /// Now _its == x[d_emitter]
        throughput *= bsdf_weight;
        pathDist += _its.opd;
        if ( _its.isEmitter() ) {
            Spectrum light_contrib = _its.Le(-ray_emitter.dir);
            if ( !light_contrib.isZero(Epsilon) ) {
                Float dist_sq = (_its.p - ray_emitter.org).squaredNorm();
                Float G = _its.geoFrame.n.dot(-ray_emitter.dir)/dist_sq;
                pdf_nee = scene.pdfEmitterSample(_its)/G;
                Float mis_weight = bsdf_pdf/(pdf_nee + bsdf_pdf);
                Spectrum value_path = throughput*light_contrib*mis_weight;
                ret[d_emitter] += value_path;
                if (ret_trans){
                    ret_trans[num_trans] = {value_path, pathDist, d_emitter};
                    num_trans++;
                }
            }
        }
    }
    return num_trans;
}
std::pair<int,int> IntegratorAD_PathSpace::weightedImportance(const Scene& scene, RndSampler* sampler, const Intersection& its0, int max_depth, const Spectrum *weight,
                                               std::pair<int, Spectrum>* ret, std::tuple<int, Spectrum, Float, int>* ret_trans) const {
    // camera - ... - [1] - [0] - its0 
    // ret_pathDist will be initialized and store path distances from camera to its0 for each path. 
    Intersection its = its0;
    Spectrum throughput(1.0f);
    Vector d0;
    Vector2 pix_uv;
    Ray ray_sensor;
    int num_valid_path = 0;
    Float dist_its0_itsLast = 0.0;
    for (int d_sensor = 1; d_sensor <= max_depth; d_sensor++) {
        // sample a new direction
        Vector wo_local, wo;
        Float bsdf_pdf, bsdf_eta;
        Spectrum bsdf_weight = its.sampleBSDF(sampler->next3D(), wo_local, bsdf_pdf, bsdf_eta, EBSDFMode::EImportanceWithCorrection);
        if (bsdf_weight.isZero())
            break;
        wo = its.toWorld(wo_local);
        Vector wi = its.toWorld(its.wi);
        Float wiDotGeoN = wi.dot(its.geoFrame.n), woDotGeoN = wo.dot(its.geoFrame.n);
        if (wiDotGeoN * its.wi.z() <= 0 || woDotGeoN * wo_local.z() <= 0)
            break;
        throughput *= bsdf_weight;
        ray_sensor = Ray(its.p, wo);
        scene.rayIntersect(ray_sensor, true, its); // update Intersection its
        /// its is fixed
        if ( !its.isValid() )
            break;
        Float sensor_val = scene.sampleAttenuatedSensorDirect(its, sampler, 0, pix_uv, d0);
        if ( ret_trans != nullptr ){
            assert(scene.camera.valid_transient());
            dist_its0_itsLast += its.opd;
        }
        if (sensor_val > Epsilon) {
            Vector wi = -ray_sensor.dir;
            Vector wo = d0, wo_local = its.toLocal(wo);
            Float wiDotGeoN = wi.dot(its.geoFrame.n), woDotGeoN = wo.dot(its.geoFrame.n);
            if (wiDotGeoN * its.wi.z() > 0 && woDotGeoN * wo_local.z() > 0) {
                assert(num_valid_path <= BDPT_MAX_PATH_LENGTH);
                ret[num_valid_path].second = (weight != nullptr) ? weight[max_depth - d_sensor] : Spectrum(1.0f);
                ret[num_valid_path].second *= throughput * its.evalBSDF(wo_local, EBSDFMode::EImportanceWithCorrection) * sensor_val;
                ret[num_valid_path].first = scene.camera.getPixelIndex(pix_uv);
                if ( ret_trans != nullptr ) {
                    Float pathDist = dist_its0_itsLast + its.getOpdFrom(scene.camera.cpos.val);
                    ret_trans[num_valid_path] = {ret[num_valid_path].first, ret[num_valid_path].second, pathDist, d_sensor};
                }
                num_valid_path++;
            }
        }
    }
    return {num_valid_path, num_valid_path};
}

void IntegratorAD_PathSpace::traceRayFromEdgeSegement(const Scene &scene, const EdgeEvaluationRecord& eRec, Float edgePdf, int max_bounces, RndSampler *sampler, std::vector<Spectrum> &image, std::vector<Spectrum>* p_transient) const {
    assert(max_bounces > 0);
    const int num_pixels = image.size()/nder;

    /*** Trace ray towards emitter from its2 ***/
    std::vector<Spectrum> L(max_bounces, Spectrum(0.0f));
    const CameraTransient& camera = scene.camera;
    const int duration = camera.duration;

    auto dist_its12 = eRec.its1.getOpdFrom(eRec.its2.p); 
    int vecsize_trans = (BDPT_MAX_PATH_LENGTH-3)*(BDPT_MAX_PATH_LENGTH-2)/2 - 1;
    int np_emitter;
    std::vector<std::tuple<Spectrum, Float, int>> pathTrans_emitter(vecsize_trans); // path record from emiter to its2
    {
        np_emitter = radiance(scene, sampler, eRec.its2, max_bounces - 1, &L[0], &pathTrans_emitter[0]);
#ifdef USE_BOUNDARY_NEE
        L[0] = Spectrum(0.0f);
        for (int i_path = 0; i_path < np_emitter; i_path++){
            std::tuple<Spectrum, Float, int> &trans_tuple = pathTrans_emitter[i_path];
            if (std::get<2>(trans_tuple) > 0)
                break;
            std::get<0>(trans_tuple) = Spectrum(0.0f);
        }
#endif
        for (int d_emitter = 1; d_emitter < max_bounces; d_emitter++)
            L[d_emitter] += L[d_emitter - 1];
    }

    /*** Trace ray towards sensor from its1 ***/
    // camera - its1 - its2 - p* - emitter 
    if (eRec.idx_pixel >= 0) {
        set_pixel_lock(eRec.idx_pixel); 
        for ( int j = 0; j < nder; ++j ) {
            // Direct connect to sensor
            Spectrum coeff = Spectrum(eRec.value0.grad(j))/edgePdf;
            image[j*num_pixels + eRec.idx_pixel] += L[max_bounces - 1]*coeff;
            if (p_transient != nullptr){
                int offset = (j*num_pixels + eRec.idx_pixel)*duration;
                for (int k = 0; k < np_emitter; k++){
                    Spectrum value_trans; Float pathDist;
                    std::tie(value_trans, pathDist, std::ignore) = pathTrans_emitter[k];
                    Float pathTime = (pathDist + eRec.its1.getOpdFrom(camera.cpos.val) + dist_its12) * INV_C;
                    int i_bin_start, i_bin_end;
                    camera.bin_range(pathTime, i_bin_start, i_bin_end);
                    camera.clip_bin_index(i_bin_start, i_bin_end);
                    Spectrum tmp_throughput = value_trans*coeff;
                    for (int i_bin = i_bin_start; i_bin <= i_bin_end; i_bin++)
                        (*p_transient)[offset + i_bin] += tmp_throughput * camera.eval_tsens(pathTime, i_bin);
                }
            }
        }
        unset_pixel_lock(eRec.idx_pixel);
    }

    // camera - p+ - its1 - its2 - p* - emitter
    if ( max_bounces > 1 ) {
        std::pair<int, Spectrum> pathThroughput[BDPT_MAX_PATH_LENGTH + 1]; 
        std::tuple<int, Spectrum, Float, int> pathTrans[(BDPT_MAX_PATH_LENGTH-3)*(BDPT_MAX_PATH_LENGTH-2)/2 - 1];
        if (p_transient == nullptr) {
            int num_path;
            std::tie(num_path, std::ignore) = weightedImportance(scene, sampler, eRec.its1, max_bounces - 1, &L[0], pathThroughput);

            for (int i = 0; i < num_path; i++) {
                int idx_pixel = pathThroughput[i].first;
                set_pixel_lock(idx_pixel);
                for (int j = 0; j < nder; j++) {
                    image[j * num_pixels + idx_pixel] += 
                            pathThroughput[i].second * eRec.value1.grad(j) / edgePdf;
                }
                unset_pixel_lock(idx_pixel); 
            }
        }
        else{ // p_transient != nullptr
            int num_path, num_path_trans;
            std::tie(num_path, num_path_trans) = weightedImportance(scene, sampler, eRec.its1, max_bounces - 1, nullptr, pathThroughput, pathTrans);
            for (int i = 0; i < num_path_trans; i++) {
                int idx_pixel; Spectrum value_camera; Float pathDist_camera; int depth_camera;
                std::tie(idx_pixel, value_camera, pathDist_camera, depth_camera) = pathTrans[i];
                set_pixel_lock(idx_pixel);
                for (int j = 0; j < np_emitter; j++){
                    Spectrum value_light; Float pathDist_light; int depth_light;
                    std::tie(value_light, pathDist_light, depth_light) = pathTrans_emitter[j];
                    if (depth_camera + depth_light > max_bounces - 1)
                        break;
                    Float pathTime = (pathDist_camera + pathDist_light + dist_its12) * INV_C;
                    int i_bin_start, i_bin_end;
                    camera.bin_range(pathTime, i_bin_start, i_bin_end);
                    camera.clip_bin_index(i_bin_start, i_bin_end);
                    for (int k = 0; k < nder; k++){
                        Spectrum throughput = value_camera*value_light * eRec.value1.grad(k) / edgePdf;
                        image[k * num_pixels + idx_pixel] += throughput;
                        for (int i_bin = i_bin_start; i_bin <= i_bin_end; i_bin++)
                            (*p_transient)[(k*num_pixels + idx_pixel)*duration + i_bin] += throughput * camera.eval_tsens(pathTime, i_bin);
                    }
                }
                unset_pixel_lock(idx_pixel);
            }
        }
    }
    L.clear();
    L.shrink_to_fit();
    pathTrans_emitter.clear();
    pathTrans_emitter.shrink_to_fit();
}

Float iptSum(const std::vector<std::vector<Spectrum>> &image_per_thread){
    Float ret = 0.0;
    for (int i = 0; image_per_thread.size(); i++)
        for (int j = 0; image_per_thread[i].size(); j++)
            ret += abs(image_per_thread[i][j][0]) + abs(image_per_thread[i][j][1]) + abs(image_per_thread[i][j][2]);
    return ret;
}
Float imgSum(const ptr<float> rendered_image, int num){
    float ret = 0.0;
    for ( int i = num/nder; i < num + num/nder; i++)
        ret += abs(rendered_image[i]);
    return ret;
}

void IntegratorAD_PathSpace::renderEdges(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient) const {
#ifdef USE_BOUNDARY_NEE
    if ( options.max_bounces > 1 ) {
#else
    if ( options.max_bounces >= 1 ) {
#endif
        const CameraTransient &camera = scene.camera;
        int num_pixels = camera.getNumPixels();
        const int nworker = omp_get_num_procs();
        int num_duple_img = options.mode == MEMORY_LOCK ? 1 : nworker; 
        std::vector<std::vector<Spectrum> > image_per_thread(num_duple_img); 
        for (int i = 0; i < num_duple_img; i++) image_per_thread[i].resize(nder*num_pixels, Spectrum(0.0f)); 
        std::vector<std::vector<Spectrum> > transient_per_thread(num_duple_img); 
        const int duration = camera.duration;
        if ( !rendered_transient.is_null() ) {
            assert(camera.valid_transient());
            for (int i = 0; i < num_duple_img; i++) // previous: nworker
                transient_per_thread[i].resize(nder * num_pixels * duration, Spectrum(0.0f));
        }

        constexpr int num_samples_per_block = 128;
        long long num_samples = static_cast<long long>(options.num_samples_secondary_edge_indirect)*num_pixels;
        const long long num_block = static_cast<long long>(std::ceil(static_cast<Float>(num_samples)/num_samples_per_block));
        num_samples = num_block*num_samples_per_block;
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
        for (long long index_block = 0; index_block < num_block; ++index_block) {
            for (int omp_i = 0; omp_i < num_samples_per_block; omp_i++) {
                const int tid = omp_get_thread_num();
                int idx_duple = options.mode == MEMORY_LOCK ? 0 : tid; 
                RndSampler sampler(options.seed, m_taskId[tid] = index_block*num_samples_per_block + omp_i);
                // indirect contribution of edge term
                int shape_id;
                RayAD edgeRay;
                Float edgePdf;
                const Edge &rEdge = scene.sampleEdgeRay(sampler.next3D(), shape_id, edgeRay, edgePdf);
                if (shape_id >= 0) {
                    EdgeEvaluationRecord eRec;
                    evalEdge(scene, shape_id, rEdge, edgeRay, &sampler, eRec);
                    if (eRec.its1.isValid() && eRec.its2.isValid()) {
                        bool zeroVelocity = eRec.value0.der.isZero(Epsilon) && eRec.value1.der.isZero(Epsilon);
                        if (!zeroVelocity) {
                            traceRayFromEdgeSegement(scene, eRec, edgePdf, options.max_bounces, &sampler,
                                image_per_thread[idx_duple], &(transient_per_thread[idx_duple]) );
                        }
                    }
                }
            }

            if ( !options.quiet ) {
                omp_set_lock(&messageLock);
                progressIndicator(Float(index_block + 1)/num_block);
                omp_unset_lock(&messageLock);
            }
        }
        if ( !options.quiet )
                std::cout << std::endl;

        for ( int i = 0; i < num_duple_img; ++i ) 
            for ( int j = 0; j < nder; ++j )
                for ( int idx_pixel = 0; idx_pixel < num_pixels; ++idx_pixel ) {
                    int offset1 = ((j + 1)*num_pixels + idx_pixel)*3,
                        offset2 = j*num_pixels + idx_pixel;
                    rendered_image[offset1    ] += image_per_thread[i][offset2][0]/static_cast<Float>(num_samples);
                    rendered_image[offset1 + 1] += image_per_thread[i][offset2][1]/static_cast<Float>(num_samples);
                    rendered_image[offset1 + 2] += image_per_thread[i][offset2][2]/static_cast<Float>(num_samples);
                }
        if ( !rendered_transient.is_null() ) {
            assert(camera.valid_transient());
            for ( int i = 0; i < num_duple_img; ++i ) 
                for ( int j = 0; j < nder; ++j )
                    for ( int idx_pixel = 0; idx_pixel < num_pixels; ++idx_pixel ) {
                        int offset1 = ((j + 1)*num_pixels + idx_pixel)*3,
                                offset2 = j*num_pixels + idx_pixel;
                        for (int i_bin = 0; i_bin < duration; i_bin++) {
                            int offset3 = offset1*duration + i_bin*3,
                                    offset4 = offset2*duration + i_bin;
                            rendered_transient[offset3    ] += transient_per_thread[i][offset4][0] / static_cast<Float>(num_samples);
                            rendered_transient[offset3 + 1] += transient_per_thread[i][offset4][1] / static_cast<Float>(num_samples);
                            rendered_transient[offset3 + 2] += transient_per_thread[i][offset4][2] / static_cast<Float>(num_samples);
                        }
                    }
        }
    }
}


void IntegratorAD_PathSpace::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient) const { 
    if ( !options.quiet ) {
        std::cout << std::scientific << std::setprecision(1) << "[INFO] grad_threshold = " << options.grad_threshold
                  << std::endl;
    }
    IntegratorAD::render(scene, options, rendered_image, rendered_transient);
}
