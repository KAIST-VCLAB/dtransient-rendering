#include "bdpt.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <vector>
#include <omp.h>

#define BDPT_USE_FULL_SCHEME
// #define BDPT_USE_INV_SCHEME


static const int nworker = omp_get_num_procs();
#if defined BDPT_USE_FULL_SCHEME || defined BDPT_USE_INV_SCHEME
    static std::vector<Spectrum> image_per_thread[BDPT_MAX_THREADS];
#endif


std::string BidirectionalPathTracer::getName() const {
#if defined BDPT_USE_FULL_SCHEME
    return "bdpt_full";
#elif defined BDPT_USE_INV_SCHEME
    return "bdpt_inv";
#else
    return "bdpt_limited";
#endif
}


std::pair<Spectrum, int> BidirectionalPathTracer::pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y,
                                                             std::tuple<Spectrum, Float, int> *ret_trans) const
{
    const int tid = omp_get_thread_num();
    assert(tid < BDPT_MAX_THREADS);

    Spectrum ret(0.0f);
    Intersection its;
    if ( scene.rayIntersect(Ray(scene.camera.samplePrimaryRay(x, y)), false, its) ) {
        int max_bounces = options.max_bounces;
        assert(max_bounces + 1 < BDPT_MAX_PATH_LENGTH);

        Spectrum *rad = m_rad[tid];
        bidir::radiance(scene, sampler, its, max_bounces, m_path[2*tid], m_path[2*tid + 1], &rad[0]);
        for ( int i = 0; i <= max_bounces; ++i ) ret += rad[i];
    }
    return {ret, 0}; 
}


SpectrumAD BidirectionalPathTracer::pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y, ptr<float> temp_hist) const
{
#ifdef BDPT_USE_FULL_SCHEME
    const int tid = omp_get_thread_num();
    const int max_bounces = options.max_bounces;
    const Camera &camera = scene.camera;
    Float inv_area;
    {
        Float fov_factor = camera.cam_to_ndc(0, 0);
        Float aspect_ratio = static_cast<Float>(camera.width)/camera.height;
        inv_area = 0.25f*fov_factor*fov_factor*aspect_ratio;
    }

    bidir::PathNode *cameraPath = m_path[2*tid], *lightPath = m_path[2*tid + 1];

    // Building the camera sub-path
    Ray cameraRay = camera.samplePrimaryRay(x, y);
    int cameraPathLen;
    if ( scene.rayIntersect(cameraRay, false, cameraPath[0].its) ) {
        cameraPath[0].pdf0 = inv_area;
        cameraPath[0].pdf0 *= std::abs(cameraPath[0].its.geoFrame.n.dot(-cameraRay.dir))/
                              (std::pow(camera.cframe.n.val.dot(cameraRay.dir), 3.0f)*cameraPath[0].its.t*cameraPath[0].its.t);
        cameraPath[0].throughput = Spectrum(1.0f);
        cameraPathLen = 1;
        if ( max_bounces > 0 )
            cameraPathLen = bidir::buildPath(scene, sampler, max_bounces + 1, false, cameraPath);
        bidir::preprocessPath(cameraPathLen, false, cameraPath);
    } else
        cameraPathLen = 0;

    // Building the light sub-path
    lightPath[0].throughput = scene.sampleEmitterPosition(sampler->next2D(), lightPath[0].its, &lightPath[0].pdf0);
    int lightPathLen = 1;
    if ( max_bounces > 0 ) {
        Vector wo;
        Float &pdf = lightPath[1].pdf0;
        Float tmp = lightPath[0].its.ptr_emitter->sampleDirection(sampler->next2D(), wo, &pdf);
        wo = lightPath[0].its.geoFrame.toWorld(wo);
        if ( scene.rayIntersect(Ray(lightPath[0].its.p, wo), true, lightPath[1].its) ) {
            lightPath[0].wo = wo;
            Float G = std::abs(lightPath[1].its.geoFrame.n.dot(-wo))/(lightPath[1].its.t*lightPath[1].its.t);
            pdf *= G;
            lightPath[1].throughput = lightPath[0].throughput*tmp;

            if ( max_bounces > 1 )
                lightPathLen = bidir::buildPath(scene, sampler, max_bounces, true, &lightPath[1]) + 1;
            else
                lightPathLen = 2;
        }
    }
    bidir::preprocessPath(lightPathLen, false, lightPath);

    Spectrum ret = Spectrum::Zero();
    if ( cameraPathLen > 0 && cameraPath[0].its.isEmitter() )
        ret = cameraPath[0].its.Le(-cameraRay.dir);

    for ( int i = 1; i <= max_bounces; ++i ) {
        for ( int s = std::max(-1, i - lightPathLen); s <= std::min(i, cameraPathLen - 1); ++s ) {
            int t = i - 1 - s, idx_pixel = -1;

            Spectrum value(1.0f);
            Float camera_val = 0.0f;
            Vector dir;

            if ( s >= 0 )
                value *= cameraPath[s].throughput;
            else {
                assert(t > 0);
                bool valid = false;
                if ( scene.isVisible(lightPath[t].its.p, true, camera.cpos.val, false) ) {
                    Vector2 pix_uv;
                    camera_val = camera.sampleDirect(lightPath[t].its.p, pix_uv, dir);
                    if ( camera_val > Epsilon ) {
                        valid = true;
                        value *= lightPath[t].its.evalBSDF(lightPath[t].its.toLocal(dir), EBSDFMode::EImportanceWithCorrection)*camera_val;
                        idx_pixel = camera.getPixelIndex(pix_uv);
                    }
                }
                if ( !valid ) value = Spectrum(0.0f);
            }
            if ( t >= 0 )
                value *= lightPath[t].throughput;
            else {
                assert(s > 0);
                value *= cameraPath[s].its.Le(-cameraPath[s - 1].wo);
            }
            if ( s >= 0 && t >= 0 )
                value *= bidir::evalSegment(scene, cameraPath[s].its, lightPath[t].its, t == 0);

            if ( !value.isZero(Epsilon) ) {
                Float f = 1.0f, pdf1, pdf0;
                Float distSqr;

                if ( s >= 0 ) {
                    const bidir::PathNode &cur = cameraPath[s];
                    pdf0 = cur.pdf0;

                    if ( t >= 0 ) {
                        const Intersection &its = lightPath[t].its;
                        dir = cur.its.p - its.p;
                        distSqr = dir.squaredNorm();
                        dir /= std::sqrt(distSqr);

                        pdf1 = t > 0 ? its.pdfBSDF(its.toLocal(dir)) : its.geoFrame.n.dot(dir)/M_PI;
                        pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir))/distSqr;
                    } else {
                        assert(t == -1);
                        pdf1 = scene.pdfEmitterSample(cur.its);
                    }
                    f += bidir::mis_ratio(pdf1, pdf0);

                    if ( s >= 1 ) {
                        const bidir::PathNode &next = cameraPath[s - 1];
                        pdf0 *= next.pdf0;

                        if ( t >= 0 ) {
                            Intersection &its = cur.its1;
                            its.wi = its.toLocal((lightPath[t].its.p - cur.its.p).normalized());
                            pdf1 *= its.pdfBSDF(its.toLocal(-next.wo));
                        } else {
                            pdf1 *= cur.its.geoFrame.n.dot(-next.wo)/M_PI;
                        }
                        pdf1 *= next.G1;

                        f += cur.w*bidir::mis_ratio(pdf1, pdf0);
                    }
                }

                if ( t >= 0 ) {
                    const bidir::PathNode &cur = lightPath[t];
                    pdf0 = cur.pdf0;

                    if ( s >= 0 ) {
                        const Intersection &its = cameraPath[s].its;
                        dir = cur.its.p - its.p;
                        distSqr = dir.squaredNorm();
                        dir /= std::sqrt(distSqr);

                        pdf1 = its.pdfBSDF(its.toLocal(dir));
                        pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir))/distSqr;
                    } else {
                        assert(s == -1);
                        dir = cur.its.p - camera.cpos.val;
                        distSqr = dir.squaredNorm();
                        dir /= std::sqrt(distSqr);

                        pdf1 = inv_area/(std::pow(camera.cframe.n.val.dot(dir), 3.0f)*distSqr);
                        pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir));
                    }
                    f += bidir::mis_ratio(pdf1, pdf0);

                    if ( t >= 1 ) {
                        const bidir::PathNode &next = lightPath[t - 1];
                        pdf0 *= next.pdf0;

                        Intersection &its = cur.its1;
                        its.wi = ( s >= 0 ? cameraPath[s].its.p : camera.cpos.val ) - cur.its.p;
                        its.wi = its.toLocal(its.wi.normalized());
                        pdf1 *= its.pdfBSDF(its.toLocal(-next.wo))*next.G1;

                        f += cur.w*bidir::mis_ratio(pdf1, pdf0);
                    }
                }

                value /= f;
                if ( !std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2]) ) {
                    omp_set_lock(&messageLock);
                    std::cerr << std::fixed << std::setprecision(2)
                              << "\n[WARN] Invalid path contribution: [" << value.transpose() << "]" << std::endl;
                    omp_unset_lock(&messageLock);
                } else {
                    if ( idx_pixel < 0 ) {
                        assert(s >= 0);
                        ret += value;
                    } else {
                        assert(s < 0);
                        image_per_thread[tid][idx_pixel] += value;
                    }
                }
            }
        }
    }

    return SpectrumAD(ret);

#elif defined BDPT_USE_INV_SCHEME
    const int tid = omp_get_thread_num();
    const int max_bounces = options.max_bounces;
    Intersection its;
    Float pdf;
    Spectrum power = scene.sampleEmitterPosition(sampler->next2D(), its, &pdf);
    // direct connection to the sensor
    {
        Vector2 pix_uv;
        Vector dir;
        Float transmittance = scene.sampleAttenuatedSensorDirect(its, sampler, max_bounces, pix_uv, dir);
        if (transmittance != 0.0f) {
            Spectrum value = power * transmittance * its.ptr_emitter->evalDirection(its.geoFrame.n, dir);
            int idx_pixel = scene.camera.getPixelIndex(pix_uv);
            image_per_thread[tid][idx_pixel] += value;
        }
    }
    Ray ray;
    ray.org = its.p;
    power *= its.ptr_emitter->sampleDirection(sampler->next2D(), ray.dir);
    ray.dir = its.geoFrame.toWorld(ray.dir);
    if ( scene.rayIntersect(ray, true, its) ) {
        // direct connection to the sensor
        {
            Vector2 pix_uv;
            Vector dir;
            Float transmittance = scene.sampleAttenuatedSensorDirect(its, sampler, max_bounces, pix_uv, dir);
            if (transmittance != 0.0f) {
                Vector wi = its.toWorld(its.wi), wo = dir, wo_local = its.toLocal(wo);
                if (wi.dot(its.geoFrame.n) * its.wi.z() > 0 && wo.dot(its.geoFrame.n) * wo_local.z() > 0) {
                    Spectrum value = power * transmittance * its.ptr_bsdf->eval(its, wo_local, EBSDFMode::EImportanceWithCorrection);
                    int idx_pixel = scene.camera.getPixelIndex(pix_uv);
                    image_per_thread[tid][idx_pixel] += value;
                }
            }
        }
        if (max_bounces > 1) {
            // bi-directional evaluate importance
            std::vector<Spectrum> L(max_bounces, power);
            std::pair<int, Spectrum> pathThroughput[BDPT_MAX_PATH_LENGTH + 1];
            int idx_pixel = scene.camera.getPixelIndex(Vector2(x, y));
            int num_path = bidir::weightedImportance(scene, sampler, its, max_bounces-1, idx_pixel, m_path[2*tid], m_path[2*tid + 1], &L[0], pathThroughput);
            for (int i = 0; i < num_path; i++) {
                image_per_thread[tid][pathThroughput[i].first] += pathThroughput[i].second;
            }
        }
    }
    return SpectrumAD(Spectrum::Zero());

#else
    return SpectrumAD(pixelColor(scene, options, sampler, x, y));
#endif
}


void BidirectionalPathTracer::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient) const 
{
#if defined BDPT_USE_FULL_SCHEME || defined BDPT_USE_INV_SCHEME
    const Camera &camera = scene.camera;
    int num_pixels = camera.getNumPixels();
    for (int i = 0; i < nworker; i++) {
        image_per_thread[i].resize(num_pixels);
        std::fill(image_per_thread[i].begin(), image_per_thread[i].end(), Spectrum(0.0f));
    }
#endif
    IntegratorAD_PathSpace::render(scene, options, rendered_image);

#if defined BDPT_USE_FULL_SCHEME || defined BDPT_USE_INV_SCHEME
    long long tot_samples = static_cast<long long>(num_pixels)*options.num_samples;
    for ( int i = 0; i < nworker; ++i )
        for ( int idx_pixel = 0; idx_pixel < num_pixels; ++idx_pixel )
            for ( int ichannel = 0; ichannel < 3; ++ichannel )
                rendered_image[idx_pixel*3 + ichannel] += static_cast<float>(image_per_thread[i][idx_pixel][ichannel]/tot_samples);
#endif

}
