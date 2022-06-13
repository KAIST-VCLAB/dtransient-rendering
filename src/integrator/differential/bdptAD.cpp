#include "bdptAD.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <vector>
#include <omp.h>

static const int nworker = omp_get_num_procs();
static std::vector<SpectrumAD> image_per_thread[BDPT_MAX_THREADS];
static std::vector<SpectrumAD> transient_per_thread[BDPT_MAX_THREADS];


std::pair<Spectrum,int> BidirectionalPathTracerAD::pixelColor(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y,
                                               std::tuple<Spectrum, Float, int> *ret_trans) const
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// len(record) <= BDPT_MAX_PATH_LENGTH
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// Called by IntegratorAD::renderPrimaryEdges
    const int tid = omp_get_thread_num();
    assert(tid < BDPT_MAX_THREADS);

    Spectrum ret(0.0f);
    Intersection its;
    int num_path = 0; 
    if ( scene.rayIntersect(Ray(scene.camera.samplePrimaryRay(x, y)), false, its) ) {
        int max_bounces = options.max_bounces;
        if ( ret_trans == nullptr) {
            Spectrum *rad = m_rad[tid];
            for (int i = 0; i <= max_bounces; i++) rad[i] = Spectrum(0.0);
            radiance(scene, sampler, its, max_bounces, &rad[0]);
            for ( int i = 0; i <= max_bounces; ++i ) ret += rad[i];
        } else{
            Spectrum *rad = m_rad[tid];
            num_path = radiance(scene, sampler, its, max_bounces, rad, ret_trans);
            auto primary_distance = (scene.camera.cpos.val - its.p).norm();
            for ( int i = 0; i <= max_bounces; ++i ) ret += rad[i];
            for ( int i = 0; i < num_path; ++i ){
                std::get<1>(ret_trans[i]) += primary_distance;
            }
        }
    }
    return {ret, num_path};
}

SpectrumAD BidirectionalPathTracerAD::pixelColorAD(const Scene &scene, const RenderOptions &options, RndSampler *sampler, Float x, Float y, ptr<float> temp_hist) const
{
    const int tid = omp_get_thread_num();
    const CameraTransient& camera = scene.camera; 
    const int max_bounces = options.max_bounces;
    const int duration = scene.camera.duration;
    const int num_pixels = scene.camera.getNumPixels();
    const int num_samples = options.num_samples;
    Float inv_area;
    {
        Float fov_factor = camera.cam_to_ndc(0, 0);
        Float aspect_ratio = static_cast<Float>(camera.width)/camera.height;
        inv_area = 0.25f*fov_factor*fov_factor*aspect_ratio;
    }

    bidir::PathNodeAD *cameraPath = m_pathAD[2*tid], *lightPath = m_pathAD[2*tid + 1];
    /// Building the camera sub-path
    RayAD cameraRay = camera.samplePrimaryRayAD(x, y);
    int cameraPathLen;
    if ( scene.rayIntersectAD(cameraRay, false, cameraPath[0].itsAD)){ 
        cameraPath[0].its = cameraPath[0].itsAD.toIntersection();
        cameraPath[0].pdf0 = inv_area;
        cameraPath[0].pdf0 *= std::abs(cameraPath[0].its.geoFrame.n.dot(-cameraRay.dir.val))/
                              (std::pow(camera.cframe.n.val.dot(cameraRay.dir.val), 3.0f)*cameraPath[0].its.t*cameraPath[0].its.t);
        cameraPath[0].throughput = SpectrumAD(Spectrum(1.0f));
        cameraPath[0].opd = cameraPath[0].itsAD.opd;
        cameraPathLen = 1;
        if ( max_bounces > 0 )
            cameraPathLen = bidir::buildPathAD( scene, sampler, max_bounces + 1, false, cameraPath);
        bidir::preprocessPathAD(cameraPathLen, false, cameraPath);
    } else
        cameraPathLen = 0;

    /// Building the light sub-path
    lightPath[0].throughput = scene.sampleEmitterPosition(sampler->next2D(), lightPath[0].itsAD, lightPath[0].J, &lightPath[0].pdf0);
    lightPath[0].throughput *= lightPath[0].J;
    // lightPath[0].throughput == Le(x0 -> x1) *J(x0) / pdf(x0) for an arbitrary x1 (i.e., uniform emission)
    lightPath[0].its = lightPath[0].itsAD.toIntersection();
    lightPath[0].opd = FloatAD(0.0);
    int lightPathLen = 1;
    if ( max_bounces > 0) {
        Vector wo;
        Float &pdf = lightPath[1].pdf0;
        lightPath[0].its.ptr_emitter->sampleDirection(sampler->next2D(), wo, &pdf);
        wo = lightPath[0].its.geoFrame.toWorld(wo);
        if ( scene.rayIntersect(Ray(lightPath[0].its.p, wo), true, lightPath[1].its) ) {
            scene.getPoint(lightPath[1].its, lightPath[0].itsAD.p, lightPath[1].itsAD, lightPath[1].J);
            lightPath[0].wo = lightPath[1].itsAD.p - lightPath[0].itsAD.p;
            lightPath[1].opd = lightPath[0].opd + lightPath[1].itsAD.opd;
            FloatAD d = lightPath[0].wo.norm();
            lightPath[0].wo /= d;
            FloatAD G = lightPath[1].itsAD.geoFrame.n.dot(-lightPath[0].wo).abs() / d.square();
            pdf *= G.val;
            lightPath[1].throughput = lightPath[0].throughput * lightPath[0].its.ptr_emitter->evalDirectionAD(lightPath[0].itsAD.geoFrame.n, lightPath[0].wo) * lightPath[1].J * G / pdf;
            // lightPath[1].throughput == lightPath[0].throughput * G(x0 -> x1) * J(x1) / (pdf(w01)*G.val)
            if ( max_bounces > 1)
                lightPathLen = bidir::buildPathAD(scene, sampler, max_bounces, true, &lightPath[1]) + 1;
            else
                lightPathLen = 2;
        }
    }
    bidir::preprocessPathAD(lightPathLen, false, lightPath);

    SpectrumAD ret(Spectrum::Zero());
    if ( cameraPathLen > 0 && cameraPath[0].its.isEmitter() )
        ret += cameraPath[0].itsAD.Le(-cameraRay.dir);

    for ( int i = 1; i <= max_bounces; ++i ) {
        for ( int s = std::max(-1, i - lightPathLen); s <= std::min(i, cameraPathLen - 1); ++s ) {
            int t = i - 1 - s, idx_pixel = -1;
            SpectrumAD value(Spectrum(0.0));
            Float camera_val = 0.0f;
            Vector dir;
            FloatAD pathDist;
            if ( s >= 0 && t >= 0) {
                /// Value case 1: BDPT
                // Value 1 assigning
                value = cameraPath[s].throughput * lightPath[t].throughput;
                pathDist = cameraPath[s].opd + lightPath[t].opd + cameraPath[s].itsAD.getOpdFrom(lightPath[t].itsAD.p);
            }
            else if ( s == -1) {
                /// Value case 2: direct component for light subpaths
                // direct connect lightPathNode to camera (cannot be the lightPath origin/emitter)
                assert(t > 0);
                if ( scene.isVisible(lightPath[t].its.p, true, camera.cpos.val, false) ) {
                    Vector2 pix_uv;
                    camera_val = camera.sampleDirect(lightPath[t].its.p, pix_uv, dir);
                    if ( camera_val > Epsilon ) {
                        Vector wi = lightPath[t].its.toWorld(lightPath[i].its.wi), wo = dir, wo_local = lightPath[t].its.toLocal(wo);
                        Float wiDotGeoN = wi.dot(lightPath[t].its.geoFrame.n), woDotGeoN = wo.dot(lightPath[t].its.geoFrame.n);
                        if (wiDotGeoN * lightPath[i].its.wi.z() > 0 || woDotGeoN * wo_local.z() > 0) {
                            RayAD tmpRay = scene.camera.samplePrimaryRayAD(pix_uv[0], pix_uv[1]);
                            IntersectionAD itsAD;
                            if ( scene.rayIntersectAD(tmpRay, false, itsAD) && (itsAD.p.val - lightPath[t].its.p).norm() < ShadowEpsilon ) {
                                // Value 2 assigning
                                value = lightPath[t-1].throughput;
                                VectorAD wiAD = lightPath[t-1].itsAD.p - itsAD.p;
                                FloatAD d = wiAD.norm();
                                wiAD /= d;
                                FloatAD G = itsAD.geoFrame.n.dot(wiAD).abs()/d.square();
                                if ( t > 1)
                                    value *= lightPath[t-1].its.ptr_bsdf->evalAD(lightPath[t-1].itsAD, lightPath[t-1].itsAD.toLocal(-wiAD), EBSDFMode::EImportanceWithCorrection) * G / lightPath[t].pdf0;
                                else
                                    value *= lightPath[0].its.ptr_emitter->evalDirectionAD(lightPath[0].itsAD.geoFrame.n, -wiAD) * G / lightPath[1].pdf0;
                                itsAD.wi = itsAD.toLocal(wiAD);
                                VectorAD woLocal = itsAD.toLocal(-tmpRay.dir);
                                value *= camera_val * itsAD.ptr_bsdf->evalAD(itsAD, woLocal, EBSDFMode::EImportanceWithCorrection);
                                idx_pixel = camera.getPixelIndex(pix_uv);
                                auto temp = lightPath[t].itsAD.getOpdFrom(camera.cpos);
                                auto temp2 = temp - (camera.cpos - lightPath[t].itsAD.p).norm();
                                if (!lightPath[t].itsAD.ptr_bsdf->isTransmissive() && temp2.abs().val > 1e-4){
                                    std::cerr << std::scientific << std::setprecision(4) << "\n[BUG3] At " << lightPath[t].itsAD.ptr_bsdf->toString()
                                              << ", opd=" << temp.val << ", t" << (camera.cpos - lightPath[t].itsAD.p).norm().val << std::endl;
                                }
                                pathDist = lightPath[t].opd + lightPath[t].itsAD.getOpdFrom(camera.cpos);
                            }
                        }
                    }
                }
            }
            else if ( t == -1 ) {
                /// Value case 3 emitter endpoint for camera subpaths
                assert(s > 0);
                // Value 3 assigning
                value = cameraPath[s].throughput * cameraPath[s].itsAD.Le(-cameraPath[s-1].wo);
                pathDist = cameraPath[s].opd;
            }

            if ( s >= 0 && t >= 0 ) {
                value *= bidir::evalSegmentAD(scene, cameraPath[s].itsAD, lightPath[t].itsAD, t == 0);
                         // G(lp[t] -> cp[s]) * rho(lp[t] -> cp[s] -> cp[s-1]) * (t==0)? Le(lp[t] -> cp[s])
                         //                                                            : rho(lp[t-1] -> lp[t] -> cp[s])
            }

            /// Combination strategy
            if ( !value.isZero(Epsilon) ) {
                Float f = 1.0, pdf1, pdf0;
                Float distSqr;

                if ( s >= 0 ) {
                    const bidir::PathNodeAD &cur = cameraPath[s];
                    pdf0 = cur.pdf0;

                    if ( t >= 0 ) {
                        const Intersection &its = lightPath[t].its;
                        dir = cur.its.p - its.p;
                        distSqr = dir.squaredNorm();
                        dir /= std::sqrt(distSqr);

                        pdf1 = t > 0 ? its.pdfBSDF(its.toLocal(dir)) : its.geoFrame.n.dot(dir)/M_PI;
                        pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir))/distSqr;
                    } else {
                        /// Value case 3 emitter endpoint for camera subpaths
                        assert(t == -1);
                        if (cur.its.isEmitter())
                            pdf1 = scene.pdfEmitterSample(cur.its);
                        else // If cur=cameraPath[s] is not an emitter, value is already zero.
                            pdf1 = 1.0;
                    }
                    /// Recall: mis_ratio(pdf0, pdf1) == (pdf0/pdf1)^2        <bidir_utils.h>
                    f += bidir::mis_ratio(pdf1, pdf0); 

                    if ( s >= 1 ) {
                        const bidir::PathNodeAD &next = cameraPath[s - 1];
                        pdf0 *= next.pdf0;

                        if ( t >= 0 ) {
                            Intersection &its = cur.its1;
                            its.wi = its.toLocal((lightPath[t].its.p - cur.its.p).normalized());
                            pdf1 *= its.pdfBSDF(its.toLocal(-next.wo.val));
                        } else {
                            /// Value case 3 emitter endpoint for camera subpaths
                            pdf1 *= cur.its.geoFrame.n.dot(-next.wo.val)/M_PI;
                        }
                        pdf1 *= next.G1;
                        f += cur.w * bidir::mis_ratio(pdf1, pdf0); 
                    }
                }

                if ( t >= 0 ) {
                    const bidir::PathNodeAD &cur = lightPath[t];
                    pdf0 = cur.pdf0;

                    if ( s >= 0 ) {
                        const Intersection &its = cameraPath[s].its;
                        dir = cur.its.p - its.p;
                        distSqr = dir.squaredNorm();
                        dir /= std::sqrt(distSqr);

                        pdf1 = its.pdfBSDF(its.toLocal(dir));
                        pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir))/distSqr;
                    } else {
                        /// Value case 2: direct component for light subpaths
                        assert(s == -1);
                        dir = cur.its.p - camera.cpos.val;
                        distSqr = dir.squaredNorm();
                        dir /= std::sqrt(distSqr);

                        pdf1 = inv_area/(std::pow(camera.cframe.n.val.dot(dir), 3.0f)*distSqr);
                        pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir));
                    }
                    f += bidir::mis_ratio(pdf1, pdf0); 

                    if ( t >= 1 ) {
                        const bidir::PathNodeAD &next = lightPath[t - 1];
                        pdf0 *= next.pdf0;

                        Intersection &its = cur.its1;
                        its.wi = ( s >= 0 ? cameraPath[s].its.p : camera.cpos.val ) - cur.its.p;
                        its.wi = its.toLocal(its.wi.normalized());
                        pdf1 *= its.pdfBSDF(its.toLocal(-next.wo.val))*next.G1;
                        f += cur.w*bidir::mis_ratio(pdf1, pdf0); 
                    }
                }

                value /= f; /// "value" is fixed in the for(i)-for(s) block

                bool val_valid = std::isfinite(value.val[0]) && std::isfinite(value.val[1]) && std::isfinite(value.val[2]) && value.val.minCoeff() >= 0.0f;
                Float tmp_val = value.der.abs().maxCoeff();
                bool der_valid = std::isfinite(tmp_val) && tmp_val < options.grad_threshold;

                if ( val_valid && der_valid ) {
                    int i_bin_start, i_bin_end;
                    auto pathTime = pathDist*INV_C;
                    camera.bin_range(pathTime, i_bin_start, i_bin_end);
                    outtimeSamples += camera.clip_bin_index(i_bin_start, i_bin_end);
                    if ( idx_pixel < 0) {
                        assert (s >= 0);
                        ret += value;
                        if (!temp_hist.is_null()) {
                                for (int i_bin = i_bin_start; i_bin <= i_bin_end; i_bin++){
                                    SpectrumAD val_trans = value * camera.eval_tsens(pathTime, i_bin);
                                    for (int i_rgb = 0; i_rgb < 3; i_rgb++)
                                        temp_hist[i_bin*3 + i_rgb] += float(static_cast<double>(val_trans.val(i_rgb))/num_samples);
                                    for (int ch = 1; ch <= nder; ch++){
                                        int offset = (ch*num_pixels*duration + i_bin)*3;
                                        for (int i_rgb = 0; i_rgb < 3; i_rgb++)
                                            temp_hist[offset + i_rgb] += float(static_cast<double>(val_trans.grad(ch-1)(i_rgb))/num_samples);
                                    }
                                }
                        }
                    } else {
                        assert (s < 0);
                        int idx_duple = set_pixel_lock(idx_pixel, tid); 
                        image_per_thread[idx_duple][idx_pixel] += value; 

                        if (!temp_hist.is_null()) {
                            for (int i_bin = i_bin_start; i_bin <= i_bin_end; i_bin++)
                                transient_per_thread[idx_duple][idx_pixel * duration + i_bin] += value * camera.eval_tsens(pathTime, i_bin);
                        }
                        unset_pixel_lock(idx_pixel); 
                    }
                } else {
                    omp_set_lock(&messageLock);
                    if ( !options.quiet ) {
                        if (!val_valid) {
                            std::cerr << std::scientific << std::setprecision(2) << "\n[WARN] Invalid path contribution: [" << value.val << "]"
                                      << std::endl;
                        }
                        if (!der_valid)
                            std::cerr << std::scientific << std::setprecision(2) << "\n[WARN] Rejecting large gradient: [" << value.der << "]" << std::endl;
                    }
                    ++rejectedSamples;
                    omp_unset_lock(&messageLock);
                }
            }

        }
    }

    return ret;
}


void BidirectionalPathTracerAD::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient) const
{
    const Camera &camera = scene.camera;
    int num_pixels = camera.getNumPixels();
    int duration = scene.camera.duration;

    int num_duple_img = options.mode == MEMORY_LOCK ? 1 : nworker; 
    long long tot_samples = static_cast<long long>(num_pixels)*options.num_samples; 
    if (tot_samples > 0) { 
        for (int i = 0; i < num_duple_img; i++) { 
            image_per_thread[i].resize(num_pixels);
            std::fill(image_per_thread[i].begin(), image_per_thread[i].end(), SpectrumAD(Spectrum::Zero()));
            if (!rendered_transient.is_null()) {
                assert(scene.camera.valid_transient());
                int num_voxels = num_pixels * duration;
                transient_per_thread[i].resize(num_voxels);
                std::fill(transient_per_thread[i].begin(), transient_per_thread[i].end(), SpectrumAD(Spectrum::Zero()));
            }
        }
    }
    rejectedSamples = 0;
    outtimeSamples = 0;
    init_pixel_lock(num_pixels);
    IntegratorAD_PathSpace::render(scene, options, rendered_image, rendered_transient);

    if (tot_samples > 0)
        for ( int i = 0; i < num_duple_img; ++i ) 
            for ( int idx_pixel = 0; idx_pixel < num_pixels; ++idx_pixel ) {
                for ( int ichannel = 0; ichannel < 3; ++ichannel ) {
                    rendered_image[idx_pixel * 3 + ichannel] += static_cast<float>(image_per_thread[i][idx_pixel].val[ichannel] / tot_samples);
                    for (int i_bin = 0; i_bin < duration; i_bin++)
                        rendered_transient[(idx_pixel * duration + i_bin)*3 + ichannel] += static_cast<float>(transient_per_thread[i][idx_pixel*duration + i_bin].val[ichannel] / tot_samples);
                }
                for (int ch = 1; ch <= nder; ++ch ) {
                    int offset = (ch*num_pixels + idx_pixel)*3;
                    rendered_image[offset    ] += static_cast<float>(image_per_thread[i][idx_pixel].grad(ch-1)(0))/tot_samples;
                    rendered_image[offset + 1] += static_cast<float>(image_per_thread[i][idx_pixel].grad(ch-1)(1))/tot_samples;
                    rendered_image[offset + 2] += static_cast<float>(image_per_thread[i][idx_pixel].grad(ch-1)(2))/tot_samples;
                    for (int i_bin = 0; i_bin < duration; i_bin++){
                        rendered_transient[offset*duration + i_bin*3    ] += static_cast<float>(transient_per_thread[i][idx_pixel*duration + i_bin].grad(ch-1)(0))/tot_samples;
                        rendered_transient[offset*duration + i_bin*3 + 1] += static_cast<float>(transient_per_thread[i][idx_pixel*duration + i_bin].grad(ch-1)(1))/tot_samples;
                        rendered_transient[offset*duration + i_bin*3 + 2] += static_cast<float>(transient_per_thread[i][idx_pixel*duration + i_bin].grad(ch-1)(2))/tot_samples;
                    }
                }
            }
    for ( int i = 0; i < num_duple_img; i++) {
        image_per_thread[i].clear();
        image_per_thread[i].shrink_to_fit();
        transient_per_thread[i].clear();
        transient_per_thread[i].shrink_to_fit();
    }

    // Primary edge sampling
    if ( options.num_samples_primary_edge > 0 && scene.ptr_edgeManager->getNumPrimaryEdges() > 0 )
        renderPrimaryEdges(scene, options, rendered_image, rendered_transient);
#ifdef USE_BOUNDARY_NEE
    if ( options.num_samples_secondary_edge_direct > 0 )
        renderEdgesDirect(scene, options, rendered_image, rendered_transient);
#endif

    if ( options.num_samples_secondary_edge_indirect > 0 )
        renderEdges(scene, options, rendered_image, rendered_transient);

    destroy_pixel_lock(); 

    if ( rejectedSamples ) {
        std::cerr << "[WARN] " << rejectedSamples << " samples rejected." << std::endl;
    }
}
