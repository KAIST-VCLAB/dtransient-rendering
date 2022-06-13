#include <iomanip>
#include "bidir_utils.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"

namespace bidir {

    namespace meta {

        // For bidir::radiance()
        int evalPaths(const Scene &scene, int max_bounces,
                       int cameraPathLength, const PathNode *cameraPath, int lightPathLength, const PathNode *lightPath,
                       Spectrum *ret, std::tuple<Spectrum, Float, int> *ret_trans = nullptr)
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////
            /// @input:
            /// cameraPath[0] ... cameraPath[cameraPathLength-1]
            ///     (CONSTRAINT: cameraPath[i].opd contains |camera - cameraPath[0]|)
            /// lightPath[0] ... lightPath[lightPathLength-1]
            ///
            /// @output:
            /// For 0 <= i <= d : (d <= max_bounces)
            ///     ret[i] += radiance by paths from cameraPath[0] consisting of i segments and i+1 vertices
            ///               (WARN: without initialization by zero!!)
            /// For 0 <= i < len(ret_trans) (<= (mb+1)(mb+2)/2 - 1)
            ///     ret_trans[i] = (value, pathDist, depth) (with initialization)
            /// @return: len(ret_trans)
            ////////////////////////////////////////////////////////////////////////////////////////////////////
            assert(cameraPathLength > 0);
            assert(cameraPathLength <= max_bounces + 1);
            assert(lightPathLength > 0);
            if (lightPathLength > max_bounces) {
                std::cout << lightPathLength << " >" << max_bounces << std::endl;
            }
            assert(lightPathLength <= max_bounces);
            assert(ret_trans==nullptr || scene.camera.valid_transient());
            int idx_trans = 0;
            Float pathDist;
            for ( int i = 1; i <= max_bounces; ++i ) {
                for ( int s = std::max(0, i - lightPathLength); s <= std::min(i, cameraPathLength - 1); ++s ) {
                    int t = i - 1 - s;

                    Spectrum value = cameraPath[s].throughput;
                    if ( t >= 0 ) {
                        /// cameraPath[0] -...- cameraPath[s] - lightPath[t] -...- lightPath[0]
                        value *= evalSegment(scene, cameraPath[s].its, lightPath[t].its, t == 0) *
                                 lightPath[t].throughput;
                        pathDist = cameraPath[s].opd + lightPath[t].opd + cameraPath[s].its.getOpdFrom(lightPath[t].its.p);
                    }
                    else {
                        /// cameraPath[0] -...- cameraPath[s==i]
                        assert(s > 0);
                        value *= cameraPath[s].its.Le(-cameraPath[s - 1].wo);
                        pathDist = cameraPath[s].opd;
                    }

                    if ( !value.isZero(Epsilon) ) {
                        /// Combination strategy
                        Float f = 1.0f, pdf1, pdf0;
                        Vector dir;
                        Float distSqr;

                        if ( s > 0 ) {
                            const PathNode &cur = cameraPath[s];
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
                            f += mis_ratio(pdf1, pdf0);

                            if ( s > 1 ) {
                                const PathNode &next = cameraPath[s - 1];
                                pdf0 *= next.pdf0;

                                if ( t >= 0 ) {
                                    Intersection &its = cur.its1;
                                    its.wi = its.toLocal((lightPath[t].its.p - cur.its.p).normalized());
                                    pdf1 *= its.pdfBSDF(its.toLocal(-next.wo));
                                } else {
                                    pdf1 *= cur.its.geoFrame.n.dot(-next.wo)/M_PI;
                                }
                                pdf1 *= next.G1;

                                f += cur.w*mis_ratio(pdf1, pdf0);
                            }
                        }

                        if ( t >= 0 ) {
                            const PathNode &cur = lightPath[t];
                            pdf0 = cur.pdf0;

                            const Intersection &its = cameraPath[s].its;
                            dir = cur.its.p - its.p;
                            distSqr = dir.squaredNorm();
                            dir /= std::sqrt(distSqr);

                            pdf1 = its.pdfBSDF(its.toLocal(dir));
                            pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir))/distSqr;

                            f += mis_ratio(pdf1, pdf0);

                            if ( t >= 1 ) {
                                const PathNode &next = lightPath[t - 1];
                                pdf0 *= next.pdf0;

                                Intersection &its = cur.its1;
                                its.wi = its.toLocal((cameraPath[s].its.p - cur.its.p).normalized());
                                pdf1 *= its.pdfBSDF(its.toLocal(-next.wo))*next.G1;

                                f += cur.w*mis_ratio(pdf1, pdf0);
                            }
                        }

                        value /= f;
                        if ( std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]) ) {
                            ret[i] += value;
                            if (ret_trans) {
                                ret_trans[idx_trans] = {value, pathDist, i};
                                idx_trans++;
                            }
                        }
                    }
                }
            }
            return idx_trans;
        }

        // For bidir::weightedImportance()
        std::pair<int, int> evalPaths(const Scene &scene, int max_bounces,
                      int cameraPathLength, const PathNode *cameraPath, int lightPathLength, const PathNode *lightPath, int pix_id,
                      const Spectrum *weight, std::pair<int, Spectrum>* ret, std::tuple<int, Spectrum, Float, int>* ret_trans = nullptr)
        {
            ////////////////////////////////////////////////////////////////////////////////////////////////////
            /// @input:
            /// camera - cameraPath[0] ... cameraPath[cameraPathLength-1]
            /// lightPath[0] ... lightPath[lightPathLength-1]
            ///
            /// @output:
            /// For 1 <= i <= d : (d <= max_bounces)
            ///     ret[i] = (idx_pixel, value)
            ///     value += importance for paths from camera consisting of i segments and i+1 vertices (counting w/o camera)
            ///               (WARN: without initialization by zero!!)
            ///               (WARN: no access to ret[0])
            /// For 0 <= i < len(ret_trans) (<= (mb+1)(mb+2)/2 - 1)
            ///     ret_trans[i] = (idx_pixel, value, pathDist, depth) (with initialization)
            /// @return: len(ret_trans)
            ////////////////////////////////////////////////////////////////////////////////////////////////////
            assert(cameraPathLength >= 0);
            assert(cameraPathLength <= max_bounces);
            assert(lightPathLength > 0);
            assert(lightPathLength <= max_bounces + 1);
            const CameraTransient &camera = scene.camera; 

            Float inv_area;
            {
                Float fov_factor = camera.cam_to_ndc(0, 0);
                Float aspect_ratio = static_cast<Float>(camera.width)/camera.height;
                inv_area = 0.25f*fov_factor*fov_factor*aspect_ratio;
            }
            assert(ret_trans==nullptr || camera.valid_transient());
            Float pathDist;
            int num_trans_path = 0;
            int num_valid_path = 1;
            for ( int i = 1; i <= max_bounces; ++i ) {
                for ( int s = std::max(-1, i - lightPathLength); s <= std::min(i - 1, cameraPathLength - 1); ++s ) {
                    int t = i - 1 - s, idx_pixel = -1;
                    Float camera_val;
                    Vector dir;
                    Spectrum value = lightPath[t].throughput;
                    bool check = true; 
                    if ( s >= 0 ) {
                        /// Case 1: camera - cameraPath[0] -...- cameraPath[s] - lightPath[t] -...- lightPath[0] (s,t >= 0)
                        value *= cameraPath[s].throughput *
                                 evalSegment(scene, cameraPath[s].its, lightPath[t].its, false);
                        pathDist = cameraPath[s].opd + lightPath[t].opd + cameraPath[s].its.getOpdFrom(lightPath[t].its.p);// NOTE: the input constraint
                    }
                    else {
                        /// Case 2: camera - lightPath[t==i] -...- lightPath[0] (t >= 1)
                        assert(t > 0);
                        bool valid = false;
                        check = false; 
                        if ( scene.isVisible(lightPath[t].its.p, true, camera.cpos.val, false) ) {
                            Vector2 pix_uv;
                            camera_val = camera.sampleDirect(lightPath[t].its.p, pix_uv, dir);
                            if ( camera_val > Epsilon ) {
                                valid = true;
                                check = true; 
                                value *= lightPath[t].its.evalBSDF(lightPath[t].its.toLocal(dir), EBSDFMode::EImportanceWithCorrection)*camera_val;
                                idx_pixel = camera.getPixelIndex(pix_uv);
                            }
                        }
                        if ( !valid ) value = Spectrum(0.0f);
                        pathDist = lightPath[t].opd + lightPath[t].its.getOpdFrom(camera.cpos.val);//(camera.cpos.val - lightPath[t].its.p).norm();
                    }

                    if ( !value.isZero(Epsilon) ) {
                        /// Combination strategy
                        Float f = 1.0f, pdf1, pdf0;
                        Float distSqr;

                        if ( s >= 0 ) {
                            const PathNode &cur = cameraPath[s];
                            pdf0 = cur.pdf0;

                            const Intersection &its = lightPath[t].its;
                            dir = cur.its.p - its.p;
                            distSqr = dir.squaredNorm();
                            dir /= std::sqrt(distSqr);

                            pdf1 = its.pdfBSDF(its.toLocal(dir));
                            pdf1 *= std::abs(cur.its.geoFrame.n.dot(-dir))/distSqr;

                            f += mis_ratio(pdf1, pdf0);

                            if ( s >= 1 ) {
                                const PathNode &next = cameraPath[s - 1];
                                pdf0 *= next.pdf0;
                                Intersection &its = cur.its1;
                                its.wi = its.toLocal((lightPath[t].its.p - cur.its.p).normalized());
                                pdf1 *= its.pdfBSDF(its.toLocal(-next.wo));
                                pdf1 *= next.G1;
                                f += cur.w*mis_ratio(pdf1, pdf0);
                            }
                        }

                        if ( t > 0 ) {
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
                            f += mis_ratio(pdf1, pdf0);

                            if ( t > 1 ) {
                                const bidir::PathNode &next = lightPath[t - 1];
                                pdf0 *= next.pdf0;

                                Intersection &its = cur.its1;
                                its.wi = ( s >= 0 ? cameraPath[s].its.p : camera.cpos.val ) - cur.its.p;
                                its.wi = its.toLocal(its.wi.normalized());
                                pdf1 *= its.pdfBSDF(its.toLocal(-next.wo))*next.G1;

                                f += cur.w*mis_ratio(pdf1, pdf0);
                            }
                        }

                        value /= f;

                        bool valid = std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]) && check; 
                        if ( valid ) { 
                            Spectrum value_weighted = value * (weight != nullptr ? weight[max_bounces - i] : Spectrum(1.0));
                            int temp_pix_id;
                            if ( idx_pixel < 0 ) {
                                assert(s >= 0);
                                ret[0].second += value_weighted; 
                                temp_pix_id = pix_id;
                            } else {
                                assert(s < 0);
                                ret[num_valid_path].first = idx_pixel;
                                ret[num_valid_path].second = value_weighted; 
                                num_valid_path++;
                                temp_pix_id = idx_pixel;
                            }
                            ret_trans[num_trans_path] = {temp_pix_id, value_weighted, pathDist, i};
                            num_trans_path++;
                        }
                    }
                }
            }
            return {num_valid_path, num_trans_path};
        }

    } //namespace meta

    int buildPath(const Scene &scene, RndSampler *sampler, int max_depth, bool importance, PathNode *path)
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @input: path[0] should be already constructed
        ///
        /// @output:
        ///     path[0], ..., path[d-1]
        ///     d <= max_depth
        /// @return: d (== len(path))
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        assert(path[0].its.isValid());
        int depth;
        for ( depth = 0; depth + 1 < max_depth; ++depth ) {
            PathNode &cur = path[depth], &next = path[depth + 1];

            Vector wo_local;
            Float &pdf = next.pdf0, bsdf_eta;
            Spectrum bsdf_val = cur.its.sampleBSDF(sampler->next3D(), wo_local, pdf, bsdf_eta,
                importance ? EBSDFMode::EImportanceWithCorrection : EBSDFMode::ERadiance);
            if ( bsdf_val.isZero(Epsilon) ) break;


            Vector &wo = cur.wo, wi;
            wo = cur.its.toWorld(wo_local); wi = cur.its.toWorld(cur.its.wi);
            if ( wi.dot(cur.its.geoFrame.n)*cur.its.wi.z() < Epsilon || wo.dot(cur.its.geoFrame.n)*wo_local.z() < Epsilon ) break;

            if ( !scene.rayIntersect(Ray(cur.its.p, wo), true, next.its) ) break;
            Float distSqr = next.its.t*next.its.t;
            pdf *= std::abs(next.its.geoFrame.n.dot(-wo))/distSqr;

            cur.G1 = std::abs(cur.its.geoFrame.n.dot(wo))/distSqr;
            next.throughput = cur.throughput*bsdf_val;
            next.opd = cur.opd + next.its.opd;
        }
        return depth + 1;
    }

    int buildPathAD(const Scene &scene, RndSampler *sampler, int max_depth, bool importance, PathNodeAD *path) {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// Input: path[0] should be already constructed
        ///
        /// Output:
        ///     path[0], ..., path[d-1]
        ///     d <= max_depth
        /// return d (== len(path))
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        assert(path[0].its.isValid());
        int depth;
        for (depth = 0; depth + 1 < max_depth; ++depth) {
            PathNodeAD &cur = path[depth], &next = path[depth + 1];
            Vector wo_local;
            Float &pdf = next.pdf0, bsdf_eta;
            if ( cur.its.sampleBSDF(sampler->next3D(), wo_local, pdf, bsdf_eta, importance ? EBSDFMode::EImportanceWithCorrection : EBSDFMode::ERadiance).isZero(Epsilon) ) break;
            Vector wo = cur.its.toWorld(wo_local), wi = cur.its.toWorld(cur.its.wi);
            if ( wi.dot(cur.its.geoFrame.n)*cur.its.wi.z() < Epsilon || wo.dot(cur.its.geoFrame.n)*wo_local.z() < Epsilon ) break;

            if ( !scene.rayIntersect(Ray(cur.its.p, wo), true, next.its)) break;
            scene.getPoint(next.its, cur.itsAD.p, next.itsAD, next.J);
            if (next.itsAD.t.val < ShadowEpsilon) break;            // Avoid numerical issue

            next.its = next.itsAD.toIntersection();
            cur.wo = -next.itsAD.toWorld(next.itsAD.wi);
            FloatAD G = next.itsAD.geoFrame.n.dot(-cur.wo).abs() / next.itsAD.t.square();
            pdf *= G.val;
            cur.G1 = std::abs(cur.its.geoFrame.n.dot(cur.wo.val))/ (next.its.t*next.its.t);
            next.throughput = cur.throughput * cur.itsAD.evalBSDF(cur.itsAD.toLocal(cur.wo), importance ? EBSDFMode::EImportanceWithCorrection : EBSDFMode::ERadiance) * G * next.J / pdf;
            // NOTE: IntersectionAD::evalBSDF returns rho((this->wi) -> (*this) -> wo) * cos(wo)
            next.opd = cur.opd + next.itsAD.opd;//next.itsAD.t;
            if (next.throughput.isZero(Epsilon)) break;
        }

        return depth + 1;
    }


    void preprocessPath(int pathLength, bool fix_first, PathNode *path) {
        assert(pathLength > 0);

        if ( fix_first ) {
            path[0].pdf1 = 0.0f;
            path[0].its1 = path[0].its;
        }
        for ( int i = fix_first ? 1 : 0; i < pathLength; ++i ) {
            bidir::PathNode &cur = path[i];
            if ( i + 2 < pathLength ) {
                const bidir::PathNode &next = path[i + 1];
                Intersection its = next.its;
                its.wi = its.toLocal(next.wo);
                cur.pdf1 = its.pdfBSDF(next.its.wi)*cur.G1;

                // Avoiding numerical issues
                if ( !std::isfinite(cur.pdf1) ) cur.pdf1 = 0.0f;
            } else
                cur.pdf1 = 0.0f;

            cur.its1 = cur.its;
        }

        for ( int i = 0; i < pathLength; ++i ) {
            if ( i == 0 ) {
                path[i].w = 0.0f;
            } else if ( i == 1 ) {
                path[i].w = fix_first ? 0.0f : 1.0f;
            } else {
                assert(i >= 2);
                path[i].w = path[i - 1].w*mis_ratio(path[i - 2].pdf1, path[i - 2].pdf0) + 1.0f;
            }
        }
    }


    void preprocessPathAD(int pathLength, bool fix_first, PathNodeAD *path) {
        assert(pathLength > 0);

        if ( fix_first ) {
            path[0].pdf1 = 0.0f;
            path[0].its1 = path[0].its;
        }
        for ( int i = fix_first ? 1 : 0; i < pathLength; ++i ) {
            bidir::PathNodeAD &cur = path[i];
            if ( i + 2 < pathLength ) {
                const bidir::PathNodeAD &next = path[i + 1];
                Intersection its = next.its;
                its.wi = its.toLocal(next.wo.val);
                cur.pdf1 = its.pdfBSDF(next.its.wi)*cur.G1;

                // Avoiding numerical issues
                if ( !std::isfinite(cur.pdf1) ) cur.pdf1 = 0.0f;
            } else
                cur.pdf1 = 0.0f;

            cur.its1 = cur.its;
        }

        for ( int i = 0; i < pathLength; ++i ) {
            if ( i == 0 ) {
                path[i].w = 0.0f;
            } else if ( i == 1 ) {
                path[i].w = fix_first ? 0.0f : 1.0f;
            } else {
                assert(i >= 2);
                path[i].w = path[i - 1].w * mis_ratio(path[i - 2].pdf1, path[i - 2].pdf0) + 1.0f;
            }
        }
    }


    Spectrum evalSegment(const Scene &scene, const Intersection &its0, const Intersection &its1, bool useEmission)
    {
        assert(!useEmission || its1.isEmitter());
        Spectrum ret(0.0f);
        if ( scene.isVisible(its0.p, true, its1.p, true) ) {
            Vector dir = its1.p - its0.p;
            Float distSqr = dir.squaredNorm();
            dir /= std::sqrt(distSqr);
            Spectrum val1 = useEmission ? Spectrum(its1.ptr_emitter->evalDirection(its1.geoFrame.n, -dir))
                                        : its1.evalBSDF(its1.toLocal(-dir), EBSDFMode::EImportanceWithCorrection);
            ret = its0.evalBSDF(its0.toLocal(dir))*val1/distSqr;
        }
        return ret;
    }


    SpectrumAD evalSegmentAD(const Scene &scene, const IntersectionAD &its0, const IntersectionAD &its1, bool useEmission)
    {
        assert(!useEmission || its1.isEmitter());
        SpectrumAD ret(Spectrum::Zero());
        if ( scene.isVisible(its0.p.val, true, its1.p.val, true) ) {
            VectorAD dir = its1.p - its0.p;
            FloatAD dist = dir.norm();
            dir /= dist;
            SpectrumAD val1 = useEmission ? its1.ptr_emitter->evalDirectionAD(its1.geoFrame.n, -dir)
                                          : its1.ptr_bsdf->evalAD(its1, its1.toLocal(-dir), EBSDFMode::EImportanceWithCorrection);
            ret = its0.ptr_bsdf->evalAD(its0, its0.toLocal(dir))*val1/(dist*dist);
        }
        return ret;
    }

    int radiance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces,
                  PathNode *camPath, PathNode *lightPath, Spectrum *ret, std::tuple<Spectrum, Float, int> *ret_trans)
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @input:
        /// BDPT path cam[0], ..., cam[max_bounces]|light[0]
        /// path legth (# of points) == max_bounces+1
        ///
        /// @output:
        /// For 0 <= i <= d:
        ///     ret[i] += radiance by paths from cameraPath[0] consisting of i segments and i+1 vertices
        ///               (WARN: without initialization by zero!!)
        /// For 0 <= i < len(ret_trans) (<= (mb+1)(mb+2)/2 - 1)
        ///     ret_trans[i] = (value, pathDist, depth) (with initialization)
        /// @return: len(ret_trans)
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        assert(its.isValid());

        for ( int i = 0; i <= max_bounces; ++i ) ret[i] = Spectrum(0.0f);
        if ( its.isEmitter() ) ret[0] = its.Le(its.toWorld(its.wi));

        // Building the camera sub-path
        camPath[0].its = its;
        camPath[0].throughput = Spectrum(1.0f);
        camPath[0].pdf0 = 1.0f;
        camPath[0].opd = 0.0;
        int camPathLen = 1;
        if ( max_bounces > 0 )
            camPathLen = buildPath(scene, sampler, max_bounces + 1, false, camPath);
        preprocessPath(camPathLen, true, camPath);

        // Building the light sub-path
        lightPath[0].throughput = scene.sampleEmitterPosition(sampler->next2D(), lightPath[0].its, &lightPath[0].pdf0);
        lightPath[0].opd = 0.0;
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
                lightPath[1].opd = lightPath[0].opd + lightPath[1].its.opd;

                if ( max_bounces > 1 )
                    lightPathLen = buildPath(scene, sampler, max_bounces - 1, true, &lightPath[1]) + 1;
                else
                    lightPathLen = 1;
            }
        }
        preprocessPath(lightPathLen, false, lightPath);

        return meta::evalPaths(scene, max_bounces, camPathLen, camPath, lightPathLen, lightPath, ret, ret_trans);
    }

    std::pair<int, int> weightedImportance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces, int pix_id,
                           PathNode* cameraPath, PathNode* lightPath, const Spectrum *weight, std::pair<int, Spectrum>* ret, std::tuple<int, Spectrum, Float, int>* ret_trans)
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// @input:
        ///
        /// @output:
        ///     camera - x[0], ..., x[max_bounces-1] - its
        /// @return:
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        assert(its.isValid());
        const Camera &camera = scene.camera;
        const CropRectangle& rect = camera.rect;

        Float inv_area;
        {
            Float fov_factor = camera.cam_to_ndc(0, 0);
            Float aspect_ratio = static_cast<Float>(camera.width)/camera.height;
            inv_area = 0.25f*fov_factor*fov_factor*aspect_ratio;
        }

        // Building the camera sub-path
        int x = rect.isValid() ? rect.offset_x + (pix_id % rect.crop_width) : pix_id % camera.width;
        int y = rect.isValid() ? rect.offset_y + static_cast<int>(pix_id / rect.crop_width) : static_cast<int>(pix_id / camera.width);
        ret[0].first = pix_id;
        ret[0].second = Spectrum(0.0f);
        Ray cameraRay = camera.samplePrimaryRay(x, y, sampler->next2D());
        int cameraPathLen;
        if ( scene.rayIntersect(cameraRay, false, cameraPath[0].its) ) {
            cameraPath[0].pdf0 = inv_area;
            cameraPath[0].pdf0 *= std::abs(cameraPath[0].its.geoFrame.n.dot(-cameraRay.dir))/
                                  (std::pow(camera.cframe.n.val.dot(cameraRay.dir), 3.0f)*cameraPath[0].its.t*cameraPath[0].its.t);
            cameraPath[0].throughput = Spectrum(static_cast<Float>(camera.getNumPixels()));
            cameraPath[0].opd = cameraPath[0].its.opd;
            cameraPathLen = 1;
            if (max_bounces > 0)
                cameraPathLen = bidir::buildPath(scene, sampler, max_bounces, false, cameraPath);
            preprocessPath(cameraPathLen, false, cameraPath);
        } else
            cameraPathLen = 0;

        // Buidling the light sub-path (from "its")
        lightPath[0].its = its;
        lightPath[0].throughput = Spectrum(1.0f);
        lightPath[0].pdf0 = 1.0f;
        lightPath[0].opd = 0.0;
        int lightPathLen = 1;
        if ( max_bounces > 0 )
            lightPathLen = bidir::buildPath(scene, sampler, max_bounces + 1, true, lightPath);
        preprocessPath(lightPathLen, true, lightPath);

        if (!ret_trans)
            return meta::evalPaths(scene, max_bounces, cameraPathLen, cameraPath, lightPathLen, lightPath, pix_id, weight, ret);
        else
            return meta::evalPaths(scene, max_bounces, cameraPathLen, cameraPath, lightPathLen, lightPath, pix_id, nullptr, ret, ret_trans);
    }

} //namespace bidir
