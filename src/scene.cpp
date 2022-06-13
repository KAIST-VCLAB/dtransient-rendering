#include "scene.h"
#include "math_func.h"
#include "edge_manager/bruteforce.h"
#include "edge_manager/tree.h"
#include "edge_manager/pathspace.h"
#include <algorithm>
#include <assert.h>
#include <iostream>


Scene::Scene(const CameraTransient &camera, 
             const std::vector<const Shape*> &shapes,
             const std::vector<const BSDF*> &bsdfs,
             const std::vector<const Emitter*> &area_lights,
             const std::vector<const PhaseFunction*> &phases,
             const std::vector<const Medium*> &mediums,
             bool use_hierarchy)
        : camera(camera)
        , shape_list(shapes), bsdf_list(bsdfs), emitter_list(area_lights), phase_list(phases), medium_list(mediums)
        , ptr_edgeManager(nullptr), ptr_psEdgeManager(nullptr)
        , use_hierarchy(use_hierarchy)
{
    // Initialize Embree scene
    embree_device = rtcNewDevice(nullptr);
    embree_scene = rtcNewScene(embree_device);
    rtcSetSceneBuildQuality(embree_scene, RTC_BUILD_QUALITY_HIGH);
    rtcSetSceneFlags(embree_scene, RTC_SCENE_FLAG_ROBUST);
    // Copy the scene into Embree (since Embree requires 16 bytes alignment)
    for (const Shape *shape : shapes) {
        auto mesh = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        auto vertices = (Vector4f*)rtcSetNewGeometryBuffer(
            mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
            sizeof(Vector4f), shape->num_vertices);
        for (auto i = 0; i < shape->num_vertices; i++) {
            auto vertex = shape->getVertex(i);
            vertices[i] = Vector4f(vertex(0), vertex(1), vertex(2), 0.f);
        }
        auto triangles = (Vector3i*) rtcSetNewGeometryBuffer(
            mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
            sizeof(Vector3i), shape->num_triangles);
        for (auto i = 0; i < shape->num_triangles; i++) {
            triangles[i] = shape->getIndices(i);
        }
        rtcSetGeometryVertexAttributeCount(mesh, 1);
        rtcCommitGeometry(mesh);
        rtcAttachGeometry(embree_scene, mesh);
        rtcReleaseGeometry(mesh);
    }
    rtcCommitScene(embree_scene);

    num_lights = area_lights.size();
    light_distrb.clear();
    if (area_lights.size() > 0) {
        light_distrb.reserve(num_lights);
        for ( int i = 0; i < num_lights; ++i ) {
            const Shape &shape = *shapes[area_lights[i]->getShapeID()];
            light_distrb.append(shape.getArea()*area_lights[i]->getIntensity().maxCoeff());
        }
        light_distrb.normalize();
    }

    shape_distrb.clear();
    shape_distrb.reserve(shapes.size());
    for (int shape_id = 0; shape_id < (int)shapes.size(); shape_id++)
        shape_distrb.append(shapes[shape_id]->getArea());
    shape_distrb.normalize();

    initEdges(Eigen::Array<Float, -1, 1>::Ones(shapes.size(), 1));
    initPathSpaceEdges(Eigen::Array<Float, -1, 1>::Ones(shapes.size(), 1),
                       Vector3i(1, 1, 1), Eigen::Array<Float, -1, 1>::Ones(1, 1),
                       Vector3i(1, 1, 1), Eigen::Array<Float, -1, 1>::Ones(1, 1));
}

void Scene::initEdges(const Eigen::Array<Float, -1, 1> &shapeWeights) {
    assert(static_cast<size_t>(shapeWeights.rows()) == shape_list.size());
    if ( ptr_edgeManager ) delete ptr_edgeManager;
    if (use_hierarchy) {
        // printf("[INFO] Hierarchy edge manager\n");
        ptr_edgeManager = new TreeEdgeManager(*this, shapeWeights);
    }
    else {
        // printf("[INFO] BruteForce edge manager\n");
        ptr_edgeManager = new BruteForceEdgeManager(*this, shapeWeights);
    }
}

void Scene::initPathSpaceEdges(const Eigen::Array<Float, -1, 1> &shapeWeights,
                               const Vector3i &direct_dims, const Eigen::Array<Float, -1, 1> &direct_data,
                               const Vector3i &indirect_dims, const Eigen::Array<Float, -1, 1> &indirect_data, bool verbose) {
    assert(static_cast<size_t>(shapeWeights.rows()) == shape_list.size());
    if ( ptr_psEdgeManager ) delete ptr_psEdgeManager;
    ptr_psEdgeManager = new PathSpaceEdgeManager(*this, shapeWeights, direct_dims, direct_data, indirect_dims, indirect_data, verbose);
}

Scene::~Scene() {
    rtcReleaseScene(embree_scene);
    rtcReleaseDevice(embree_device);
    if ( ptr_edgeManager ) delete ptr_edgeManager;
    if ( ptr_psEdgeManager ) delete ptr_psEdgeManager;
}

bool Scene::isVisible(const Vector &p, bool pOnSurface, const Vector &q, bool qOnSurface) const {
    Vector dir = q - p;
    Float dist = dir.norm();
    dir /= dist;

    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = p.x();
    rtc_ray_hit.ray.org_y = p.y();
    rtc_ray_hit.ray.org_z = p.z();
    rtc_ray_hit.ray.dir_x = dir.x();
    rtc_ray_hit.ray.dir_y = dir.y();
    rtc_ray_hit.ray.dir_z = dir.z();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.ray.tnear = pOnSurface ? ShadowEpsilon : 0.0f;
    rtc_ray_hit.ray.tfar = qOnSurface ? (1.0f - ShadowEpsilon)*dist : dist;
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);

    return rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID;
}

Float Scene::evalTransmittance(const Ray& _ray, bool onSurface, const Medium* ptr_medium, Float remaining, RndSampler* sampler, int max_interactions) const {
    Ray ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    Float transmittance = 1.0f;
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.x();
    rtc_ray_hit.ray.org_y = ray.org.y();
    rtc_ray_hit.ray.org_z = ray.org.z();
    rtc_ray_hit.ray.dir_x = ray.dir.x();
    rtc_ray_hit.ray.dir_y = ray.dir.y();
    rtc_ray_hit.ray.dir_z = ray.dir.z();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = (1.0f - ShadowEpsilon) * remaining;
    const Shape* ptr_shape = nullptr;
    int interactions = 0;
    Intersection new_its;
    while (remaining > 0) {
        Float tmax = remaining;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
        if (rtc_ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
            if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull()) {
                return 0.0f;
            }
            ptr_shape->rayIntersect((int)rtc_ray_hit.hit.primID, ray, new_its);
            if ( new_its.t < tmax ) tmax = new_its.t;
        }
        if (ptr_medium != nullptr) {
            transmittance *= ptr_medium->evalTransmittance(ray, tmin, tmax, sampler);
        }
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            break;
        // If null surface, consider transmittance
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.dot(ray.dir)>0.0
                          ? ptr_shape->med_ext_id : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org = ray(tmax);
        rtc_ray_hit.ray.org_x = ray.org.x();
        rtc_ray_hit.ray.org_y = ray.org.y();
        rtc_ray_hit.ray.org_z = ray.org.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        remaining -= tmax;
        rtc_ray_hit.ray.tfar = tmax = (1 - ShadowEpsilon)*remaining;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
    }
    return transmittance;
}

FloatAD Scene::evalTransmittanceAD(const RayAD& _ray, bool onSurface, const Medium* ptr_medium, FloatAD remaining, RndSampler* sampler, int max_interactions) const {
    RayAD ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    FloatAD transmittance(1.0f);
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.val.x();
    rtc_ray_hit.ray.org_y = ray.org.val.y();
    rtc_ray_hit.ray.org_z = ray.org.val.z();
    rtc_ray_hit.ray.dir_x = ray.dir.val.x();
    rtc_ray_hit.ray.dir_y = ray.dir.val.y();
    rtc_ray_hit.ray.dir_z = ray.dir.val.z();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = (1.0f - ShadowEpsilon) * remaining.val;
    const Shape* ptr_shape = nullptr;
    int interactions = 0;
    IntersectionAD new_its;
    while (remaining > 0) {
        FloatAD tmax = remaining;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
        if (rtc_ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
            if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull()) {
                return FloatAD();
            }
            ptr_shape->rayIntersectAD((int)rtc_ray_hit.hit.primID, ray, new_its);
            if ( new_its.t < tmax ) tmax = new_its.t;
        }
        if (ptr_medium != nullptr) {
            transmittance *= ptr_medium->evalTransmittanceAD(ray, tmin, tmax, sampler);
        }
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            break;
        // If null surface, consider transmittance
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.dot(ray.dir)>0.0
                          ? ptr_shape->med_ext_id : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org = ray(tmax);
        rtc_ray_hit.ray.org_x = ray.org.val.x();
        rtc_ray_hit.ray.org_y = ray.org.val.y();
        rtc_ray_hit.ray.org_z = ray.org.val.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        remaining -= tmax;
        tmax = (1.0f - ShadowEpsilon)*remaining;
        rtc_ray_hit.ray.tfar = tmax.val;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
    }
    return transmittance;
}

Spectrum Scene::sampleEmitterDirect(const Intersection &its, const Vector2 &_rnd_light, RndSampler* sampler, Vector& wo, Float &pdf, Vector *pos_emitter) const {
    Vector2 rnd_light(_rnd_light);
    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();
    PositionSamplingRecord pRec;
    shape_list[shape_id]->samplePosition(rnd_light, pRec);
    pdf /= shape_list[shape_id]->getArea();
    const Vector &light_pos = pRec.p, &light_norm = pRec.n;
    Vector dir = light_pos - its.p;

    if ( (its.ptr_bsdf->isTransmissive() || its.ptr_bsdf->isTwosided() || (dir.dot(its.geoFrame.n) > Epsilon && dir.dot(its.shFrame.n) > Epsilon)) &&
         dir.dot(light_norm) < -Epsilon && pdf > Epsilon )
    {
        Float dist = dir.norm();
        dir = dir/dist;
        RTCIntersectContext rtc_context;
        rtcInitIntersectContext(&rtc_context);
        RTCRayHit rtc_ray_hit;
        rtc_ray_hit.ray.org_x = its.p.x();
        rtc_ray_hit.ray.org_y = its.p.y();
        rtc_ray_hit.ray.org_z = its.p.z();
        rtc_ray_hit.ray.dir_x = dir.x();
        rtc_ray_hit.ray.dir_y = dir.y();
        rtc_ray_hit.ray.dir_z = dir.z();
        rtc_ray_hit.ray.mask = (unsigned int)(-1);
        rtc_ray_hit.ray.time = 0.f;
        rtc_ray_hit.ray.flags = 0;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.ray.tnear = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = (1 - ShadowEpsilon) * dist;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);

        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            pdf *= dist*dist / light_norm.dot(-dir);
            wo = its.toLocal(dir);
            if (pos_emitter)
                *pos_emitter = light_pos;
            return emitter_list[light_id]->eval(light_norm, -dir)/pdf;
        }
    }
    return Spectrum::Zero();
}



SpectrumAD Scene::sampleEmitterDirectAD(const IntersectionAD &its, const Vector2 &_rnd_light, RndSampler* sampler, VectorAD& wo, Float &pdf, Float *psInfo) const {
    Vector2 rnd_light(_rnd_light);
    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();
    PositionSamplingRecord pRec;
    shape_list[shape_id]->samplePosition(rnd_light, pRec);
    pdf /= shape_list[shape_id]->getArea();
    const Vector &light_pos = pRec.p, &light_norm = pRec.n;

    Vector dir = light_pos - its.p.val;
    if ( (its.ptr_bsdf->isTransmissive() || its.ptr_bsdf->isTwosided() || (dir.dot(its.geoFrame.n.val) > Epsilon && dir.dot(its.shFrame.n.val) > Epsilon)) &&
         dir.dot(light_norm) < -Epsilon && pdf > Epsilon )
    {
        Float dist = dir.norm();
        dir = dir/dist;
        RTCIntersectContext rtc_context;
        rtcInitIntersectContext(&rtc_context);
        RTCRayHit rtc_ray_hit;
        rtc_ray_hit.ray.org_x = its.p.val.x();
        rtc_ray_hit.ray.org_y = its.p.val.y();
        rtc_ray_hit.ray.org_z = its.p.val.z();
        rtc_ray_hit.ray.dir_x = dir.x();
        rtc_ray_hit.ray.dir_y = dir.y();
        rtc_ray_hit.ray.dir_z = dir.z();
        rtc_ray_hit.ray.mask = (unsigned int)(-1);
        rtc_ray_hit.ray.time = 0.f;
        rtc_ray_hit.ray.flags = 0;
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.ray.tnear = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);

        // Only works for area light
        const Emitter &light = *emitter_list[light_id];
        assert(light.shape_id >= 0);
        if (static_cast<int>(rtc_ray_hit.hit.geomID) == light.shape_id) {
            const Shape *shape = shape_list[light.shape_id];
            if ( psInfo == NULL ) {
                IntersectionAD new_its;
                shape->rayIntersectAD(rtc_ray_hit.hit.primID, RayAD(its.p, dir), new_its);
                if ( std::abs(new_its.t.val - dist) < ShadowEpsilon ) {
                    pdf *= dist*dist / light_norm.dot(-dir);
                    wo = its.toLocal(dir);
                    return light.evalAD(new_its.geoFrame.n, VectorAD(-dir))/pdf;
                }
            } else {
                Intersection new_its;
                shape->rayIntersect(rtc_ray_hit.hit.primID, Ray(its.p.val, dir), new_its);
                if ( std::abs(new_its.t - dist) < ShadowEpsilon ) {
                    new_its.indices = Vector2i(light.shape_id, rtc_ray_hit.hit.primID);
                    VectorAD x2, n2;
                    FloatAD J;
                    getPoint(new_its, x2, n2, J);
                    if ( (x2.val - light_pos).norm() > ShadowEpsilon ) {
                        std::cerr << "[WARN] invalid emitter sample: [" << x2.val.transpose() << "] != [" << light_pos.transpose() << "]" << std::endl;
                        return SpectrumAD();
                    }

                    VectorAD _dir = x2 - its.p;
                    FloatAD _distSqr = _dir.squaredNorm();
                    _dir /= _distSqr.sqrt();
                    FloatAD G = n2.dot(-_dir)/_distSqr;

                    wo = its.toLocal(_dir);
                    *psInfo = G.val;
                    return light.evalAD(n2, -_dir)*G*J/pdf;
                }
            }
        }
    }
    return SpectrumAD();
}

Spectrum Scene::sampleAttenuatedEmitterDirect(const Vector &pscatter, const Vector2 &_rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                              Vector& wo, Float& pdf) const {
    Vector2 rnd_light(_rnd_light);
    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();
    PositionSamplingRecord pRec;
    shape_list[shape_id]->samplePosition(rnd_light, pRec);
    pdf /= shape_list[shape_id]->getArea();
    const Vector &light_pos = pRec.p, &light_norm = pRec.n;

    Vector dir = light_pos - pscatter;
    if (dir.dot(light_norm)<0 && pdf != 0) {
        Float dist = dir.norm();
        dir = dir/dist;
        Ray shadow_ray(pscatter, dir);
        Float transmittance = evalTransmittance(shadow_ray, 0.0, ptr_medium, dist, sampler, max_interactions);
        if (transmittance != 0) {
            pdf *= dist*dist / light_norm.dot(-dir);
            wo = dir;
            return transmittance * emitter_list[light_id]->eval(light_norm, -dir)/pdf;
        }
    }
    return Spectrum::Zero();
}

SpectrumAD Scene::sampleAttenuatedEmitterDirectAD(const VectorAD &pscatter, const Vector2 &_rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                                  VectorAD& wo, Float& pdf) const {
    Vector2 rnd_light(_rnd_light);
    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();
    PositionSamplingRecord pRec;
    shape_list[shape_id]->samplePosition(rnd_light, pRec);
    pdf /= shape_list[shape_id]->getArea();
    const Vector &light_pos = pRec.p, &light_norm = pRec.n;

    Vector dir = light_pos - pscatter.val;
    if (dir.dot(light_norm)<0) {
        Float dist = dir.norm();
        dir /= dist;

        IntersectionAD itsNear, itsFar;
        SpectrumAD ret = rayIntersectAndLookForEmitterAD(RayAD(pscatter, dir), false, sampler, ptr_medium, max_interactions, itsNear, pdf, &itsFar);
        if ( !ret.isZero(Epsilon) && pdf > Epsilon && itsFar.ptr_shape == shape_list[shape_id] && std::abs(itsFar.t.val - dist) < ShadowEpsilon ) {
            wo = dir;
            return ret/pdf;
        }
    }
    return SpectrumAD();
}

Spectrum Scene::sampleAttenuatedEmitterDirect(const Intersection& its, const Vector2 &_rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                              Vector& wo, Float& pdf, bool flag) const {
    Vector2 rnd_light(_rnd_light);
    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();
    PositionSamplingRecord pRec;
    shape_list[shape_id]->samplePosition(rnd_light, pRec);
    pdf /= shape_list[shape_id]->getArea();
    const Vector &light_pos = pRec.p, &light_norm = pRec.n;

    Vector dir = light_pos - its.p;
    if ( (its.ptr_bsdf->isTransmissive() || its.ptr_bsdf->isTwosided() || (dir.dot(its.geoFrame.n) > Epsilon && dir.dot(its.shFrame.n) > Epsilon)) &&
         dir.dot(light_norm) < -Epsilon && pdf > Epsilon )
    {
        if (its.isMediumTransition())
            ptr_medium = its.getTargetMedium(dir);

        Float dist = dir.norm();
        dir = dir/dist;
        Ray shadow_ray(its.p, dir);
        wo = its.toLocal(dir);
        if ( flag && math::signum(its.wi.z()) != math::signum(wo.z()) ) --max_interactions;
        if ( max_interactions >= 0 ) {
            Float transmittance = evalTransmittance(shadow_ray, ShadowEpsilon, ptr_medium, dist, sampler, max_interactions);
            if (transmittance != 0) {
                pdf *= dist*dist / light_norm.dot(-dir);
                return transmittance * emitter_list[light_id]->eval(light_norm, -dir)/pdf;
            }
        }
    }
    return Spectrum::Zero();
}

SpectrumAD Scene::sampleAttenuatedEmitterDirectAD(const IntersectionAD& its, const Vector2 &_rnd_light, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                                  VectorAD& wo, Float& pdf) const {
    Vector2 rnd_light(_rnd_light);
    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();
    PositionSamplingRecord pRec;
    shape_list[shape_id]->samplePosition(rnd_light, pRec);
    pdf /= shape_list[shape_id]->getArea();
    const Vector &light_pos = pRec.p, &light_norm = pRec.n;

    Vector dir = light_pos - its.p.val;
    if ( (its.ptr_bsdf->isTransmissive() || its.ptr_bsdf->isTwosided() || (dir.dot(its.geoFrame.n.val) > Epsilon && dir.dot(its.shFrame.n.val) > Epsilon)) &&
         dir.dot(light_norm) < -Epsilon && pdf > Epsilon )
    {
        if (its.isMediumTransition())
            ptr_medium = its.getTargetMedium(dir);

        Float dist = dir.norm();
        dir /= dist;

        IntersectionAD itsNear, itsFar;
        SpectrumAD ret = rayIntersectAndLookForEmitterAD(RayAD(its.p, dir), true, sampler, ptr_medium, max_interactions, itsNear, pdf, &itsFar);
        if ( !ret.isZero(Epsilon) && pdf > Epsilon && itsFar.ptr_shape == shape_list[shape_id] && std::abs(itsFar.t.val - dist) < ShadowEpsilon ) {
            wo = its.toLocal(dir);
            return ret/pdf;
        }
    }
    return SpectrumAD();
}

Float Scene::pdfEmitterSample(const Intersection& its) const {
    int light_id = its.ptr_shape->light_id;
    assert(light_id >= 0);
    return light_distrb[light_id]/its.ptr_shape->getArea();
}

Float Scene::pdfEmitterSample(const IntersectionAD& its) const {
    int light_id = its.ptr_shape->light_id;
    assert(light_id >= 0);
    return light_distrb[light_id]/its.ptr_shape->getArea();
}

bool Scene::rayIntersect(const Ray &ray, bool onSurface, Intersection& its, Float* p_opt_path_dist) const {
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;
    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.x();
    rtc_ray_hit.ray.org_y = ray.org.y();
    rtc_ray_hit.ray.org_z = ray.org.z();
    rtc_ray_hit.ray.dir_x = ray.dir.x();
    rtc_ray_hit.ray.dir_y = ray.dir.y();
    rtc_ray_hit.ray.dir_z = ray.dir.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        its.opd = std::numeric_limits<Float>::infinity();
        its.ptr_shape = nullptr;
        return false;
    } else {
        // Fill in the corresponding pointers
        its.indices[0] = static_cast<int>(rtc_ray_hit.hit.geomID);
        its.ptr_shape = shape_list[its.indices[0]];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        // Ray-Shape intersection
        its.indices[1] = static_cast<int>(rtc_ray_hit.hit.primID);
        its.ptr_shape->rayIntersect(its.indices[1], ray, its);
        if (p_opt_path_dist)
            *p_opt_path_dist = (ray.org - its.p).norm();
        return true;
    }
}

bool Scene::rayIntersectAD(const RayAD &ray, bool onSurface, IntersectionAD& its, FloatAD* p_opt_path_dist) const {
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    RTCIntersectContext rtc_context;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.val.x();
    rtc_ray_hit.ray.org_y = ray.org.val.y();
    rtc_ray_hit.ray.org_z = ray.org.val.z();
    rtc_ray_hit.ray.dir_x = ray.dir.val.x();
    rtc_ray_hit.ray.dir_y = ray.dir.val.y();
    rtc_ray_hit.ray.dir_z = ray.dir.val.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        its.ptr_shape = nullptr;
        return false;
    } else {
        // Fill in the corresponding pointers
        its.indices[0] = static_cast<int>(rtc_ray_hit.hit.geomID);
        its.ptr_shape = shape_list[its.indices[0]];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        // Ray-Shape intersection
        its.indices[1] = static_cast<int>(rtc_ray_hit.hit.primID);
        its.ptr_shape->rayIntersectAD(its.indices[1], ray, its);
        if (p_opt_path_dist)
            *p_opt_path_dist = (ray.org - its.p).norm();
        return true;
    }
}
Spectrum Scene::rayIntersectAndLookForEmitter(const Ray &_ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                              Intersection &its, Float& pdf_nee) const {
    Ray ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    int interactions = 0;
    RTCIntersectContext rtc_context;
    Vector scattering_pos = ray.org;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.x();
    rtc_ray_hit.ray.org_y = ray.org.y();
    rtc_ray_hit.ray.org_z = ray.org.z();
    rtc_ray_hit.ray.dir_x = ray.dir.x();
    rtc_ray_hit.ray.dir_y = ray.dir.y();
    rtc_ray_hit.ray.dir_z = ray.dir.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    // Apply first intersection and store the intersection record to its
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    const Shape* ptr_shape = nullptr;
    its.ptr_shape = nullptr;
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        return Spectrum::Zero();
    } else {
        its.ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        int tri_id = (int)rtc_ray_hit.hit.primID;
        its.ptr_shape->rayIntersect(tri_id, ray, its);
    }

    Float transmittance = 1.0;
    Intersection new_its;
    while (true) {
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            return Spectrum::Zero();

        ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        ptr_shape->rayIntersect((int)rtc_ray_hit.hit.primID, ray, new_its);
        Float tmax = new_its.t;

        if (ptr_medium)
            transmittance *= ptr_medium->evalTransmittance(ray, tmin, tmax, sampler);
        // check if hit emitter
        if (ptr_shape->isEmitter()) {
            auto dist_sq = (new_its.p - scattering_pos).squaredNorm();
            auto geometry_term = new_its.wi.z()/ dist_sq;
            int light_id = ptr_shape->light_id;
            pdf_nee = light_distrb[light_id]/(ptr_shape->getArea() * geometry_term);
            return transmittance * emitter_list[light_id]->eval(new_its, -ray.dir);
        }
        // check if hit a surface (not null surface) or emitter
        if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull())
            return Spectrum::Zero();

        // If null surface, keep tracing
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.dot(ray.dir)>0.0 ? ptr_shape->med_ext_id
                                                              : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org += ray.dir*tmax;
        rtc_ray_hit.ray.org_x = ray.org.x();
        rtc_ray_hit.ray.org_y = ray.org.y();
        rtc_ray_hit.ray.org_z = ray.org.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = tmax = std::numeric_limits<Float>::infinity();
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    }
}

SpectrumAD Scene::rayIntersectAndLookForEmitterAD(const RayAD &_ray, bool onSurface, RndSampler* sampler, const Medium* ptr_medium, int max_interactions,
                                                  IntersectionAD &its, Float& pdf_nee, IntersectionAD *itsFar) const
{
    RayAD ray(_ray);
    Float tmin = onSurface ? ShadowEpsilon : 0.0f;

    int interactions = 0;
    RTCIntersectContext rtc_context;
    Vector scattering_pos = ray.org.val;
    rtcInitIntersectContext(&rtc_context);
    RTCRayHit rtc_ray_hit;
    rtc_ray_hit.ray.org_x = ray.org.val.x();
    rtc_ray_hit.ray.org_y = ray.org.val.y();
    rtc_ray_hit.ray.org_z = ray.org.val.z();
    rtc_ray_hit.ray.dir_x = ray.dir.val.x();
    rtc_ray_hit.ray.dir_y = ray.dir.val.y();
    rtc_ray_hit.ray.dir_z = ray.dir.val.z();
    rtc_ray_hit.ray.tnear = tmin;
    rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
    rtc_ray_hit.ray.mask = (unsigned int)(-1);
    rtc_ray_hit.ray.time = 0.f;
    rtc_ray_hit.ray.flags = 0;
    rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    // Apply first intersection and store the intersection record to its
    rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    const Shape* ptr_shape = nullptr;
    its.ptr_shape = nullptr;
    if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        its.t = std::numeric_limits<Float>::infinity();
        return SpectrumAD();
    } else {
        its.ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        its.ptr_med_int = (its.ptr_shape->med_int_id >= 0) ? medium_list[its.ptr_shape->med_int_id] : nullptr;
        its.ptr_med_ext = (its.ptr_shape->med_ext_id >= 0) ? medium_list[its.ptr_shape->med_ext_id] : nullptr;
        its.ptr_emitter = (its.ptr_shape->light_id>= 0) ? emitter_list[its.ptr_shape->light_id] : nullptr;
        assert(its.ptr_shape->bsdf_id >= 0);
        its.ptr_bsdf = bsdf_list[its.ptr_shape->bsdf_id];
        int tri_id = (int)rtc_ray_hit.hit.primID;
        its.ptr_shape->rayIntersectAD(tri_id, ray, its);
    }

    FloatAD transmittance(1.0), dist;
    IntersectionAD new_its;
    while (true) {
        if (rtc_ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            return SpectrumAD();

        ptr_shape = shape_list[(int)rtc_ray_hit.hit.geomID];
        ptr_shape->rayIntersectAD((int)rtc_ray_hit.hit.primID, ray, new_its);
        FloatAD tmax = new_its.t;
        dist += new_its.t;

        if (ptr_medium)
            transmittance *= ptr_medium->evalTransmittanceAD(ray, tmin, tmax, sampler);
        // check if hit emitter
        if (ptr_shape->isEmitter()) {
            auto dist_sq = (new_its.p.val - scattering_pos).squaredNorm();
            auto geometry_term = new_its.wi.val.z()/dist_sq;
            int light_id = ptr_shape->light_id;
            pdf_nee = light_distrb[light_id]/(ptr_shape->getArea() * geometry_term);
            if ( itsFar != nullptr ) {
                *itsFar = new_its; itsFar->ptr_shape = ptr_shape; itsFar->t = dist;
            }
            return transmittance * emitter_list[light_id]->evalAD(new_its, -ray.dir);
        }
        // check if hit a surface (not null surface) or emitter
        if (interactions == max_interactions || !bsdf_list[ptr_shape->bsdf_id]->isNull())
            return SpectrumAD();

        // If null surface, keep tracing
        if (ptr_shape->isMediumTransition()) {
            int med_id  = new_its.geoFrame.n.val.dot(ray.dir.val) > 0.0 ? ptr_shape->med_ext_id
                                                                        : ptr_shape->med_int_id;
            ptr_medium = med_id != -1 ? medium_list[med_id] : nullptr;
        }
        ray.org += ray.dir*tmax;
        tmax = std::numeric_limits<Float>::infinity();
        rtc_ray_hit.ray.org_x = ray.org.val.x();
        rtc_ray_hit.ray.org_y = ray.org.val.y();
        rtc_ray_hit.ray.org_z = ray.org.val.z();
        rtc_ray_hit.ray.tnear = tmin = ShadowEpsilon;
        rtc_ray_hit.ray.tfar = std::numeric_limits<Float>::infinity();
        rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        interactions++;
        rtcIntersect1(embree_scene, &rtc_context, &rtc_ray_hit);
    }
}

Spectrum Scene::sampleEmitterPosition(const Vector2 &_rnd_light, Intersection& its, Float *_pdf) const {
    Vector2 rnd_light(_rnd_light);
    Float pdf;

    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();
    PositionSamplingRecord pRec;
    its.indices[0] = shape_id;
    its.indices[1] = shape_list[shape_id]->samplePosition(rnd_light, pRec);
    its.barycentric = pRec.uv;
    pdf /= shape_list[shape_id]->getArea();

    its.ptr_emitter = emitter_list[light_id];
    its.ptr_shape = shape_list[its.ptr_emitter->getShapeID()];


    int med_id = its.ptr_shape->med_ext_id;
    its.ptr_med_ext =  med_id >= 0 ? medium_list[med_id] : nullptr;
    med_id = its.ptr_shape->med_int_id;
    its.ptr_med_int = med_id >= 0 ? medium_list[med_id] : nullptr;

    its.p = pRec.p;
    its.wi = Vector::Zero();
    its.t = 0.0f;
    its.geoFrame = its.shFrame = Frame(pRec.n);
    if ( _pdf ) *_pdf = pdf;
    return its.ptr_emitter->getIntensity()/pdf;
}


SpectrumAD Scene::sampleEmitterPosition(const Vector2 &_rnd_light, IntersectionAD& its, FloatAD& J, Float *_pdf) const {
    Vector2 rnd_light(_rnd_light);
    Float pdf;
    const int light_id = light_distrb.sampleReuse(rnd_light[0], pdf);
    const int shape_id = emitter_list[light_id]->getShapeID();

    its.indices[0] = shape_id;
    its.ptr_emitter = emitter_list[light_id];
    its.ptr_shape = shape_list[shape_id];
    int med_id = its.ptr_shape->med_ext_id;
    its.ptr_med_ext = med_id >= 0 ? medium_list[med_id] : nullptr;
    med_id = its.ptr_shape->med_int_id;
    its.ptr_med_int = med_id >= 0 ? medium_list[med_id] : nullptr;

    PositionSamplingRecordAD pRec;
    its.indices[1] = shape_list[shape_id]->samplePositionAD(rnd_light, pRec);
    pdf /= shape_list[shape_id]->getArea();

    its.p = pRec.p;
    its.wi = VectorAD(Vector::Zero());
    J = pRec.J;
    its.t = 0.0f;
    its.geoFrame = its.shFrame = FrameAD(pRec.n);

    if (_pdf) *_pdf = pdf;

    return SpectrumAD(its.ptr_emitter->getIntensity())/pdf;
}


Float Scene::sampleAttenuatedSensorDirect(const Intersection& its, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const {
    Float value = camera.sampleDirect(its.p, pixel_uv, dir);
    if (value != 0.0f) {
        const Medium* ptr_medium = its.getTargetMedium(dir);
        Float dist = (its.p-camera.cpos.val).norm();
        value *= evalTransmittance(Ray(its.p, dir), true, ptr_medium, dist, sampler, max_interactions);
        return value;
    } else {
        return 0.0f;
    }
}

Float Scene::sampleAttenuatedSensorDirect(const Vector& p, const Medium* ptr_med, RndSampler* sampler, int max_interactions, Vector2& pixel_uv, Vector& dir) const {
    Float value = camera.sampleDirect(p, pixel_uv, dir);
    if (value != 0.0f) {
        Float dist = (p-camera.cpos.val).norm();
        value *= evalTransmittance(Ray(p, dir), false, ptr_med, dist, sampler, max_interactions);
        return value;
    } else {
        return 0.0f;
    }
}

Vector2i Scene::samplePosition(const Vector2 &_rnd2, PositionSamplingRecord &pRec) const {
    Vector2i ret;
    Vector2 rnd2(_rnd2);
    ret[0] = static_cast<int>(shape_distrb.sampleReuse(rnd2[0]));
    ret[1] = shape_list[ret[0]]->samplePosition(rnd2, pRec);
    return ret;
}

Vector2i Scene::samplePositionAD(const Vector2 &_rnd2, PositionSamplingRecordAD &pRec) const {
    Vector2i ret;
    Vector2 rnd2(_rnd2);
    ret[0] = static_cast<int>(shape_distrb.sampleReuse(rnd2[0]));
    ret[1] = shape_list[ret[0]]->samplePositionAD(rnd2, pRec);
    return ret;
}

void Scene::getPoint(const Intersection &its, VectorAD &x, VectorAD &n, FloatAD &J) const {
    assert(its.indices[0] >= 0 && its.indices[0] < static_cast<int>(shape_list.size()));
    const Shape &shape = *shape_list[its.indices[0]];
    shape.getPoint(its.indices[1], Vector2AD(its.barycentric), x, n, J);
}

void Scene::getPoint(const Intersection &its, const VectorAD &p, IntersectionAD& its_AD, FloatAD &J) const {
   assert(its.indices[0] >= 0 && its.indices[0] < static_cast<int>(shape_list.size()));
   const Shape &shape = *shape_list[its.indices[0]];
   its_AD.ptr_shape     = its.ptr_shape;
   its_AD.ptr_med_int   = its.ptr_med_int;
   its_AD.ptr_med_ext   = its.ptr_med_ext;
   its_AD.ptr_bsdf      = its.ptr_bsdf;
   its_AD.ptr_emitter   = its.ptr_emitter;
   its_AD.indices       = its.indices;
   its_AD.barycentric   = Vector2AD(its.barycentric);
   shape.getPoint(its.indices[1], Vector2AD(its.barycentric), its_AD, J);

   VectorAD dir = its_AD.p - p;
   its_AD.t = dir.norm();
   its_AD.opd = its_AD.getIorDir(dir) * its_AD.t;
   dir /= its_AD.t;
   its_AD.wi = its_AD.toLocal(-dir);
}

void Scene::getPoint(const Intersection &its, IntersectionAD& its_AD, FloatAD &J) const {
   assert(its.indices[0] >= 0 && its.indices[0] < static_cast<int>(shape_list.size()));
   const Shape &shape = *shape_list[its.indices[0]];
   its_AD.ptr_shape     = its.ptr_shape;
   its_AD.ptr_med_int   = its.ptr_med_int;
   its_AD.ptr_med_ext   = its.ptr_med_ext;
   its_AD.ptr_bsdf      = its.ptr_bsdf;
   its_AD.ptr_emitter   = its.ptr_emitter;
   its_AD.indices       = its.indices;
   its_AD.barycentric   = Vector2AD(its.barycentric);
   its_AD.t             = FloatAD(its.t);
   its_AD.opd = FloatAD(its.opd);
   its_AD.wi            = VectorAD(its.wi);
   shape.getPoint(its.indices[1], Vector2AD(its.barycentric), its_AD, J);
}


void Scene::getPointAD(const IntersectionAD &its, VectorAD &x, VectorAD &n, FloatAD &J) const {
    assert(its.indices[0] >= 0 && its.indices[0] < static_cast<int>(shape_list.size()));
    const Shape &shape = *shape_list[its.indices[0]];
    shape.getPoint(its.indices[1], its.barycentric, x, n, J);
}

const Edge& Scene::sampleEdgeRay(const Vector &_rnd, int &shape_id, RayAD &ray, Float &pdf) const {
    assert(ptr_psEdgeManager);
    return ptr_psEdgeManager->sampleEdgeRay(_rnd, shape_id, ray, pdf, false);
}

const Edge& Scene::sampleEdgeRayDirect(const Vector &_rnd, int &shape_id, RayAD &ray, Float &pdf) const {
    assert(ptr_psEdgeManager);
    return ptr_psEdgeManager->sampleEdgeRay(_rnd, shape_id, ray, pdf, true);
}
