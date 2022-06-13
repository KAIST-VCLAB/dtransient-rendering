#pragma once
#ifndef INTERSECTION_AD_H__
#define INTERSECTION_AD_H__

#include "frameAD.h"
#include "intersection.h"

struct IntersectionAD {
    IntersectionAD() : ptr_shape(nullptr), ptr_med_int(nullptr), ptr_med_ext(nullptr), ptr_bsdf(nullptr), ptr_emitter(nullptr) {}

    IntersectionAD(const IntersectionAD &its)
        : ptr_shape(its.ptr_shape), ptr_med_int(its.ptr_med_int), ptr_med_ext(its.ptr_med_ext)
        , ptr_bsdf(its.ptr_bsdf), ptr_emitter(its.ptr_emitter)
        , t(its.t), p(its.p), geoFrame(its.geoFrame), shFrame(its.shFrame), uv(its.uv), wi(its.wi)
        , indices(its.indices), barycentric(its.barycentric) {}

    IntersectionAD(const Shape* ptr_shape, const Medium* ptr_med_int, const Medium* ptr_med_ext,
                   const BSDF* ptr_bsdf, const Emitter* ptr_emitter,
                   const FloatAD &t, const VectorAD &p, const FrameAD &geoFrame, const FrameAD &shFrame,
                   const Vector2AD &uv, const VectorAD &wi,
                   const Vector2i &indices, const Vector2AD &barycentric)
        : ptr_shape(ptr_shape), ptr_med_int(ptr_med_int), ptr_med_ext(ptr_med_ext)
        , ptr_bsdf(ptr_bsdf), ptr_emitter(ptr_emitter)
        , t(t), p(p), geoFrame(geoFrame), shFrame(shFrame), uv(uv), wi(wi)
        , indices(indices), barycentric(barycentric) {}
    IntersectionAD(const Shape* ptr_shape, const Medium* ptr_med_int, const Medium* ptr_med_ext,
                   const BSDF* ptr_bsdf, const Emitter* ptr_emitter,
                   const FloatAD &t, const FloatAD &opd, const VectorAD &p, const FrameAD &geoFrame, const FrameAD &shFrame,
                   const Vector2AD &uv, const VectorAD &wi,
                   const Vector2i &indices, const Vector2AD &barycentric)
            : ptr_shape(ptr_shape), ptr_med_int(ptr_med_int), ptr_med_ext(ptr_med_ext)
            , ptr_bsdf(ptr_bsdf), ptr_emitter(ptr_emitter)
            , t(t), opd(opd), p(p), geoFrame(geoFrame), shFrame(shFrame), uv(uv), wi(wi)
            , indices(indices), barycentric(barycentric) {}
    inline Intersection toIntersection() const {
        Intersection its(ptr_shape, ptr_med_int, ptr_med_ext, ptr_bsdf, ptr_emitter,
                         t.val, opd.val, p.val, geoFrame.toFrame(), shFrame.toFrame(), uv.val, wi.val,
                         indices, barycentric.val);
        return its;
    }

    // Pointers
    const Shape* ptr_shape;
    const Medium* ptr_med_int;
    const Medium* ptr_med_ext;
    const BSDF* ptr_bsdf;
    const Emitter* ptr_emitter;

    FloatAD t;          // Distance traveled along the ray
    FloatAD opd;         // Optical path distance traveled along the ray
    VectorAD p;         // Intersection point in 3D
    FrameAD geoFrame;   // Geometry Frame
    FrameAD shFrame;    // Shading Frame
    Vector2AD uv;       // uv surface coordinate
    VectorAD wi;        // Incident direction in local shading frame

    Vector2i indices;
    Vector2AD barycentric;

    inline bool isValid() const { return ptr_shape != nullptr; }
    inline bool isEmitter() const { return ptr_emitter != nullptr; }
    inline VectorAD toWorld(const VectorAD& v) const { return shFrame.toWorld(v); }
    inline VectorAD toLocal(const VectorAD& v) const { return shFrame.toLocal(v); }
    // Does the surface marked as a transition between two media
    inline bool isMediumTransition() const { return ptr_med_int!=nullptr || ptr_med_ext!=nullptr; }
    inline const Medium *getTargetMedium(const Vector &d) const { return d.dot(geoFrame.n.val)>0 ? ptr_med_ext : ptr_med_int;}
    inline const Medium *getTargetMedium(Float cosTheta) const { return cosTheta>0 ? ptr_med_ext : ptr_med_int;}
    inline const BSDF *getBSDF() const { return ptr_bsdf; }
    inline SpectrumAD Le(const VectorAD &wo) const { return ptr_emitter == nullptr ? SpectrumAD() : ptr_emitter->evalAD(*this, wo); }
    inline SpectrumAD evalBSDF(const VectorAD &wo, EBSDFMode mode = EBSDFMode::ERadiance) const { return ptr_bsdf->evalAD(*this, wo, mode); }
    inline SpectrumAD sampleBSDF(const Array& rnd, VectorAD& wo, Float &pdf, Float &eta, EBSDFMode mode = EBSDFMode::ERadiance) const { return ptr_bsdf->sampleAD(*this, rnd, wo, pdf, eta, mode); }
    inline Float pdfBSDF(const Vector& wo) const { return ptr_bsdf == nullptr ? 1.0 : ptr_bsdf->pdf(*this, wo); }

    inline FloatAD getIorDir(const VectorAD& dir) const {
        return ptr_bsdf->isTransmissive() && dir.dot(geoFrame.n).val>0 ? ptr_bsdf->ior() : 1.0f;
    }
    inline FloatAD getIorFrom(const VectorAD& pos) const {
        return getIorDir(p-pos);
    }
    inline FloatAD getOpdFrom(const VectorAD& pos) const {
        return getIorFrom(pos)*(p-pos).norm();
    }
};

#endif //INTERSECTION_AD_H__
