#pragma once
#ifndef INTERSECTION_H__
#define INTERSECTION_H__

#include "utils.h"
#include "frame.h"
#include "shape.h"
#include "medium.h"
#include "bsdf.h"
#include "emitter.h"
#include <assert.h>
#include <sstream>
#include <iostream>

struct Intersection {
    Intersection(): ptr_shape(nullptr), ptr_med_int(nullptr), ptr_med_ext(nullptr), ptr_bsdf(nullptr), ptr_emitter(nullptr) {}

    Intersection(const Intersection &its)
        : ptr_shape(its.ptr_shape), ptr_med_int(its.ptr_med_int), ptr_med_ext(its.ptr_med_ext)
        , ptr_bsdf(its.ptr_bsdf), ptr_emitter(its.ptr_emitter)
        , t(its.t), p(its.p), geoFrame(its.geoFrame), shFrame(its.shFrame), uv(its.uv), wi(its.wi)
        , indices(its.indices), barycentric(its.barycentric) {}

    Intersection(const Shape* ptr_shape, const Medium* ptr_med_int, const Medium* ptr_med_ext,
                 const BSDF* ptr_bsdf, const Emitter* ptr_emitter,
                 const Float &t, const Vector &p, const Frame &geoFrame, const Frame &shFrame,
                 const Vector2 &uv, const Vector &wi,
                 const Vector2i &indices, const Vector2 &barycentric)
        : ptr_shape(ptr_shape), ptr_med_int(ptr_med_int), ptr_med_ext(ptr_med_ext)
        , ptr_bsdf(ptr_bsdf), ptr_emitter(ptr_emitter)
        , t(t), p(p), geoFrame(geoFrame), shFrame(shFrame), uv(uv), wi(wi)
        , indices(indices), barycentric(barycentric) {}
    Intersection(const Shape* ptr_shape, const Medium* ptr_med_int, const Medium* ptr_med_ext,
                 const BSDF* ptr_bsdf, const Emitter* ptr_emitter,
                 const Float &t, const Float &opd, const Vector &p, const Frame &geoFrame, const Frame &shFrame,
                 const Vector2 &uv, const Vector &wi,
                 const Vector2i &indices, const Vector2 &barycentric)
            : ptr_shape(ptr_shape), ptr_med_int(ptr_med_int), ptr_med_ext(ptr_med_ext)
            , ptr_bsdf(ptr_bsdf), ptr_emitter(ptr_emitter)
            , t(t), opd(opd), p(p), geoFrame(geoFrame), shFrame(shFrame), uv(uv), wi(wi)
            , indices(indices), barycentric(barycentric) {}
    // Pointers
    const Shape* ptr_shape;
    const Medium* ptr_med_int;
    const Medium* ptr_med_ext;
    const BSDF* ptr_bsdf;
    const Emitter* ptr_emitter;

    Float t;        // Distance traveled along the ray
    Float opd;         // Optical path distance traveled along the ray
    Vector p;       // Intersection point in 3D
    Frame geoFrame; // Geometry Frame
    Frame shFrame;  // Shading Frame
    Vector2 uv;     // uv surface coordinate
    Vector wi;      // Incident direction in local shading frame

    Vector2i indices;
    Vector2 barycentric;

    inline bool isValid() const { return ptr_shape != nullptr; }
    inline bool isEmitter() const { return ptr_emitter != nullptr; }
    inline Vector toWorld(const Vector& v) const { return shFrame.toWorld(v); }
    inline Vector toLocal(const Vector& v) const { return shFrame.toLocal(v); }
    // Does the surface marked as a transition between two media
    inline bool isMediumTransition() const { return ptr_med_int!=nullptr || ptr_med_ext!=nullptr; }
    inline const Medium *getTargetMedium(const Vector &d) const { return d.dot(geoFrame.n)>0 ? ptr_med_ext : ptr_med_int;}
    inline const Medium *getTargetMedium(Float cosTheta) const { return cosTheta>0 ? ptr_med_ext : ptr_med_int;}
    inline const BSDF *getBSDF() const { return ptr_bsdf; }
    inline Spectrum Le(const Vector &wo) const { return ptr_emitter == nullptr ? Spectrum(0.0f) : ptr_emitter->eval(*this, wo); }
    inline Spectrum evalBSDF(const Vector &wo, EBSDFMode mode = EBSDFMode::ERadiance) const { return ptr_bsdf->eval(*this, wo, mode); }
    inline Spectrum sampleBSDF(const Array& rnd, Vector& wo, Float &pdf, Float &eta, EBSDFMode mode = EBSDFMode::ERadiance) const { return ptr_bsdf->sample(*this, rnd, wo, pdf, eta, mode); }
    inline Float pdfBSDF(const Vector& wo) const { return ptr_bsdf == nullptr ? 1.0 : ptr_bsdf->pdf(*this, wo); }
    inline Float getIorDir(const Vector& dir) const {
        return ptr_bsdf->isTransmissive() && dir.dot(geoFrame.n)>0 ? ptr_bsdf->ior().val : 1.0f;
    }
    inline Float getIorFrom(const Vector& pos) const {
        return getIorDir(p-pos);
    }
    inline Float getOpdFrom(const Vector& pos) const {
        return getIorFrom(pos)*(p-pos).norm();
    }
};

#endif
