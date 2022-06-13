#pragma once
#ifndef BSDF_ROUGH_DIELECTRIC_H__
#define BSDF_ROUGH_DIELECTRIC_H__

#include "bsdf.h"
#include "ptr.h"
#include "microfacet.h"

struct RoughDielectricBSDF: BSDF
{
    inline RoughDielectricBSDF(Float alpha, Float intIOR, Float extIOR) : m_distr(alpha) {
        m_eta = intIOR/extIOR;
        m_invEta = 1.0f/m_eta;
        m_spectrum = SpectrumAD(1.0f);
    }

    inline RoughDielectricBSDF(Float alpha, Float intIOR, Float extIOR, const Spectrum3f& spectrum)
        : m_distr(alpha), m_spectrum(spectrum.cast<Float>()) {
        m_eta = intIOR/extIOR;
        m_invEta = 1.0f/m_eta;
    }

    // For pyBind
    inline RoughDielectricBSDF(float alpha, float intIOR, float extIOR, ptr<float> dAlpha)
        : m_distr(alpha), m_eta(intIOR/extIOR)
    {
        m_invEta = 1.0f/m_eta;
        m_spectrum = SpectrumAD(1.0f);
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 1> >(dAlpha.get(), nder, 1).cast<Float>());
    }

    inline RoughDielectricBSDF(float alpha, float intIOR, float extIOR, ptr<float> dAlpha, ptr<float> dEta)
        : m_distr(alpha), m_eta(intIOR/extIOR)
    {
        m_spectrum = SpectrumAD(1.0f);
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 1> >(dAlpha.get(), nder, 1).cast<Float>(),
                       Eigen::Map<Eigen::Array<float, nder, 1> >(dEta.get(), nder, 1).cast<Float>());
    }

    inline RoughDielectricBSDF(float alpha, float intIOR, float extIOR, const Spectrum3f& spectrum,
                               ptr<float> dAlpha)
            : m_distr(alpha), m_eta(intIOR/extIOR), m_spectrum(spectrum.cast<Float>())
    {
        m_invEta = 1.0f/m_eta;
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 1> >(dAlpha.get(), nder, 1).cast<Float>());
    }

    inline RoughDielectricBSDF(float alpha, float intIOR, float extIOR, const Spectrum3f& spectrum,
                               ptr<float> dAlpha, ptr<float> dEta)
            : m_distr(alpha), m_eta(intIOR/extIOR), m_spectrum(spectrum.cast<Float>())
    {
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 1> >(dAlpha.get(), nder, 1).cast<Float>(),
                       Eigen::Map<Eigen::Array<float, nder, 1> >(dEta.get(), nder, 1).cast<Float>());
    }

    inline RoughDielectricBSDF(float alpha, float intIOR, float extIOR, const Spectrum3f& spectrum,
                               ptr<float> dAlpha, ptr<float> dEta, ptr<float> dSpectrum)
            : m_distr(alpha), m_eta(intIOR/extIOR), m_spectrum(spectrum.cast<Float>())
    {
        initVelocities(Eigen::Map<Eigen::Array<float, nder, 1> >(dAlpha.get(), nder, 1).cast<Float>(),
                       Eigen::Map<Eigen::Array<float, nder, 1> >(dEta.get(), nder, 1).cast<Float>(),
                       Eigen::Map<Eigen::Array<float, nder, 3, Eigen::RowMajor> >(dSpectrum.get(), nder, 3).cast<Float>());
    }

    inline void initVelocities(const Eigen::Array<Float, nder, 1> &dAlpha) {
        m_distr.initVelocities(dAlpha);
    }

    inline void initVelocities(const Eigen::Array<Float, nder, 1> &dAlpha, const Eigen::Array<Float, nder, 1> &dEta) {
        m_distr.initVelocities(dAlpha);
        m_eta.der = dEta;
        m_invEta = 1.0f/m_eta;
    }

    inline void initVelocities(const Eigen::Array<Float, nder, 1> &dAlpha, const Eigen::Array<Float, nder, 1> &dEta,
                               const Eigen::Array<Float, nder, 3> &dSpectrum) {
        m_distr.initVelocities(dAlpha);
        m_eta.der = dEta;
        m_invEta = 1.0f/m_eta;
        m_spectrum.der = dSpectrum;
    }

    Spectrum eval(const Intersection &its, const Vector &wo,  EBSDFMode mode = EBSDFMode::ERadiance) const;
    SpectrumAD evalAD(const IntersectionAD &its, const VectorAD &wo,  EBSDFMode mode = EBSDFMode::ERadiance) const;

    Spectrum sample(const Intersection &its, const Array3 &rnd, Vector &wo, Float &pdf, Float &eta,  EBSDFMode mode = EBSDFMode::ERadiance) const;
    Float pdf(const Intersection &its, const Vector &wo) const;

    inline bool isTransmissive() const { return true; }
    inline bool isTwosided() const { return true; }
    inline bool isNull() const { return false; }

    FloatAD ior() const;

    std::string toString() const {
        std::ostringstream oss;
        oss << "BSDF_rough_dielectric [" << '\n'
            //<< "  alpha = " << m_alpha << '\n'
            << "  eta = " << m_eta << '\n'
            << "]" << std::endl;
        return oss.str();
    }

    MicrofacetDistribution m_distr;
    FloatAD m_eta, m_invEta;
    SpectrumAD m_spectrum; // for both reflectance and transmittance
};

#endif
