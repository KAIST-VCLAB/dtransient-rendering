#include "scene.h"
#include "emitter/area.h"
#include "emitter/area2.h"
#include "bsdf/null.h"
#include "bsdf/diffuse.h"
#include "bsdf/texturedDiffuse.h"
#include "bsdf/phong.h"
#include "bsdf/roughconductor.h"
#include "bsdf/roughdielectric.h"
#include "bsdf/twosided.h"
#include "medium/homogeneous.h"
#include "medium/heterogeneous.h"
#include "integrator/bdpt.h"
#include "integratorADps.h"
#include "integrator/differential/bdptAD.h"
#include "integrator/differential/bdptADbb.h" 
#include "config.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(dtrr, m) {
    m.doc() = "dtrr"; // optional module docstring

    py::class_<ptr<float>>(m, "float_ptr")
        .def(py::init<>())
        .def(py::init<std::size_t>());
    py::class_<ptr<int>>(m, "int_ptr")
        .def(py::init<std::size_t>());

    py::class_<Spectrum3f>(m, "Spectrum3f")
        .def(py::init<float, float, float>());

    py::class_<Camera>(m, "Camera")
        .def(py::init<int, int, ptr<float>, ptr<float>, float, int, ptr<float>>())
        .def("set_rect", &Camera::setCropRect);

    py::class_<Float>(m, "Float")
            .def(py::init<float>());

    py::class_<CameraTransient>(m, "CameraTransient")
            .def(py::init<int, int, ptr<float>, ptr<float>, float, int, ptr<float>, int, Float, Float, Float>())
            .def("set_rect", &CameraTransient::setCropRect);

    py::class_<Shape>(m, "Shape")
        .def(py::init<ptr<float>,
                      ptr<int>,
                      ptr<float>,
                      ptr<float>,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      ptr<float>>())
        .def_readonly("num_vertices", &Shape::num_vertices)
        .def("has_uvs", &Shape::hasUVs)
        .def("has_normals", &Shape::hasNormals);

    py::class_<BSDF>(m, "BSDF");
    py::class_<DiffuseBSDF, BSDF>(m, "BSDF_diffuse")
        .def(py::init<Spectrum3f>())
        .def(py::init<Spectrum3f>());
    py::class_<TexturedDiffuseBSDF, BSDF>(m, "BSDF_texturedDiffuse")
        .def(py::init<int, int, ptr<float>>());
    py::class_<NullBSDF, BSDF>(m, "BSDF_null")
        .def(py::init<>());
    py::class_<PhongBSDF, BSDF>(m, "BSDF_Phong")
        .def(py::init<Spectrum3f, Spectrum3f, float>())
        .def(py::init<Spectrum3f, Spectrum3f, float, ptr<float>, ptr<float>, ptr<float>>());
    py::class_<RoughConductorBSDF, BSDF>(m, "BSDF_roughconductor")
        .def(py::init<float, Spectrum3f, Spectrum3f>())
        .def(py::init<float, Spectrum3f, Spectrum3f, ptr<float>>())
        .def(py::init<float, Spectrum3f, Spectrum3f, ptr<float>, ptr<float>, ptr<float>>());
    py::class_<RoughDielectricBSDF, BSDF>(m, "BSDF_roughdielectric")
        .def(py::init<float, float, float>())
        .def(py::init<float, float, float, Spectrum3f>())
        .def(py::init<float, float, float, ptr<float>>())
        .def(py::init<float, float, float, ptr<float>, ptr<float>>())
        .def(py::init<float, float, float, Spectrum3f, ptr<float>>())
        .def(py::init<float, float, float, Spectrum3f, ptr<float>, ptr<float>>())
        .def(py::init<float, float, float, Spectrum3f, ptr<float>, ptr<float>, ptr<float>>());
    py::class_<TwosidedBSDF, BSDF>(m, "BSDF_twosided")
        .def(py::init<const BSDF *>());

    py::class_<PhaseFunction>(m, "Phase");
    py::class_<HGPhaseFunction, PhaseFunction>(m, "HG")
        .def(py::init<float>())
        .def(py::init<float, ptr<float>>());
    py::class_<IsotropicPhaseFunction, PhaseFunction>(m, "Isotropic")
        .def(py::init<>());

    py::class_<Emitter>(m, "Emitter");
    py::class_<AreaLight, Emitter>(m, "AreaLight")
        .def(py::init<int, Spectrum3f>())
        .def(py::init<int, Spectrum3f, ptr<float>>());
    py::class_<AreaLightEx, Emitter>(m, "AreaLightEx")
        .def(py::init<int, Spectrum3f, float>())
        .def(py::init<int, Spectrum3f, float, ptr<float>, ptr<float>>());

    py::class_<Medium>(m, "Medium");
    py::class_<Homogeneous, Medium>(m, "Homogeneous")
        .def(py::init<float, Spectrum3f, int>())
        .def(py::init<float, Spectrum3f, int, ptr<float>, ptr<float>>());
    py::class_<Heterogeneous, Medium>(m, "Heterogeneous")
        .def(py::init<const std::string &, ptr<float>, float, Spectrum3f, int>())
        .def(py::init<const std::string &, ptr<float>, float, Spectrum3f, int,
                      ptr<float>,  ptr<float>,  ptr<float>,  ptr<float>>())
        .def(py::init<const std::string &, ptr<float>, float, const std::string &, int>())
        .def(py::init<const std::string &, ptr<float>, float, const std::string &, int,
                      ptr<float>,  ptr<float>,  ptr<float>>());

    py::class_<Scene>(m, "Scene")
        .def(py::init<const CameraTransient &, 
                      const std::vector<const Shape*> &,
                      const std::vector<const BSDF*> &,
                      const std::vector<const Emitter*> &,
                      const std::vector<const PhaseFunction*> &,
                      const std::vector<const Medium*> &>())
        .def(py::init<const CameraTransient &, 
                      const std::vector<const Shape*> &,
                      const std::vector<const BSDF*> &,
                      const std::vector<const Emitter*> &,
                      const std::vector<const PhaseFunction*> &,
                      const std::vector<const Medium*> &,
                      bool>())
        .def("initEdges", &Scene::initEdgesPy)
        .def("initPathSpaceEdges", &Scene::initPathSpaceEdgesPy1_1)
        .def("initPathSpaceEdges", &Scene::initPathSpaceEdgesPy1_2)
        .def("initPathSpaceEdges", &Scene::initPathSpaceEdgesPy2_1)
        .def("initPathSpaceEdges", &Scene::initPathSpaceEdgesPy2_2);

    py::class_<RenderOptions>(m, "RenderOptions")
        .def(py::init<uint64_t, int, int, int, int, bool>())
        .def(py::init<uint64_t, int, int, int, int, bool, int>())
        .def(py::init<uint64_t, int, int, int, int, bool, int, float>())
        .def(py::init<const RenderOptions>())
        .def_readwrite("seed", &RenderOptions::seed)
        .def_readwrite("spp", &RenderOptions::num_samples)
        .def_readwrite("sppe", &RenderOptions::num_samples_primary_edge)
        .def_readwrite("sppse", &RenderOptions::num_samples_secondary_edge)
        .def_readwrite("max_bounces", &RenderOptions::max_bounces)
        .def_readwrite("quiet", &RenderOptions::quiet)
        .def_readwrite("mode", &RenderOptions::mode)
        .def_readwrite("ddistCoeff", &RenderOptions::ddistCoeff)
        .def_readwrite("sppse0", &RenderOptions::num_samples_secondary_edge_direct)
        .def_readwrite("sppse1", &RenderOptions::num_samples_secondary_edge_indirect)
        .def_readwrite("grad_threshold", &RenderOptions::grad_threshold)
        .def_readwrite("primary_delta", &RenderOptions::primary_delta);

    py::class_<GuidingOptions>(m, "GuidingOptions")
        .def(py::init<int, const std::vector<int>&>())
        .def(py::init<int, const std::vector<int>&, size_t, size_t>())
        .def(py::init<int, const std::vector<int>&, size_t, size_t, float>())
        .def_readwrite("params", &GuidingOptions::params)
        .def_readwrite("num_cam_path", &GuidingOptions::num_cam_path)
        .def_readwrite("num_light_path", &GuidingOptions::num_light_path)
        .def_readwrite("search_radius", &GuidingOptions::search_radius)
        .def_readwrite("quiet", &GuidingOptions::quiet);

    py::class_<Integrator>(m, "Integrator");
    py::class_<IntegratorAD, Integrator>(m, "IntegratorAD");
    py::class_<IntegratorAD_PathSpace, IntegratorAD>(m, "IntegratorAD_PathSpace");
    py::class_<BidirectionalPathTracer, IntegratorAD_PathSpace>(m, "BidirectionalPathTracer")
        .def(py::init<>())
        .def("render", &BidirectionalPathTracer::render);

    py::class_<BidirectionalPathTracerAD, IntegratorAD_PathSpace>(m, "BidirectionalPathTracerAD")
        .def(py::init<>())
        .def("render", &BidirectionalPathTracerAD::render)
        .def("preprocess",&BidirectionalPathTracerAD::preprocess)
        .def_readonly("num_rejected", &BidirectionalPathTracerAD::rejectedSamples)
        .def_readonly("num_outtime", &BidirectionalPathTracerAD::outtimeSamples);

    py::class_<BidirectionalPathTracerADBiBd, BidirectionalPathTracerAD>(m, "BidirectionalPathTracerADBiBd")
            .def(py::init<>())
            .def("render", &BidirectionalPathTracerADBiBd::render)
            .def("preprocess",&BidirectionalPathTracerADBiBd::preprocess)
            .def_readonly("num_rejected", &BidirectionalPathTracerADBiBd::rejectedSamples)
            .def_readonly("num_outtime", &BidirectionalPathTracerADBiBd::outtimeSamples);


    m.attr("nder") = nder;
    m.attr("angleEps") = AngleEpsilon;
    m.attr("edgeEps") = EdgeEpsilon;

}
