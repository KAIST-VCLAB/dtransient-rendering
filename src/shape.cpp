#include "shape.h"
#include "ray.h"
#include "rayAD.h"
#include "intersection.h"
#include "intersectionAD.h"
#include "math_func.h"
#include <map>
#include <iomanip>


Shape::Shape(ptr<float> vertices, ptr<int> indices, ptr<float> uvs, ptr<float> normals, int num_vertices, int num_triangles,
            int light_id, int bsdf_id, int med_int_id, int med_ext_id, ptr<float> velocities):
            num_vertices(num_vertices), num_triangles(num_triangles), light_id(light_id), bsdf_id(bsdf_id), med_int_id(med_int_id), med_ext_id(med_ext_id)
{
    this->vertices.resize(num_vertices);
    for (int i = 0; i < num_vertices; i++)
        this->vertices[i] = Vector3AD(Vector3(vertices[3*i], vertices[3*i + 1], vertices[3*i + 2]));

#ifdef SHAPE_COMPUTE_VTX_NORMAL
    assert(normals.get() == nullptr);
#else
    if (normals.get() != nullptr) {
        assert(light_id < 0); // Do not allow emitters to have shading normals
        this->normals.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++)
            this->normals[i] = Vector3AD(Vector3(normals[3*i], normals[3*i + 1], normals[3*i + 2]).normalized());
    }
#endif

    this->indices.resize(num_triangles);
    for (int i = 0; i < num_triangles; i++)
        this->indices[i] = Vector3i(indices[3*i], indices[3*i + 1], indices[3*i + 2]);

    if (uvs.get() != nullptr ) {
        this->uvs.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++)
            this->uvs[i] = Vector2(uvs[2*i], uvs[2*i + 1]);
    }

    if (!velocities.is_null()) {
        int nrows = 3;
        int ncols = num_vertices;
        for (int ider = 0; ider < nder; ider++) {
            Eigen::MatrixXf dx(nrows, ncols);
            dx = Eigen::Map<Eigen::MatrixXf>(velocities.get() + ider*nrows*ncols, dx.rows(), dx.cols());
#ifdef SHAPE_COMPUTE_VTX_NORMAL
            initVelocities(dx.cast<Float>(), ider);
#else
            if (normals.get() != nullptr) {
                Eigen::MatrixXf dn(nrows, ncols);
                dn = Eigen::Map<Eigen::MatrixXf>(velocities.get() + num_vertices*3*nder, dn.rows(), dn.cols());
                initVelocities(dx.cast<Float>(), dn.cast<Float>(), ider);
            } else {
                initVelocities(dx.cast<Float>(), ider);
            }
#endif
        }
    } else {
        computeFaceNormals();
#ifdef SHAPE_COMPUTE_VTX_NORMAL
        computeVertexNormals();
#endif
    }

    constructEdges();
}

void Shape::zeroVelocities() {
    for ( int i = 0; i < num_vertices; i++ )
        vertices[i].zeroGrad();
    if ( !normals.empty() ) {
        assert(static_cast<int>(normals.size()) == num_vertices);
        for ( int i = 0; i < num_vertices; i++ )
            normals[i].zeroGrad();
    }
    computeFaceNormals();
#ifdef SHAPE_COMPUTE_VTX_NORMAL
    computeVertexNormals();
#endif
}

void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx) {
    assert(dx.rows() == 3*nder && dx.cols() == num_vertices);
    for ( int i = 0; i < num_vertices; i++ )
        for ( int j = 0; j < nder; ++j )
            vertices[i].grad(j) = dx.block(j*3, i, 3, 1);
    computeFaceNormals();
#ifdef SHAPE_COMPUTE_VTX_NORMAL
    computeVertexNormals();
#endif
}

void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, int der_index) {
    assert(dx.rows() == 3 && dx.cols() == num_vertices &&
           der_index >= 0 && der_index < nder);
    for ( int i = 0; i < num_vertices; i++ )
        vertices[i].grad(der_index) = dx.col(i);
    computeFaceNormals();
#ifdef SHAPE_COMPUTE_VTX_NORMAL
    computeVertexNormals();
#endif
}

#ifndef SHAPE_COMPUTE_VTX_NORMAL
void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, const Eigen::Matrix<Float, -1, -1> &dn) {
    assert(dx.rows() == 3*nder && dx.cols() == num_vertices &&
           dn.rows() == 3*nder && dn.cols() == num_vertices && normals.size() == vertices.size());
    initVelocities(dx);
    for ( int i = 0; i < num_vertices; i++ )
        for ( int j = 0; j < nder; ++j )
            normals[i].grad(j) = dn.block(j*3, i, 3, 1);
}

void Shape::initVelocities(const Eigen::Matrix<Float, -1, -1> &dx, const Eigen::Matrix<Float, -1, -1> &dn, int der_index) {
    assert(dn.rows() == 3 && dn.cols() == num_vertices && normals.size() == vertices.size() &&
           der_index >= 0 && der_index < nder);
    initVelocities(dx, der_index);
    for ( int i = 0; i < num_vertices; i++ )
        normals[i].grad(der_index) = dn.col(i);
}
#endif

void Shape::advance(Float stepSize, int derId) {
    assert(derId >= 0 && derId < nder);
    for ( int i = 0; i < num_vertices; ++i ) {
        vertices[i].val = vertices[i].advance(stepSize, derId);
        if ( !normals.empty() )
            normals[i].val = normals[i].advance(stepSize, derId);
    }
    computeFaceNormals();
#ifdef SHAPE_COMPUTE_VTX_NORMAL
    computeVertexNormals();
#endif
    constructEdges();
}

void Shape::computeFaceNormals() {
    face_distrb.clear();
    face_distrb.reserve(num_triangles);

    faceNormals.resize(num_triangles);
    //faceRotations.resize(num_triangles);
    for ( int i = 0; i < num_triangles; ++i ) {
        const auto& ind = getIndices(i);
        const auto& v0 = getVertexAD(ind(0));
        const auto& v1 = getVertexAD(ind(1));
        const auto& v2 = getVertexAD(ind(2));
        auto& cur = faceNormals[i];
        cur = (v1 - v0).cross(v2 - v0);
        Float area = cur.val.norm();
        face_distrb.append(0.5f*area);
        if ( area < Epsilon ) {
            std::cout << ind << std::endl;
            std::cout << v0.val << std::endl;
            std::cout << v1.val << std::endl;
            std::cout << v2.val << std::endl;
            std::cerr << std::scientific << std::setprecision(2)
                      << "[Warning] Vanishing normal for face #" << i << " (norm = " << area << ")" << std::endl;
        }
        else
            cur.normalize();

        //faceRotations[i] = Eigen::AngleAxis(0.5f*M_PI, cur.val).toRotationMatrix();
    }

    face_distrb.normalize();
}

#ifdef SHAPE_COMPUTE_VTX_NORMAL
FloatAD unitAngle(const Vector3AD &u, const Vector3AD &v) {
    if (u.val.dot(v.val) < 0.0)
        return M_PI - (0.5f * (v+u).norm()).asin();
    else
        return 2.0 * (0.5f * (v-u).norm()).asin();
}

void Shape::computeVertexNormals() {
    normals.resize(num_vertices);
    for ( int i = 0; i < num_triangles; ++i ) {
        const Vector3i& ind = getIndices(i);
        Vector3AD fn;
        for (int j = 0; j < 3; j++) {
            const auto& v0 = getVertexAD(ind(j));
            const auto& v1 = getVertexAD(ind((j+1)%3));
            const auto& v2 = getVertexAD(ind((j+2)%3));
            Vector3AD sideA = v1 - v0, sideB = v2 - v0;
            if (j == 0) fn = sideA.cross(sideB).normalized();
            FloatAD angleAD = unitAngle(sideA.normalized(), sideB.normalized());
            normals[ind(j)] += fn * angleAD;
        }
    }

    for (int i = 0; i < num_vertices; i++) {
        VectorAD &n = normals[i];
        FloatAD length = n.norm();
        assert(length > Epsilon);
        n /= length;
    }
}
#endif

Float Shape::getArea(int index) const {
    auto& ind = getIndices(index);
    auto& v0 = getVertex(ind(0));
    auto& v1 = getVertex(ind(1));
    auto& v2 = getVertex(ind(2));
    return 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

FloatAD Shape::getAreaAD(int index) const {
    auto& ind = getIndices(index);
    auto& v0 = getVertexAD(ind(0));
    auto& v1 = getVertexAD(ind(1));
    auto& v2 = getVertexAD(ind(2));
    return 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

// void Shape::samplePosition(int index, const Vector2 &rnd2, Vector &pos, Vector &norm) const {
//     const Vector3i& ind = getIndices(index);
//     const Vector& v0 = getVertex(ind(0));
//     const Vector& v1 = getVertex(ind(1));
//     const Vector& v2 = getVertex(ind(2));
//     Float a = std::sqrt(rnd2[0]);
//     pos = v0 + (v1 - v0)*(1.0f - a) + (v2 - v0)*(a*rnd2[1]);
//     norm = faceNormals[index].val;
// }

// void Shape::samplePositionAD(int index, const Vector2 &rnd2, VectorAD &pos, VectorAD &norm) const {
//     const Vector3i& ind = getIndices(index);
//     const VectorAD& v0 = getVertex(ind(0));
//     const VectorAD& v1 = getVertex(ind(1));
//     const VectorAD& v2 = getVertex(ind(2));
//     FloatAD a = std::sqrt(rnd2[0]);
//     pos = v0 + (v1 - v0)*(1.0f - a) + (v2 - v0)*(a*rnd2[1]);
//     norm = faceNormals[index];
// }

int Shape::samplePosition(const Vector2 &_rnd2, PositionSamplingRecord &pRec) const {
    Vector2 rnd2(_rnd2);
    int index = static_cast<int>(face_distrb.sampleReuse(rnd2[0]));
    const Vector3i& ind = getIndices(index);
    const Vector& v0 = getVertex(ind(0));
    const Vector& v1 = getVertex(ind(1));
    const Vector& v2 = getVertex(ind(2));
    Float a = std::sqrt(rnd2[0]);
    pRec.p = v0 + (v1 - v0)*(1.0f - a) + (v2 - v0)*(a*rnd2[1]);
    pRec.n = faceNormals[index].val;
    pRec.uv = Vector2{ 1.0f - a, a*rnd2[1] };
    return index;
}

int Shape::samplePositionAD(const Vector2 &_rnd2, PositionSamplingRecordAD &pRec) const {
    Vector2 rnd2(_rnd2);
    int index = static_cast<int>(face_distrb.sampleReuse(rnd2[0]));
    const Vector3i& ind = getIndices(index);
    const VectorAD& v0 = getVertexAD(ind(0));
    const VectorAD& v1 = getVertexAD(ind(1));
    const VectorAD& v2 = getVertexAD(ind(2));
    Float a = std::sqrt(rnd2[0]);
    pRec.p = v0 + (v1 - v0)*(1.0f - a) + (v2 - v0)*(a*rnd2[1]);
    pRec.n = faceNormals[index];
    pRec.J = getAreaAD(index);
    pRec.J /= pRec.J.val;
    return index;
}

void Shape::rayIntersect(int tri_index, const Ray &ray, Intersection &its) const {
    const Vector3i &ind = getIndices(tri_index);
    const Vector &v0 = getVertex(ind(0)), &v1 = getVertex(ind(1)), &v2 = getVertex(ind(2));
    Vector2 uvs0, uvs1, uvs2;
    if (uvs.size() != 0) {
        uvs0 = getUV(ind(0));
        uvs1 = getUV(ind(1));
        uvs2 = getUV(ind(2));
    } else {
        uvs0 = Vector2{0, 0};
        uvs1 = Vector2{1, 0};
        uvs2 = Vector2{1, 1};
    }
    const Array uvt = rayIntersectTriangle(v0, v1, v2, ray);
    const Float &u = uvt(0), &v = uvt(1), &t = uvt(2);
    const Float w = 1.f - (u + v);
    const Vector2 uv = w * uvs0 + u * uvs1 + v * uvs2;
    const Vector hit_pos = ray.org + ray.dir * t;
    Vector geom_normal = faceNormals[tri_index].val;
    Vector shading_normal = geom_normal;
    if (normals.size() != 0) {
        Vector n0 = getShadingNormal(ind(0)), n1 = getShadingNormal(ind(1)), n2 = getShadingNormal(ind(2));
        Vector nn = w * n0 + u * n1 + v * n2;
        // Shading normal computation
        shading_normal = nn.normalized();
        // Flip geometric normal to the same side of shading normal
        if (geom_normal.dot(shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }
    its.geoFrame = Frame(geom_normal);
    its.shFrame = Frame(shading_normal);
    its.p = hit_pos;
    its.t = t;
    its.uv = uv;
    its.wi = its.toLocal(-ray.dir);

    its.barycentric = Vector2(u, v);

    its.opd = its.getIorDir(ray.dir) * t;
}

void Shape::rayIntersectAD(int tri_index, const RayAD &ray, IntersectionAD &its) const {
    const Vector3i &ind = getIndices(tri_index);
    const VectorAD &v0 = getVertexAD(ind(0)), &v1 = getVertexAD(ind(1)), &v2 = getVertexAD(ind(2));
    Vector2 uvs0, uvs1, uvs2;
    if (uvs.size() != 0) {
        uvs0 = getUV(ind(0));
        uvs1 = getUV(ind(1));
        uvs2 = getUV(ind(2));
    } else {
        uvs0 = Vector2{0, 0};
        uvs1 = Vector2{1, 0};
        uvs2 = Vector2{1, 1};
    }
    const ArrayAD uvt = rayIntersectTriangleAD(v0, v1, v2, ray);
    ArrayAD::ElementConstRefType u = uvt(0), v = uvt(1), t = uvt(2);
    const FloatAD w = 1.f - (u + v);
    const Vector2AD uv = w*Vector2AD(uvs0) + u*Vector2AD(uvs1) + v*Vector2AD(uvs2);
    const VectorAD hit_pos = ray.org + ray.dir * t;
    VectorAD geom_normal = faceNormals[tri_index];
    VectorAD shading_normal = geom_normal;
    if (normals.size() != 0) {
        VectorAD n0 = getShadingNormalAD(ind(0)), n1 = getShadingNormalAD(ind(1)), n2 = getShadingNormalAD(ind(2));
        VectorAD nn = w * n0 + u * n1 + v * n2;
        // Shading normal computation
        shading_normal = nn.normalized();
        // Flip geometric normal to the same side of shading normal
        if (geom_normal.dot(shading_normal) < 0.f) {
            geom_normal = -geom_normal;
        }
    }
    its.geoFrame = FrameAD(geom_normal);
    its.shFrame = FrameAD(shading_normal);
    its.p = hit_pos;
    its.t = t;
    its.uv = uv;
    its.wi = its.toLocal(-ray.dir);

    its.barycentric = Vector2AD(u, v);

    its.opd = its.getIorDir(ray.dir) * t;
}

void Shape::constructEdges() {
    std::map<std::pair<int,int>, std::vector<Vector2i>> edge_map;
    for ( int itri = 0; itri < num_triangles; itri++ ) {
        Vector3i ind = getIndices(itri);
        for ( int iedge = 0; iedge < 3; iedge++ ) {
            int k1 = iedge, k2 = (iedge + 1) % 3;
            std::pair<int, int> key = (ind[k1] < ind[k2]) ? std::make_pair(ind[k1], ind[k2])
                                                          : std::make_pair(ind[k2], ind[k1]);
            if (edge_map.find(key) == edge_map.end())
                edge_map[key] = std::vector<Vector2i>();
            edge_map[key].push_back(Vector2i(itri, ind[(iedge + 2) % 3]));
        }
    }

    edges.clear();
    for ( const auto &it : edge_map ) {
        Float length = (getVertex(it.first.first) - getVertex(it.first.second)).norm();

        // check if good mesh
        if ( it.second.size() > 2 ) {
            std::cerr << "Every edge can be shared by at most 2 faces!" << std::endl;
            assert(false);
        }
        else if ( it.second.size() == 2 ) {
            const int ind0 = it.second[0][0], ind1 = it.second[1][0];
            if ( ind0 == ind1 ) {
                std::cerr << "Duplicated faces!" << std::endl;
                assert(false);
            }

            const Vector &n0 = faceNormals[ind0].val, &n1 = faceNormals[ind1].val;
            Float val = n0.dot(n1);
            if ( val < -1.0f + EdgeEpsilon ) {
                std::cerr << "Inconsistent normal orientation!" << std::endl;
                assert(false);
            }
            else if ( val < 1.0f - EdgeEpsilon ) {
                Float tmp0 = n0.dot(vertices[it.second[1][1]].val - vertices[it.first.first].val),
                      tmp1 = n1.dot(vertices[it.second[0][1]].val - vertices[it.first.first].val);
                assert(math::signum(tmp0)*math::signum(tmp1) > 0.5f);
                edges.push_back(Edge(it.first.first, it.first.second, ind0, ind1, length, it.second[0][1], tmp0 > Epsilon ? -1 : 1));
            }
        }
        else {
            assert(it.second.size() == 1);
            edges.push_back(Edge(it.first.first, it.first.second, it.second[0][0], -1, length, it.second[0][1], 0));
        }
    }

    edge_distrb.clear();
    for ( const Edge &edge : edges ) edge_distrb.append(edge.length);
    edge_distrb.normalize();
}

int Shape::isSihoulette(const Edge& edge, const Vector& p) const {
    if (edge.f0 == -1 || edge.f1 == -1) {
        // Only adjacent to one face
        return 2;
    }
    const Vector &v0 = getVertex(edge.v0), &v1 = getVertex(edge.v1);

    bool frontfacing0 = false;
    const Vector3i &ind0 = getIndices(edge.f0);
    for (int i = 0; i < 3; i++) {
        if (ind0[i] != edge.v0 && ind0[i] != edge.v1) {
            const Vector& v = getVertex(ind0[i]);
            Vector n0 = (v0 - v).cross(v1 - v).normalized();
            frontfacing0 = n0.dot(p - v) > 0.0f;
            break;
        }
    }

    bool frontfacing1 = false;
    const Vector3i &ind1 = getIndices(edge.f1);
    for (int i = 0; i < 3; i++) {
        if (ind1[i] != edge.v0 && ind1[i] != edge.v1) {
            const Vector& v = getVertex(ind1[i]);
            Vector n1 = (v1 - v).cross(v0 - v).normalized();
            frontfacing1 = n1.dot(p - v) > 0.0f;
            break;
        }
    }
    if ((frontfacing0 && !frontfacing1) || (!frontfacing0 && frontfacing1))
        return 2;

    // If we are not using Phong normal, every edge is silhouette
    return hasNormals() ? 0 : 1;
}

const Edge& Shape::sampleEdge(Float& rnd, Float& pdf) const {
    Float pdf1;
    int idx_edge = edge_distrb.sampleReuse(rnd, pdf1);
    pdf *= pdf1;
    return edges[idx_edge];
}

void Shape::getPoint(int tri_index, const Vector2AD &barycentric, VectorAD &x, VectorAD &n, FloatAD &J) const {
    assert(tri_index >= 0 && tri_index < num_triangles);
    const Vector3i &ind = indices[tri_index];
    const VectorAD &v0 = vertices[ind[0]], &v1 = vertices[ind[1]], &v2 = vertices[ind[2]];
    x = (1.0f - barycentric(0) - barycentric(1))*v0 + barycentric(0)*v1 + barycentric(1)*v2;
    n = faceNormals[tri_index];
    J = getAreaAD(tri_index);
    J /= J.val;
}

void Shape::getPoint(int tri_index, const Vector2AD &barycentric, IntersectionAD& its_AD, FloatAD &J) const {
    VectorAD geom_normal;
    getPoint(tri_index, barycentric, its_AD.p, geom_normal, J);

    const Vector3i &ind = indices[tri_index];
    const FloatAD w = 1.0f - barycentric(0) - barycentric(1);

    VectorAD shading_normal;
    if ( normals.size() != 0 ) {
        const VectorAD &n0 = getShadingNormalAD(ind(0)), &n1 = getShadingNormalAD(ind(1)), &n2 = getShadingNormalAD(ind(2));
        shading_normal = (w*n0 + barycentric(0)*n1 + barycentric(1)*n2).normalized();
        if ( geom_normal.val.dot(shading_normal.val) < 0.f ) {
            geom_normal = -geom_normal;
        }
    } else {
        shading_normal = geom_normal;
    }
    its_AD.geoFrame = FrameAD(geom_normal);
    its_AD.shFrame  = FrameAD(shading_normal);

    Vector2 uvs0, uvs1, uvs2;
    if (uvs.size() != 0) {
        uvs0 = getUV(ind(0));
        uvs1 = getUV(ind(1));
        uvs2 = getUV(ind(2));
    } else {
        uvs0 = Vector2{0, 0};
        uvs1 = Vector2{1, 0};
        uvs2 = Vector2{1, 1};
    }
    its_AD.uv = w*Vector2AD(uvs0) + barycentric(0)*Vector2AD(uvs1) + barycentric(1)*Vector2AD(uvs2);
}
