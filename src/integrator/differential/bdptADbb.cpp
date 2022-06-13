
#include "scene.h"
#include "bdptADbb.h"


int BidirectionalPathTracerADBiBd::radiance(const Scene& scene, RndSampler* sampler, const Intersection &its, int max_bounces,
                                        Spectrum *ret, std::tuple<Spectrum, Float, int>* ret_trans) const
{
    const int tid = omp_get_thread_num();
    assert(tid < BDPT_MAX_THREADS && max_bounces + 1 < BDPT_MAX_PATH_LENGTH);
    return bidir::radiance(scene, sampler, its, max_bounces, m_path[2*tid], m_path[2*tid + 1], ret, ret_trans);
}

std::pair<int, int> BidirectionalPathTracerADBiBd::weightedImportance(const Scene& scene, RndSampler* sampler, const Intersection& its, int max_bounces, const Spectrum *weight,
                                                                  std::pair<int, Spectrum>* ret, std::tuple<int, Spectrum, Float, int>* ret_trans) const
{
    const int tid = omp_get_thread_num();
    assert(tid < BDPT_MAX_THREADS && max_bounces + 1 < BDPT_MAX_PATH_LENGTH);
    return bidir::weightedImportance(scene, sampler, its, max_bounces, static_cast<int>(m_taskId[tid] % scene.camera.getNumPixels()),
                                     m_path[2*tid], m_path[2*tid + 1], weight, ret, ret_trans);
}