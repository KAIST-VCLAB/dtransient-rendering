#include "integratorAD.h"
#include "scene.h"
#include "sampler.h"
#include "rayAD.h"
#include <chrono>
#include <iomanip>

static const int nworker = omp_get_num_procs();


IntegratorAD::IntegratorAD() {
    omp_init_lock(&messageLock);
}


IntegratorAD::~IntegratorAD() {
    omp_destroy_lock(&messageLock);
}


void IntegratorAD::renderPrimaryEdges(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient) const {
    using namespace std::chrono;

    const auto &camera = scene.camera;
    const bool cropped = camera.rect.isValid();
    const int num_pixels = camera.getNumPixels();
    const EdgeManager* ptr_eManager = scene.ptr_edgeManager;
    const int duration = camera.duration;

    std::vector<RndSampler> samplers;
    for (int iworker = 0; iworker < nworker; iworker++)
        samplers.push_back(RndSampler(options.seed, iworker));

    const Float imageDelta = options.primary_delta; 
    Eigen::Array<Float, -1, -1> edge_contrib = Eigen::Array<Float, -1, -1>::Zero(num_pixels*nder*3, nworker);
    Eigen::Array<Float, -1, -1> edge_contrib_trans = Eigen::Array<Float, -1, -1>::Zero(num_pixels*duration*nder*3, nworker);

    const int num_samples_per_block = 128;
    const int num_block = static_cast<int>(std::ceil(ptr_eManager->getPrimaryEdgePDFsum() *
                                                     options.num_samples_primary_edge/num_samples_per_block));
    const int num_samples = num_block*num_samples_per_block;

    auto _start = high_resolution_clock::now();
    const Medium* med_cam = camera.getMedID() == -1 ? nullptr : scene.medium_list[camera.getMedID()];

    int finished_block = 0;
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
    for (int index_block = 0; index_block < num_block; ++index_block) {
        for (int omp_i = 0; omp_i < num_samples_per_block; omp_i++) {
            const int tid = omp_get_thread_num();
            Vector2i xyPixel;
            Vector2AD p;
            Vector2 norm, p1;
            Float maxD = ptr_eManager->samplePrimaryEdge(camera, samplers[tid].next1D(), xyPixel, p, norm);
            if ( !p.der.isZero(Epsilon) ) {
                Ray ray = camera.samplePrimaryRay(xyPixel.x(), xyPixel.y(), p.val);
                Float trans = scene.evalTransmittance(ray, false, med_cam, maxD, &samplers[tid], options.max_bounces - 1);
                if (trans > Epsilon) {
                    Spectrum deltaVal;
                    int len_array = (options.max_bounces+1)*(options.max_bounces+2)/2-1;//options.max_bounces + 1;
                    int np1, np2;
                    std::vector<std::tuple<Spectrum, Float, int>> valarr1(len_array);
                    std::vector<std::tuple<Spectrum, Float, int>> valarr2(len_array);
                    p1 = xyPixel.cast<Float>() + p.val - norm*imageDelta;
                    std::pair<Spectrum, int> temp_pair = pixelColor(scene, options, &samplers[tid], p1.x(), p1.y(), &valarr1[0]);
                    np1 = std::get<1>(temp_pair);
                    deltaVal = std::get<0>(temp_pair);
                    p1 = xyPixel.cast<Float>() + p.val + norm*imageDelta;
                    temp_pair = pixelColor(scene, options, &samplers[tid], p1.x(), p1.y(), &valarr2[0]);
                    np2 = std::get<1>(temp_pair);
                    deltaVal -= std::get<0>(temp_pair);

                        int idx_pixel = cropped ? (xyPixel.x() - camera.rect.offset_x) +
                                                  camera.rect.crop_width * (xyPixel.y() - camera.rect.offset_y)
                                                : xyPixel.x() + camera.width * xyPixel.y();
                        for (int j = 0; j < nder; ++j) {
                            int offset = (j * num_pixels + idx_pixel) * 3;
                            auto coeff = norm.dot(p.grad(j)) * ptr_eManager->getPrimaryEdgePDFsum();
                            edge_contrib.block<3, 1>(offset, tid) += coeff * deltaVal;
                            for (int type = 0; type < 2; type++) {
                                int np = type == 0 ? np1 : np2;
                                for (int i_path = 0; i_path < np; i_path++) {
                                    Spectrum val_trans;
                                    Float pathTime;
                                    std::tie(val_trans, pathTime, std::ignore) =
                                            type == 0 ? valarr1[i_path] : valarr2[i_path];
                                    pathTime *= INV_C;
                                    int i_bin_start, i_bin_end;
                                    camera.bin_range(pathTime, i_bin_start, i_bin_end);
                                    camera.clip_bin_index(i_bin_start, i_bin_end);
                                    for (int i_bin = i_bin_start; i_bin <= i_bin_end; i_bin++)
                                        edge_contrib_trans.block<3, 1>(offset * duration + i_bin * 3, tid)
                                                += coeff * val_trans * camera.eval_tsens(pathTime, i_bin)
                                                   * (type == 0 ? 1.0 : -1.0);
                                }
                            }
                        }
                }
            }
        }

        if ( !options.quiet ) {
            omp_set_lock(&messageLock);
            progressIndicator(Float(++finished_block)/num_block);
            omp_unset_lock(&messageLock);
        }
    }
    edge_contrib /= static_cast<Float>(num_samples);
    Eigen::ArrayXf output = edge_contrib.rowwise().sum().cast<float>();
    for ( int i = 0; i < num_pixels; ++i )
        for ( int j = 0; j < nder; ++j ) {
            int offset0 = ((j + 1)*num_pixels + i)*3, offset1 = (j*num_pixels + i)*3;
            rendered_image[offset0    ] += output[offset1    ];
            rendered_image[offset0 + 1] += output[offset1 + 1];
            rendered_image[offset0 + 2] += output[offset1 + 2];
        }
    if ( !rendered_transient.is_null() ) {
        edge_contrib_trans /= static_cast<Float>(num_samples);
        Eigen::ArrayXf output_trans = edge_contrib_trans.rowwise().sum().cast<float>();
        for (int i = 0; i < num_pixels; ++i)
            for (int j = 0; j < nder; ++j) {
                int offset0 = ((j + 1) * num_pixels + i) * 3, offset1 = (j * num_pixels + i) * 3;
                for (int k = 0; k < duration; ++k) {
                    int offset2 = offset0 * duration + k * 3, offset3 = offset1 * duration + k * 3;
                    rendered_transient[offset2] += output_trans[offset3];
                    rendered_transient[offset2 + 1] += output_trans[offset3 + 1];
                    rendered_transient[offset2 + 2] += output_trans[offset3 + 2];
                }
            }
    }
    if ( !options.quiet )
        std::cout << "\nDone in " << duration_cast<seconds>(high_resolution_clock::now() - _start).count() << " seconds." << std::endl;
}


void IntegratorAD::render(const Scene &scene, const RenderOptions &options, ptr<float> rendered_image, ptr<float> rendered_transient) const { 
    using namespace std::chrono;

    const auto &camera = scene.camera;
    const bool cropped = camera.rect.isValid();
    const int num_pixels = cropped ? camera.rect.crop_width * camera.rect.crop_height
                                   : camera.width * camera.height;
    const int size_block = 4;
    const int duration = camera.duration;

    if ( !options.quiet )
        std::cout << "Rendering using [ " << getName() << " ] and " << nworker << " workers ..." << std::endl;

    // Pixel sampling
    if ( options.num_samples > 0 )
    {
        int num_block = std::ceil((Float)num_pixels/size_block);
        auto _start = high_resolution_clock::now();

        int finished_block = 0;
#pragma omp parallel for num_threads(nworker) schedule(dynamic, 1)
        for (int index_block = 0; index_block < num_block; index_block++) {
            int block_start = index_block*size_block;
            int block_end = std::min((index_block+1)*size_block, num_pixels);

            for (int idx_pixel = block_start; idx_pixel < block_end; idx_pixel++) {
                int ix = cropped ? camera.rect.offset_x + idx_pixel % camera.rect.crop_width
                                 : idx_pixel % camera.width;
                int iy = cropped ? camera.rect.offset_y + idx_pixel / camera.rect.crop_width
                                 : idx_pixel / camera.width;
                RndSampler sampler(options.seed, idx_pixel);

                SpectrumAD pixel_val;
                ptr<float> tmp_temp_hist;
                if (!rendered_transient.is_null()) {
                    tmp_temp_hist = rendered_transient + idx_pixel * scene.camera.duration * 3;
                    for (int i_bin = 0; i_bin < duration; i_bin++){
                        tmp_temp_hist[i_bin*3    ] = 0.0f;
                        tmp_temp_hist[i_bin*3 + 1] = 0.0f;
                        tmp_temp_hist[i_bin*3 + 2] = 0.0f;
                        for (int ch = 1; ch <= nder; ch++){
                            int offset = (ch*num_pixels*duration + i_bin)*3;
                            tmp_temp_hist[offset    ] = 0.0f;
                            tmp_temp_hist[offset + 1] = 0.0f;
                            tmp_temp_hist[offset + 2] = 0.0f;
                        }
                    }
                }
                for (int idx_sample = 0; idx_sample < options.num_samples; idx_sample++) {
                    const Array2 rnd = options.num_samples_primary_edge >= 0 ? Array2(sampler.next1D(), sampler.next1D()) : Array2(0.5f, 0.5f);

                    SpectrumAD tmp = pixelColorAD(scene, options, &sampler, static_cast<Float>(ix + rnd.x()), static_cast<Float>(iy + rnd.y()), tmp_temp_hist); 
                    bool val_valid = std::isfinite(tmp.val[0]) && std::isfinite(tmp.val[1]) && std::isfinite(tmp.val[2]) && tmp.val.minCoeff() >= 0.0f;
                    Float tmp_val = tmp.der.abs().maxCoeff();
                    bool der_valid = std::isfinite(tmp_val) && tmp_val < options.grad_threshold;
                    if ( val_valid && der_valid ) {
                        pixel_val += tmp;
                    } else {
                        omp_set_lock(&messageLock);
                        if (!val_valid)
                            std::cerr << std::scientific << std::setprecision(2) << "\n[WARN] Invalid path contribution: [" << tmp.val << "]" << std::endl;
                        if (!der_valid)
                            std::cerr << std::scientific << std::setprecision(2) << "\n[WARN] Rejecting large gradient: [" << tmp.der << "]" << std::endl;
                        omp_unset_lock(&messageLock);
                    }
                }
                pixel_val /= options.num_samples;
                // The corresponding implementation for transient is in bdptAD::pixelColorAD
                // as assigning to ptr<float> temp_hist parameter.
                rendered_image[idx_pixel*3    ] = static_cast<float>(pixel_val.val(0));
                rendered_image[idx_pixel*3 + 1] = static_cast<float>(pixel_val.val(1));
                rendered_image[idx_pixel*3 + 2] = static_cast<float>(pixel_val.val(2));
                for ( int ch = 1; ch <= nder; ++ch ) {
                    int offset = (ch*num_pixels + idx_pixel)*3;
                    rendered_image[offset    ] = static_cast<float>((pixel_val.grad(ch - 1))(0));
                    rendered_image[offset + 1] = static_cast<float>((pixel_val.grad(ch - 1))(1));
                    rendered_image[offset + 2] = static_cast<float>((pixel_val.grad(ch - 1))(2));
                }
            }

            if ( !options.quiet ) {
                omp_set_lock(&messageLock);
                progressIndicator(Float(++finished_block)/num_block);
                omp_unset_lock(&messageLock);
            }
        }
        if ( !options.quiet )
            std::cout << "\nDone in " << duration_cast<seconds>(high_resolution_clock::now() - _start).count() << " seconds." << std::endl;
    }

}
