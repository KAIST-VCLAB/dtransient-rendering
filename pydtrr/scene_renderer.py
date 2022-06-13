import dtrr
import torch
import time

print_timing = True
def serialize_scene(scene, use_hierarchy = False):
    """
        Given a Pydtrr scene, convert it to a linear list of argument,
        so that we can use it in PyTorch.
    """
    cam = scene.camera
    num_shapes = len(scene.shapes)
    num_bsdfs = len(scene.bsdfs)
    num_lights = len(scene.area_lights)
    num_medium = len(scene.mediums)
    num_phases = len(scene.phases)
    for light_id, light in enumerate(scene.area_lights):
        scene.shapes[light.shape_id].light_id = light_id
    args = []
    args.append(num_shapes)
    args.append(num_bsdfs)
    args.append(num_lights)
    args.append(num_medium)
    args.append(num_phases)

    # appending None means derivatives. See SceneTransform.__init__
    args.append(cam.cam_to_world)
    args.append(cam.cam_to_ndc)
    args.append(cam.clip_near)
    args.append(cam.resolution)
    args.append(cam.med_id)
    args.append(None)
    args.append(cam.crop_rect)
    args.append(cam.duration)
    args.append(cam.tresolution)
    args.append(cam.tstart)
    args.append(cam.tresolution_light)

    for shape in scene.shapes:
        args.append(shape.vertices)
        args.append(shape.indices)
        args.append(shape.uvs)
        args.append(shape.normals)
        args.append(shape.bsdf_id)
        args.append(shape.light_id)
        args.append(shape.med_ext_id)
        args.append(shape.med_int_id)
        args.append(None)

    for bsdf in scene.bsdfs:
        args.append(bsdf.type)
        if bsdf.type == 'diffuse':
            args.append(bsdf.diffuse_reflectance)
            args.append(None)
        elif bsdf.type == 'texturediffuse':
            args.append(bsdf.texture_diffuse_reflectance)
            args.append(None)
        elif bsdf.type == 'null':
            pass
        elif bsdf.type == 'phong':
            args.append(bsdf.diffuse_reflectance)
            args.append(bsdf.specular_reflectance)
            args.append(bsdf.exponent)
            args.append(None)
            args.append(None)
            args.append(None)
        elif bsdf.type == 'roughdielectric':
            args.append(bsdf.alpha)
            args.append(bsdf.intIOR)
            args.append(bsdf.extIOR)
            args.append(bsdf.spectrum)
            args.append(None)
            args.append(None)
            args.append(None)
        elif bsdf.type == 'roughconductor':
            args.append(bsdf.alpha)
            args.append(bsdf.k)
            args.append(bsdf.eta)
            args.append(None)
            args.append(None)
            args.append(None)
        elif bsdf.type == 'twosided':
            args.append(bsdf.nested)
        else:
            raise
    for light in scene.area_lights:
        args.append(light.type)
        if light.type == 'area_light':
            args.append(light.shape_id)
            args.append(light.intensity)
            args.append(None)
        elif light.type == 'area_lightEx':
            args.append(light.shape_id)
            args.append(light.intensity)
            args.append(light.kappa)
            args.append(None)
            args.append(None)
        else:
            raise

    for medium in scene.mediums:
        args.append(medium.type)
        if medium.type == 'homogeneous':
            args.append(medium.sigma_t)
            args.append(medium.albedo)
            args.append(medium.phase_id)
            args.append(None)
            args.append(None)
        elif medium.type == 'heterogeneous':
            args.append(medium.fn_density)
            args.append(medium.albedo)
            args.append(medium.to_world)
            args.append(medium.scalar)
            args.append(medium.phase_id)
            args.append(None)
            args.append(None)
            args.append(None)
            args.append(None)
        else:
            raise

    for phase in scene.phases:
        args.append(phase.type)
        if phase.type == 'isotropic':
            pass
        elif phase.type == 'hg':
            args.append(phase.g)
            args.append(None)
        else:
            raise

    args.append(None)
    args.append(use_hierarchy)
    return args

def build_scene(options, *args):
    num_shapes = args[0]
    num_bsdfs  = args[1]
    num_lights = args[2]
    num_medium = args[3]
    num_phases = args[4]

    cam_to_world = args[5]
    assert(cam_to_world.is_contiguous())
    cam_to_ndc   = args[6]
    clip_near    = args[7]
    resolution   = args[8]
    cam_med_id   = args[9]
    cam_vel      = args[10]
    rect         = args[11]
    duration = args[12]
    tresolution = args[13]
    tstart = args[14]
    tresolution_light = args[15]
    if cam_vel is not None:
        assert(cam_vel.is_contiguous())
    camera = dtrr.CameraTransient(resolution[0], resolution[1],
                            dtrr.float_ptr(cam_to_world.data_ptr()),
                            dtrr.float_ptr(cam_to_ndc.data_ptr()),
                            clip_near, cam_med_id,
                            dtrr.float_ptr(cam_vel.data_ptr() if cam_vel is not None else 0),
                            duration,
                            tresolution,
                            tstart,
                            tresolution_light)
    camera.set_rect(rect[0], rect[1], rect[2], rect[3])

    current_index = 16 
    shapes = []
    for i in range(num_shapes):
        vertices    = args[current_index]
        indices     = args[current_index + 1]
        uvs         = args[current_index + 2]
        normals     = args[current_index + 3]
        bsdf_id     = args[current_index + 4]
        light_id    = args[current_index + 5]
        med_ext_id  = args[current_index + 6]
        med_int_id  = args[current_index + 7]
        shape_vel   = args[current_index + 8]
        assert(vertices.is_contiguous())
        assert(indices.is_contiguous())
        if uvs is not None:
            assert(uvs.is_contiguous())
        if normals is not None:
            assert(normals.is_contiguous())
        if shape_vel is not None:
            assert( shape_vel.is_contiguous() )
        shapes.append(dtrr.Shape(\
            dtrr.float_ptr(vertices.data_ptr()),
            dtrr.int_ptr(indices.data_ptr()),
            dtrr.float_ptr(uvs.data_ptr() if uvs is not None else 0),
            dtrr.float_ptr(normals.data_ptr() if normals is not None else 0),
            int(vertices.shape[0]),
            int(indices.shape[0]),
            light_id, bsdf_id, med_int_id, med_ext_id,
            dtrr.float_ptr(shape_vel.data_ptr() if shape_vel is not None else 0)))
        current_index += 9
    bsdfs = []
    bsdfs1 = []

    for i in range(num_bsdfs):
        if args[current_index] == 'null':
            bsdfs.append(dtrr.BSDF_null())
            current_index += 1
        elif args[current_index] == 'twosided':
            if args[current_index + 1].type == 'texturediffuse':
                width = args[current_index + 1].width
                height = args[current_index + 1].height
                texture_diffuse_reflectance = args[current_index + 1].texture_diffuse_reflectance
                nested = dtrr.BSDF_texturedDiffuse(width, height, dtrr.float_ptr(texture_diffuse_reflectance.data_ptr()))
                bsdfs1.append(nested)
                bsdfs.append(dtrr.BSDF_twosided(nested))
            elif args[current_index + 1].type == 'diffuse':
                diffuse_reflectance = args[current_index + 1].diffuse_reflectance
                vec_reflectance     = dtrr.Spectrum3f(diffuse_reflectance[0], diffuse_reflectance[1], diffuse_reflectance[2])
                nested = dtrr.BSDF_diffuse(vec_reflectance)
                bsdfs1.append(nested)
                bsdfs.append(dtrr.BSDF_twosided(nested))
            else:
                print("[ERROR]: Unsupported twoside BSDF.")
                assert(False)
            current_index += 2
        elif args[current_index] == 'texturediffuse':
            texture_diffuse_reflectance = args[current_index + 1]
            bsdfs.append(dtrr.BSDF_texturedDiffuse(texture_diffuse_reflectance.shape[0], texture_diffuse_reflectance.shape[1], dtrr.float_ptr(texture_diffuse_reflectance.data_ptr())))
            current_index += 3
        elif args[current_index] == 'diffuse':
            diffuse_reflectance = args[current_index + 1]
            vec_reflectance     = dtrr.Spectrum3f(diffuse_reflectance[0], diffuse_reflectance[1], diffuse_reflectance[2])
            default             = args[current_index + 2] is None
            if default:
                bsdfs.append(dtrr.BSDF_diffuse(vec_reflectance))
            else:
                d_reflectance = args[current_index + 2]
                assert(d_reflectance.is_contiguous())
                bsdfs.append(dtrr.BSDF_diffuse(vec_reflectance, dtrr.float_ptr(d_reflectance.data_ptr())))
            current_index += 3
        elif args[current_index] == 'phong':
            diffuse_reflectance  = args[current_index + 1]
            specular_reflectance = args[current_index + 2]
            exponent = args[current_index + 3]
            vec_kd  = dtrr.Spectrum3f(diffuse_reflectance[0], diffuse_reflectance[1], diffuse_reflectance[2])
            vec_ks  = dtrr.Spectrum3f(specular_reflectance[0], specular_reflectance[1], specular_reflectance[2])
            default = args[current_index + 4] is None
            if default:
                bsdfs.append(dtrr.BSDF_Phong(vec_kd, vec_ks, exponent))
            else:
                d_diffuse = args[current_index + 4]
                assert( d_diffuse.is_contiguous() )
                d_specular = args[current_index + 5]
                assert( (d_specular is not None) and d_specular.is_contiguous() )
                d_exponent = args[current_index + 6]
                assert( (d_exponent is not None) and d_exponent.is_contiguous() )
                bsdfs.append(dtrr.BSDF_Phong(vec_kd, vec_ks, exponent,
                                                dtrr.float_ptr(d_diffuse.data_ptr()),
                                                dtrr.float_ptr(d_specular.data_ptr()),
                                                dtrr.float_ptr(d_exponent.data_ptr())))
            current_index += 7
        elif args[current_index] == 'roughdielectric':
            alpha   = args[current_index + 1]
            intIOR  = args[current_index + 2]
            extIOR  = args[current_index + 3]
            spectrum = args[current_index + 4]
            vec_spectrum = dtrr.Spectrum3f(spectrum[0], spectrum[1], spectrum[2])
            default = args[current_index + 5] is None 
            if default:
                bsdfs.append(dtrr.BSDF_roughdielectric(alpha, intIOR, extIOR, vec_spectrum)) 
            else:
                d_alpha = args[current_index + 5] 
                assert( d_alpha.is_contiguous() )
                d_eta = args[current_index + 6] 
                assert( (d_eta is not None) and d_eta.is_contiguous())
                d_spectrum = args[current_index + 7]
                assert( (d_spectrum is not None) and d_spectrum.is_contiguous() )
                bsdfs.append(dtrr.BSDF_roughdielectric(alpha, intIOR, extIOR, vec_spectrum,
                                                           *[dtrr.float_ptr(x.data_ptr()) for x in [d_alpha, d_eta, d_spectrum]]))
            current_index += 8 
        elif args[current_index] == 'roughconductor':
            alpha   = args[current_index + 1]
            k       = args[current_index + 2]
            vec_k   = dtrr.Spectrum3f(k[0], k[1], k[2])
            eta     = args[current_index + 3]
            vec_eta = dtrr.Spectrum3f(eta[0], eta[1], eta[2])
            default = args[current_index + 4] is None
            if default:
                bsdfs.append(dtrr.BSDF_roughconductor(alpha, vec_eta, vec_k))
            else:
                d_alpha = args[current_index + 4]
                assert( d_alpha.is_contiguous() )
                d_k = args[current_index + 5]
                assert( (d_k is not None) and d_k.is_contiguous() )
                d_eta = args[current_index + 6]
                assert( (d_eta is not None) and d_eta.is_contiguous())
                bsdfs.append(dtrr.BSDF_roughconductor(alpha, vec_eta, vec_k,
                                                          dtrr.float_ptr(d_alpha.data_ptr()),
                                                          dtrr.float_ptr(d_eta.data_ptr()),
                                                          dtrr.float_ptr(d_k.data_ptr())))
            current_index += 7
        else:
            raise
    area_lights = []
    for i in range(num_lights):
        if args[current_index] == 'area_light':
            shape_id    = args[current_index + 1]
            intensity   = dtrr.Spectrum3f(args[current_index + 2][0], args[current_index + 2][1], args[current_index + 2][2])
            default     = args[current_index + 3] is None
            if default:
                area_lights.append(dtrr.AreaLight(shape_id, intensity))
            else:
                d_intensity = args[current_index + 3]
                area_lights.append(dtrr.AreaLight(shape_id, intensity, dtrr.float_ptr(d_intensity.data_ptr())))
            current_index += 4
        else:
            shape_id    = args[current_index + 1]
            intensity   = dtrr.Spectrum3f(args[current_index + 2][0], args[current_index + 2][1], args[current_index + 2][2])
            kappa       = args[current_index + 3]
            default     = args[current_index + 4] is None
            if default:
                area_lights.append(dtrr.AreaLightEx(shape_id, intensity, kappa))
            else:
                d_intensity = args[current_index + 4]
                assert( d_intensity.is_contiguous() )
                d_kappa = args[current_index + 5]
                assert( (d_kappa is not None) and d_kappa.is_contiguous() )
                area_lights.append(dtrr.AreaLightEx(shape_id, intensity, kappa, dtrr.float_ptr(d_intensity.data_ptr()), dtrr.float_ptr(d_kappa.data_ptr())))
            current_index += 6

    mediums = []
    for i in range(num_medium):
        if args[current_index] == 'homogeneous':
            sigma_t     = args[current_index + 1]
            albedo      = args[current_index + 2]
            vec_albedo  = dtrr.Spectrum3f(albedo[0], albedo[1], albedo[2])
            phase_id    = args[current_index + 3]
            default     = args[current_index + 4] is None
            if default:
                mediums.append(dtrr.Homogeneous(sigma_t, vec_albedo, phase_id))
            else:
                d_sigmaT = args[current_index + 4]
                assert( d_sigmaT.is_contiguous() )
                d_albedo = args[current_index + 5]
                assert( (d_albedo is not None) and d_albedo.is_contiguous() )
                mediums.append(dtrr.Homogeneous(sigma_t, vec_albedo, phase_id, dtrr.float_ptr(d_sigmaT.data_ptr()),
                                                                                  dtrr.float_ptr(d_albedo.data_ptr())))
            current_index += 6
        elif args[current_index] == 'heterogeneous':
            fn_density  = args[current_index + 1]
            albedo      = args[current_index + 2]
            vol_albedo  = isinstance(albedo, str)
            to_world    = args[current_index + 3]
            scalar      = args[current_index + 4]
            phase_id    = args[current_index + 5]
            default     = args[current_index + 6] is None
            if default:
                if vol_albedo:
                    mediums.append(dtrr.Heterogeneous(fn_density, dtrr.float_ptr(to_world.data_ptr()), float(scalar), albedo, phase_id))
                else:
                    vec_albedo  = dtrr.Spectrum3f(albedo[0], albedo[1], albedo[2])
                    mediums.append(dtrr.Heterogeneous(fn_density, dtrr.float_ptr(to_world.data_ptr()), float(scalar), vec_albedo, phase_id))
            else:
                d_albedo        = args[current_index + 6]
                assert( d_albedo.is_contiguous() )
                d_scalar        = args[current_index + 7]
                assert( (d_scalar is not None) and d_scalar.is_contiguous() )
                vec_translate   = args[current_index + 8]
                assert( (vec_translate is not None) and vec_translate.is_contiguous() )
                vec_rotate      = args[current_index + 9]
                assert( (vec_rotate is not None) and vec_rotate.is_contiguous() )
                if vol_albedo:
                    mediums.append(dtrr.Heterogeneous(fn_density, dtrr.float_ptr(to_world.data_ptr()), float(scalar), albedo, phase_id,
                                                     dtrr.float_ptr(vec_translate.data_ptr()),
                                                     dtrr.float_ptr(vec_rotate.data_ptr()),
                                                     dtrr.float_ptr(d_scalar.data_ptr())))
                else:
                    vec_albedo  = dtrr.Spectrum3f(albedo[0], albedo[1], albedo[2])
                    mediums.append(dtrr.Heterogeneous(fn_density, dtrr.float_ptr(to_world.data_ptr()), float(scalar), vec_albedo, phase_id,
                                                     dtrr.float_ptr(vec_translate.data_ptr()),
                                                     dtrr.float_ptr(vec_rotate.data_ptr()),
                                                     dtrr.float_ptr(d_scalar.data_ptr()),
                                                     dtrr.float_ptr(d_albedo.data_ptr())))
            current_index += 10
    phases = []
    for i in range(num_phases):
        if args[current_index] == 'isotropic':
            phases.append(dtrr.Isotropic())
            current_index += 1
        elif args[current_index] == 'hg':
            g = args[current_index + 1]
            default  = args[current_index + 2] is None
            if default:
                phases.append(dtrr.HG(g));
            else:
                d_g  = args[current_index + 2]
                assert( d_g.is_contiguous() )
                phases.append(dtrr.HG(g, dtrr.float_ptr(d_g.data_ptr())));
            current_index += 3
        else:
            raise

    use_hierarchy = args[-1]
    scene = dtrr.Scene(camera, shapes, bsdfs, area_lights, phases, mediums, use_hierarchy)

    if args[current_index] is not None:
        print(args[current_index])
        scene.initEdges(dtrr.float_ptr(args[current_index].data_ptr()))
    if rect[2] == -1 or rect[3] == -1:
        rendered_image = torch.zeros(dtrr.nder + 1, resolution[1], resolution[0], 3)
        rendered_transient = torch.zeros(dtrr.nder + 1, resolution[1], resolution[0], duration, 3)
    else:
        rendered_image = torch.zeros(dtrr.nder + 1, rect[3], rect[2], 3)
        rendered_transient = torch.zeros(dtrr.nder + 1, rect[3], rect[2], duration, 3)
    return (camera, shapes, bsdfs + bsdfs1, area_lights, phases, mediums, scene), rendered_image, rendered_transient

def render(integrator, options, sceneData, rendered_image, rendered_transient = None):
    start = time.time()
    if rendered_transient == None:
        integrator.render(sceneData[-1], options, dtrr.float_ptr(rendered_image.data_ptr()))
    else:
        integrator.render(sceneData[-1], options, dtrr.float_ptr(rendered_image.data_ptr()), dtrr.float_ptr(rendered_transient.data_ptr()))
    time_elapsed = time.time() - start
    if print_timing and not options.quiet:
        hours, rem = divmod(time_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        hours = int(hours)
        minutes = int(minutes)
        if hours > 0:
            print("Total render time: {:0>2}h {:0>2}m {:0>2.2f}s".format(hours, minutes, seconds))
        elif minutes > 0:
            print("Total render time: {:0>2}m {:0>2.2f}s".format(minutes, seconds))
        else:
            print("Total render time: {:0>2.2f}s".format(seconds))
    return time_elapsed


def render_scene(integrator, options, *args):
    sceneData, rendered_image, rendered_transient = build_scene(options, *(args))
    render(integrator, options, sceneData, rendered_image, rendered_transient)
    return rendered_image, rendered_transient


def render_scene_timed(integrator, options, *args):
    #print("render_scene_timed started.")
    sceneData, rendered_image, rendered_transient = build_scene(options, *(args))
    time_elapsed = render(integrator, options, sceneData, rendered_image, rendered_transient)
    return rendered_image, rendered_transient, time_elapsed
