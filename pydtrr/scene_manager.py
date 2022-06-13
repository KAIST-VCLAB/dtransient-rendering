import numpy as np
import torch
import dtrr
from dtrr import nder
import copy
from math import cos, sin, sqrt
import math
import struct
import pydtrr.transform as transform
from scipy.ndimage.filters import gaussian_filter
import pydtrr.scene_renderer

class SceneTransform:
    def __init__(self, type_name, *args):
        self.type_name = type_name
        if type_name == "CAMERA_TRANSLATE":
            self.vec_vel = args[0]
        elif type_name == "SHAPE_TRANSLATE":
            self.vec_vel = args[0]
            self.shape_id = args[1]
        elif type_name == "VERTEX_TRANSLATE":
            self.vec_vel = args[0]
            self.shape_id = args[1]
            self.vert_id  = args[2]
        elif type_name == "CAMERA_ROTATE":
            self.vec_vel = args[0]
        elif type_name == "SHAPE_ROTATE":
            self.vec_vel = args[0]
            self.shape_id = args[1]
        elif type_name == "SHAPE_GLOBAL_ROTATE":
            self.vec_vel = args[0]
            self.shape_id = args[1]
        elif type_name == "SHAPE_SCALE":
            assert (type(args[0]) is float) and (args[0] > 0)
            self.scl = args[0]
            self.shape_id = args[1]
        elif type_name == "SHAPE_BLENDING":
            self.shape_id = args[0]
            mesh_list = pydtrr.load_obj(args[1])
            self.vert_target = mesh_list[0].vertices.cpu()
            normals = mesh_list[0].normals
            if normals is not None:
                self.norm_target = normals.cpu()
            else:
                self.norm_target = None
        elif type_name == "BSDF_VARY":
            self.bsdf_type = args[0]
            self.bsdf_id   = args[1]
            if self.bsdf_type == "diffuse":
                self.d_reflectance = args[2]
            elif self.bsdf_type == "twosided":
                self.nested = args[2]
            elif self.bsdf_type == "phong":
                self.d_diffuse  = args[2]
                self.d_specular = args[3]
                self.d_exponent = args[4]
            elif self.bsdf_type == "roughdielectric":
                self.d_alpha    = args[2]
                self.d_eta      = args[3]
                self.d_spectrum = args[4] 
            elif self.bsdf_type == "roughconductor":
                self.d_alpha    = args[2]
                self.d_k        = args[3]
                self.d_eta      = args[4]
        elif type_name == "MEDIUM_VARY":
            self.med_type = args[0]
            self.med_id   = args[1]
            if self.med_type == 'homogeneous':
                self.d_sigmaT = args[2]
                self.d_albedo = args[3]
            elif self.med_type == 'heterogeneous':
                self.vec_translate = args[2]
                self.vec_rotate    = args[3]
                self.d_albedo      = args[4]
                self.d_scalar      = args[5]
        elif type_name == "PHASE_VARY":
            self.phase_type = args[0]
            assert(self.phase_type == "hg")
            self.phase_id   = args[1]
            self.d_g        = args[2]
        elif type_name == "EMITTER_VARY":
            self.emitter_type = args[0]
            assert(self.emitter_type == "area_lightEx")
            self.emitter_id  = args[1]
            self.d_intensity = args[2]
            self.d_kappa     = args[3]
        elif type_name == "EMITTER_INVSCALE":
            self.emitter_type = args[0]
            assert(self.emitter_type == "area_light")
            self.emitter_id  = args[1]
            assert (type(args[2]) is float) and (args[2] > 0)
            self.scl         = args[2]
        else:
            print("Transform type [%s] not supported" % type_name)
            assert(False)

    def determine_range(self, init_val, tmin, tmax):
        if self.type_name == "BSDF_VARY":
            if self.bsdf_type == "diffuse":
                for ch in range(3):
                    if self.d_reflectance[ch] == 0.0:
                        continue
                    a = -init_val[ch]/self.d_reflectance[ch]
                    b = (1.0-init_val[ch])/self.d_reflectance[ch]
                    tmin = max(tmin, a if self.d_reflectance[ch] > 0 else b)
                    tmax = min(tmax, b if self.d_reflectance[ch] > 0 else a)
            elif self.bsdf_type == "phong":
                diffuse = init_val[0]
                specular = init_val[1]
                exponent = init_val[2]
                for ch in range(3):
                    lim_diffuse = -diffuse[ch]/self.d_diffuse[ch]
                    lim_specular = -specular[ch]/self.d_specular[ch]
                    if self.d_diffuse[ch] > 0:
                        tmin = max(tmin, lim_diffuse)
                    elif self.d_diffuse[ch] < 0:
                        tmax = min(tmax, lim_diffuse)
                    if self.d_specular[ch] > 0:
                        tmin = max(tmin, lim_specular)
                    elif self.d_specular[ch] < 0:
                        tmax = min(tmax, lim_specular)
                if self.d_exponent > 0.0:
                    tmin = max(tmin, -exponent/self.d_exponent)
                elif self.d_exponent < 0.0:
                    tmax = min(tmin, -exponent/self.d_exponent)
            elif self.bsdf_type == "roughdielectric":
                alpha_min = 0.01
                alpha_max = 0.3
                a = (alpha_min - init_val[0])/self.d_alpha
                b = (alpha_max - init_val[0])/self.d_alpha
                if self.d_alpha > 0.0:
                    tmin = max(tmin, a)
                    tmax = min(tmin, b)
                elif self.d_alpha < 0.0:
                    tmin = max(tmin, b)
                    tmax = min(tmax, a)
                if self.d_eta > 0.0:
                    tmin = max(tmin, -init_val[1]/self.d_eta)
                elif self.d_eta < 0.0:
                    tmax = min(tmax, -init_val[1]/self.d_eta)
                spectrum = init_val[2]
                for ch in range(3):
                    lim_spectrum = -spectrum[ch]/self.d_spectrum[ch]
                    if self.d_spectrum[ch] > 0:
                        tmin = max(tmin, lim_spectrum)
                    elif self.d_spectrum[ch] < 0:
                        tmax = min(tmax, lim_spectrum)
            elif self.bsdf_type == "roughconductor":
                alpha_min = 0.01
                alpha_max = 0.3
                a = (alpha_min - init_val[0])/self.d_alpha
                b = (alpha_max - init_val[0])/self.d_alpha
                if self.d_alpha > 0.0:
                    tmin = max(tmin, a)
                    tmax = min(tmin, b)
                elif self.d_alpha < 0.0:
                    tmin = max(tmin, b)
                    tmax = min(tmax, a)
                if self.d_k > 0.0:
                    tmin = max(tmin, -init_val[1]/self.d_eta)
                elif self.d_k < 0.0:
                    tmax = min(tmax, -init_val[1]/self.d_eta)
                if self.d_eta > 0.0:
                    tmin = max(tmin, -init_val[2]/self.d_eta)
                elif self.d_eta < 0.0:
                    tmax = min(tmax, -init_val[2]/self.d_eta)
                print(tmin, tmax)
        elif self.type_name == "MEDIUM_VARY":
            if self.med_type == "homogeneous":
                sigma_t = init_val[0]
                albedo  = init_val[1]
                for ch in range(3):
                    if self.d_albedo[ch] == 0.0:
                        continue
                    a = -albedo[ch]/self.d_albedo[ch]
                    b = (1.0-albedo[ch])/self.d_albedo[ch]
                    tmin = max(tmin, a if self.d_albedo[ch] > 0 else b)
                    tmax = min(tmax, b if self.d_albedo[ch] > 0 else a)
                if self.d_sigmaT > 0.0:
                    tmin = max(tmin, -sigma_t/self.d_sigmaT)
                elif self.d_sigmaT < 0.0:
                    tmax = min(tmax, -sigma_t/self.d_sigmaT)
            elif self.med_type == "heterogeneous":
                albedo = init_val[0]
                scalar = init_val[1]
                for ch in range(3):
                    if self.d_albedo[ch] == 0.0:
                        continue
                    a = -albedo[ch]/self.d_albedo[ch]
                    b = (1.0-albedo[ch])/self.d_albedo[ch]
                    tmin = max(tmin, a if self.d_albedo[ch] > 0 else b)
                    tmax = min(tmax, b if self.d_albedo[ch] > 0 else a)
                if self.d_scalar > 0.0:
                    tmin = max(tmin, -scalar/self.d_scalar)
                elif self.d_scalar < 0.0:
                    tmax = min(tmax, -scalar/self.d_scalar)
        assert ( tmax > tmin )
        return tmin, tmax

# scene_args:
    # num_shapes
    # num_bsdfs
    # num_lights
    # num_mediums
    # num_phases
    # [camera] cam_to_world (4x4)                       -- requires update
    # [camera] cam_to_ndc (3x3)                         -- requires update
    # [camera] clip_near
    # [camera] resolution (2)
    # [camera] medium_id
    # [camera] translate_vel & rotate_vel               -- requires update
    # [camera] rect (x,y,w,h)
    # -- Iterate through all shapes
        # [shape i] vertices                            -- requires update
        # [shape i] indices
        # [shape i] uvs (None if not exist)
        # [shape i] normals (None if not exist)         -- requires update
        # [shape i] bsdf_id
        # [shape i] light_id
        # [shape i] med_id (ext)
        # [shape i] med_id (int)
        # [shape i] vert_vel                           -- requires update
    # -- Iterate through all bsdfs
        # [bsdf i] bsdf_type
        # IF bsdf_type == "diffuse"
            # [bsdf i] reflectance
            # [bsdf i] d_reflectance
        # IF bsdf_type == "null"
        # IF bsdf_type == "phong"
            # [bsdf i] diffuse_reflectance
            # [bsdf i] specular_reflectance
            # [bsdf i] exponent
            # [bsdf i] d_diffuse_reflectance
            # [bsdf i] d_specular_reflectance
            # [bsdf i] d_exponent
        # IF bsdf_type == "roughdielectric"
            # [bsdf i] alpha
            # [bsdf i] intIOR
            # [bsdf i] extIOR
            # [bsdf i] d_alpha
            # [bsdf i] d_eta
        # IF bsdf_type == "roughconductor"
            # [bsdf i] alpha
            # [bsdf i] k
            # [bsdf i] eta
            # [bsdf i] d_alpha
            # [bsdf i] d_k
            # [bsdf i] d_eta
    # -- Iterate through all emitters
        # [emitter i] emitter_type
        # IF emitter_type == "area_light":
            # [emitter i] shape_id
            # [emitter i] intensity
            # [emitter i] d_intensity
        # IF emitter_type == "area_lightEx":
            # [emitter i] shape_id
            # [emitter i] intensity
            # [emitter i] kappa
            # [emitter i] d_intensity
            # [emitter i] d_kappa
    # -- Iterate through all mediums
        # [medium i] med_type
        # IF med_type == "homogeneous":
            # [medium i] sigma_t
            # [medium i] albedo
            # [medium i] phase_id
            # [medium i] d_sigmaT
            # [medium i] d_albedo
        # IF med_type == "heterogeneous":
            # [medium i] fn_density
            # [medium i] albedo (tensor/string)
            # [medium i] to_world
            # [medium i] scalar
            # [medium i] phase_id
            # [medium i] d_albedo
            # [medium i] d_scalar
            # [medium i] vec_translate
            # [medium i] vec_rotate
    # -- Iterate through all phases
        # [phase i] phase_type
        # IF phase_type == "isotropic":
        # IF phase_type == "hg":
            # [phase i] g
            # [phase i] d_g
    # Edge sampling weight for different shapes

class SceneManager:
    def __init__(self, args, transforms, initial_param = torch.tensor([0]*nder, dtype=torch.float), check_range = False):
        # determine the offsets
        self.offsets = {"camera": 5, "shape": [], "bsdf": [], "medium": [], "phase": [], "emitter": []}
        self.num_shapes = args[0]
        self.num_bsdfs  = args[1]
        self.num_lights = args[2]
        self.num_medium = args[3]
        self.num_phases = args[4]
        current_index = 16 
        for i in range(self.num_shapes):
            self.offsets["shape"].append(current_index)
            current_index += 9
        for i in range(self.num_bsdfs):
            self.offsets["bsdf"].append(current_index)
            if args[current_index] == 'null':
                current_index += 1
            elif args[current_index] == 'diffuse':
                current_index += 3
            elif args[current_index] == 'texturediffuse':
                current_index += 3
            elif args[current_index] == 'twosided':
                current_index += 3
            elif args[current_index] == 'phong':
                current_index += 7
            elif args[current_index] == 'roughdielectric':
                current_index += 8 
            elif args[current_index] == 'roughconductor':
                current_index += 7
        for i in range(self.num_lights):
            self.offsets["emitter"].append(current_index)
            if args[current_index] == 'area_light':
                current_index += 4
            else:
                current_index += 6
        for i in range(self.num_medium):
            self.offsets["medium"].append(current_index)
            if args[current_index] == 'homogeneous':
                current_index += 6
            elif args[current_index] == 'heterogeneous':
                current_index += 10
        for i in range(self.num_phases):
            self.offsets["phase"].append(current_index)
            if args[current_index] == 'isotropic':
                current_index += 1
            elif args[current_index] == 'hg':
                current_index += 3
        self.offsets["weight"] = current_index

        self.args_0 = copy.deepcopy(args)
        # set centers for the shapes
        self.shape_centers_0 = []
        for ishape in range(self.num_shapes):
            offset = self.offsets["shape"][ishape]
            num_verts = self.args_0[offset].size(0)
            self.shape_centers_0.append(self.args_0[offset].sum(0)/num_verts)
        # set centers for volumes
        self.volume_centers_0 = []
        for imed in range(self.num_medium):
            offset = self.offsets["medium"][imed]
            if self.args_0[offset] == "homogeneous":
                self.volume_centers_0.append(None)
            elif self.args_0[offset] == "heterogeneous":
                fn = self.args_0[offset + 1]
                to_world = self.args_0[offset + 3]
                with open(fn, 'rb') as fin:
                    assert(fin.read(3) == b'VOL')
                    assert(fin.read(1) == b'\x03')
                    assert struct.unpack('I', fin.read(4)) == (1,)
                    sz = struct.unpack('3I', fin.read(12))
                    ch = struct.unpack('I', fin.read(4))
                    pMin = np.array(struct.unpack('3f', fin.read(12)))
                    pMax = np.array(struct.unpack('3f', fin.read(12)))
                    aabb_center = (pMax + pMin) * 0.5
                aabb_center = torch.tensor([aabb_center[0], aabb_center[1], aabb_center[2], 1], dtype=torch.float)
                aabb_center = torch.mm(to_world ,aabb_center.unsqueeze(1)).squeeze(1)
                self.volume_centers_0.append(aabb_center[0:3])
        self.transforms = transforms
        self.ranges = []
        if not check_range:
            for ider in range(nder):
                self.ranges.append([-math.inf, math.inf])
        else:
            self.determine_range()
        assert(nder == len(transforms))
        self.args = copy.deepcopy(self.args_0)
        self.param_0 = initial_param
        self.set_arguments(torch.tensor([0]*nder, dtype=torch.float))
        self.grid_guide_direct   = torch.zeros([1, 1, 1], dtype=torch.float)
        self.grid_guide_indirect = torch.zeros([1, 1, 1], dtype=torch.float)
        self.shape_weights = torch.ones(self.num_shapes, dtype=torch.float)
        self.cam2world_list = None

    def set_multicam(self, cam2world_list):
        self.cam2world_list = cam2world_list
        print("%d cameras are setup for the SceneManager" % len(self.cam2world_list))


    def set_direct_guiding(self, direct_param, integrator, opts, quiet = False):
        guiding_epsilon = 1e-4
        sceneData, out, out_transient = pydtrr.build_scene(opts, *(self.args))
        scene = sceneData[-1]
        self.grid_guide_direct = torch.zeros(direct_param[0 : -1], dtype=torch.float)
        guiding_option = dtrr.GuidingOptions(0, direct_param)
        guiding_option.quiet = quiet
        integrator.preprocess(scene, 1, guiding_option, dtrr.float_ptr(self.grid_guide_direct.data_ptr()))
        self.grid_guide_direct = torch.from_numpy(gaussian_filter(self.grid_guide_direct.numpy(), sigma=0.5) + guiding_epsilon)

    def set_indirect_guiding(self, guide_type, indirect_param, num_cam_paths, num_light_paths, radius, integrator, opts, quiet=False):
        # [guide_type] 0:Direct Guiding, 1:Indirect Guiding (KNN), 2:Indirect Guiding (Radius Search), 3:Indirect Guiding (Old)
        
        guiding_epsilon = 1e-4
        sceneData, out, out_transient = pydtrr.build_scene(opts, *(self.args))
        scene = sceneData[-1]
        self.grid_guide_indirect = torch.zeros(indirect_param[0 : -1], dtype=torch.float)
        if guide_type == 3:
            guide_option = dtrr.GuidingOptions(guide_type, indirect_param)
        else:
            guide_option = dtrr.GuidingOptions(guide_type, indirect_param, num_cam_paths, num_light_paths, radius)
        guide_option.quiet = quiet
        integrator.preprocess(scene, opts.max_bounces, guide_option, dtrr.float_ptr(self.grid_guide_indirect.data_ptr()))
        self.grid_guide_indirect = torch.from_numpy(gaussian_filter(self.grid_guide_indirect.numpy(), sigma=0.5) + guiding_epsilon)

    def set_edge_weight(self, weights):
        assert(weights.numel() == self.num_shapes)
        self.args[self.offsets["weight"]] = weights.contiguous()

    def print(self):

        print("# BSDF = %d" % self.num_bsdfs)
        for i in range(self.num_bsdfs):
            offset = self.offsets["bsdf"][i]
            if self.args[offset] == 'null':
                print("  [bsdf %d = null]" % i)
            elif self.args[offset] == 'twosided':
                print("  [bsdf is twosided] ")
            elif self.args[offset] == 'diffuse':
                reflectance = self.args[offset + 1]
                print("  [bsdf %d = diffuse] reflectance = (%.2f, %.2f, %.2f)"%(i, reflectance[0], reflectance[1], reflectance[2]))
            elif self.args[offset] == 'phong':
                diffuse  = self.args[offset + 1]
                specular = self.args[offset + 2]
                exponent = self.args[offset + 3]
                print("  [bsdf %d = phong] diffuse = (%.2f, %.2f, %.2f), specular = (%.2f, %.2f, %.2f), exponent = %.2f "
                         % (i, diffuse[0], diffuse[1], diffuse[2], specular[0], specular[1], specular[2], exponent))
            elif self.args[offset] == 'roughdielectric':
                spectrum = self.args[offset + 4]
                print("  [bsdf %d = roughdielectric] alpha = %2.f, intIOR = %.2f, extIOR = %.2f, spectrum = (%.2f, %.2f, %.2f)"
                         % (i, self.args[offset + 1], self.args[offset + 2], self.args[offset + 3], spectrum[0], spectrum[1], spectrum[2]))
            elif self.args[offset] == 'roughconductor':
                print("  [bsdf %d = roughconductor] alpha = %2.f, k = %.2f, eta = %.2f"
                         % (i, self.args[offset + 1], self.args[offset + 2], self.args[offset + 3]))

        print("# Medium = %d" % self.num_medium)
        for i in range(self.num_medium):
            offset = self.offsets["medium"][i]
            if self.args[offset] == 'homogeneous':
                sigma_t = self.args[offset + 1]
                albedo = self.args[offset + 2]
                print("  [medium %d = homogeneous] sigma_t = %.2f, albedo = (%.2f, %.2f, %.2f)" % (i, sigma_t, albedo[0], albedo[1], albedo[2]))
            elif self.args[offset] == 'heterogeneous':
                print("  [medium %d = heterogeneous]")

    # set velocity to zero & set position/parameters to the initial state
    def clean_up(self):
        # camera related
        offset = self.offsets["camera"]
        self.args[offset]       = copy.deepcopy(self.args_0[offset])
        self.args[offset + 5]   = torch.zeros(nder*2, 3)
        # shape related
        for offset in self.offsets["shape"]:
            num_verts = self.args_0[offset].size(0)
            has_normal = self.args_0[offset + 3] is not None
            self.args[offset]       = self.args_0[offset].clone()
            if has_normal:
                self.args[offset + 3]   = self.args_0[offset + 3].clone()
            self.args[offset + 8]   = torch.zeros(nder*2*num_verts, 3, dtype=torch.float) if has_normal else\
                                      torch.zeros(nder*num_verts,   3, dtype=torch.float)
        # bsdf related
        for offset in self.offsets["bsdf"]:
            bsdf_type = self.args_0[offset]
            if bsdf_type == "diffuse":
                self.args[offset + 1] = copy.deepcopy(self.args_0[offset + 1])
                self.args[offset + 2] = copy.deepcopy(self.args_0[offset + 2])
            if bsdf_type == "twosided":
                self.args[offset + 1] = copy.deepcopy(self.args_0[offset + 1])
                self.args[offset + 2] = copy.deepcopy(self.args_0[offset + 2])
            elif bsdf_type == "phong":
                self.args[offset + 1:offset + 4] = copy.deepcopy(self.args_0[offset + 1:offset + 4])
                self.args[offset + 4] = torch.zeros(nder, 3, dtype=torch.float)
                self.args[offset + 5] = torch.zeros(nder, 3, dtype=torch.float)
                self.args[offset + 6] = torch.zeros(nder, 1, dtype=torch.float)
            elif bsdf_type == "roughdielectric":
                self.args[offset + 1: offset + 5] = copy.deepcopy(self.args_0[offset + 1 : offset + 5])
                self.args[offset + 5] = torch.zeros(nder, 1, dtype=torch.float) 
                self.args[offset + 6] = torch.zeros(nder, 1, dtype=torch.float) 
                self.args[offset + 7] = torch.zeros(nder, 3, dtype=torch.float)
            elif bsdf_type == "roughconductor":
                self.args[offset + 1 : offset + 4] = copy.deepcopy(self.args_0[offset + 1 : offset + 4])
                self.args[offset + 4] = torch.zeros(nder, 1, dtype=torch.float)
                self.args[offset + 5] = torch.zeros(nder, 3, dtype=torch.float)
                self.args[offset + 6] = torch.zeros(nder, 3, dtype=torch.float)
        # medium related
        for offset in self.offsets["medium"]:
            med_type = self.args_0[offset]
            if med_type == "homogeneous":
                self.args[offset + 1:offset + 3] = copy.deepcopy(self.args_0[offset + 1:offset + 3])
                self.args[offset + 4] = torch.zeros(nder, 1, dtype=torch.float)
                self.args[offset + 5] = torch.zeros(nder, 3, dtype=torch.float)
            elif med_type == "heterogeneous":
                self.args[offset + 2:offset + 5] = copy.deepcopy(self.args_0[offset + 2:offset + 5])
                self.args[offset + 6] = torch.zeros(nder, 3, dtype=torch.float)
                self.args[offset + 7] = torch.zeros(nder, 1, dtype=torch.float)
                self.args[offset + 8] = torch.zeros(nder, 3, dtype=torch.float)
                self.args[offset + 9] = torch.zeros(nder, 3, dtype=torch.float)
        # phase related
        for offset in self.offsets["phase"]:
            phase_type = self.args_0[offset]
            if phase_type == "hg":
                self.args[offset + 1] = copy.deepcopy(self.args_0[offset + 1])
                self.args[offset + 2] = torch.zeros(nder, 1, dtype=torch.float)
        # emitter related
        for offset in self.offsets["emitter"]:
            light_type = self.args_0[offset]
            if light_type == "area_light":
                self.args[offset + 2] = copy.deepcopy(self.args_0[offset + 2])
                self.args[offset + 3] = torch.zeros(nder, 3, dtype=torch.float)
            elif light_type == "area_lightEx":
                self.args[offset + 2] = copy.deepcopy(self.args_0[offset + 2])
                self.args[offset + 3] = copy.deepcopy(self.args_0[offset + 3])
                self.args[offset + 4] = torch.zeros(nder, 3, dtype=torch.float)
                self.args[offset + 5] = torch.zeros(nder, 1, dtype=torch.float)

    def reset(self, resetGuiding=True):
        self.set_arguments(torch.tensor([0.0]*nder, dtype=torch.float))
        if resetGuiding:
            self.grid_guide_direct   = torch.zeros([1, 1, 1], dtype=torch.float)
            self.grid_guide_indirect = torch.zeros([1, 1, 1], dtype=torch.float)
            self.shape_weights = torch.ones(self.num_shapes, dtype=torch.float)

    def determine_range(self):
        for ivar, Tset in enumerate(self.transforms):
            tmin = -math.inf
            tmax = math.inf
            for T in Tset:
                if T.type_name == "BSDF_VARY":
                    offset = self.offsets["bsdf"][T.bsdf_id] + 1
                    if T.bsdf_type == "diffuse":
                        tmin, tmax = T.determine_range(self.args_0[offset] ,tmin, tmax)
                    elif T.bsdf_type == "twosided":
                        tmin, tmax = T.determine_range(self.args_0[offset] ,tmin, tmax)
                    elif T.bsdf_type == "phong":
                        tmin, tmax = T.determine_range(self.args_0[offset: offset + 3] ,tmin, tmax)
                    elif T.bsdf_type == "roughdielectric":
                        tmin, tmax = T.determine_range(self.args_0[offset : offset + 3] ,tmin, tmax)
                    elif T.bsdf_type == "roughconductor":
                        tmin, tmax = T.determine_range(self.args_0[offset : offset + 2] ,tmin, tmax)
                elif T.type_name == "MEDIUM_VARY":
                    offset = self.offsets["medium"][T.med_id] + 1
                    if T.med_type == "homogeneous":
                        tmin, tmax = T.determine_range(self.args_0[offset : offset + 2] ,tmin, tmax)
                    elif T.med_type == "heterogeneous":
                        tmin, tmax = T.determine_range([ self.args_0[offset + 1], self.args_0[offset + 3] ] ,
                                                         tmin, tmax)
            self.ranges.append([tmin, tmax])
            print("param #%d range = (%.2f, %.2f)" % (ivar, tmin, tmax))

    def set_arguments(self, var_list):
        var_grad = torch.tensor([0]*nder, dtype=torch.float)
        var_clamped = var_list.clone() + self.param_0
        clamped = [False] * nder
        for ivar, var in enumerate(var_list):
            if var < self.ranges[ivar][0] or var > self.ranges[ivar][1]:
                clamped[ivar] = True
                if var < self.ranges[ivar][0]:
                    var_clamped[ivar] = self.ranges[ivar][0]
                    var_grad[ivar]    = 2 * (var-self.ranges[ivar][0]) * (self.ranges[ivar][1]-self.ranges[ivar][0])
                else:
                    var_clamped[ivar] = self.ranges[ivar][1]
                    var_grad[ivar]    = 2 * (var-self.ranges[ivar][1]) * (self.ranges[ivar][1]-self.ranges[ivar][0])
                print("Variable #%d clamped from %.2f to %.2f" % (ivar, var, var_clamped[ivar]))

        self.clean_up()
        centers = copy.deepcopy(self.shape_centers_0)
        centers_vol = copy.deepcopy(self.volume_centers_0)
        # set up vels & params basef on transforms and vars
        for ivar, Tset in enumerate(self.transforms):
            for T in Tset:
                # Camera related
                if T.type_name == "CAMERA_TRANSLATE":
                    offset = self.offsets["camera"]
                    self.args[offset][0:3, 3] += T.vec_vel * var_clamped[ivar]
                elif T.type_name == "CAMERA_ROTATE":
                    offset = self.offsets["camera"]
                    axis_world = torch.mm(self.args_0[offset][0:3, 0:3], T.vec_vel.unsqueeze(1)).squeeze(1)
                    self.args[offset][0:3, 0:3] = torch.mm(transform.gen_rotation_matrix3x3(axis_world, var_clamped[ivar]),
                                                                                         self.args[offset][0:3, 0:3])
                # Shape related
                elif T.type_name == "SHAPE_TRANSLATE":
                    offset = self.offsets["shape"][T.shape_id]
                    num_verts  = self.args_0[offset].size(0)
                    self.args[offset][:] += T.vec_vel * var_clamped[ivar]
                    centers[T.shape_id] += T.vec_vel * var_clamped[ivar]
                elif T.type_name == "VERTEX_TRANSLATE":
                    offset = self.offsets["shape"][T.shape_id]
                    num_verts  = self.args_0[offset].size(0)
                    self.args[offset][T.vert_id] += T.vec_vel * var_clamped[ivar]
                elif T.type_name == "SHAPE_ROTATE":
                    offset = self.offsets["shape"][T.shape_id]
                    num_verts  = self.args_0[offset].size(0)
                    R = transform.gen_rotation_matrix3x3(T.vec_vel, var_clamped[ivar])
                    for ivert in range(num_verts):
                        pos_local = self.args[offset][ivert] - centers[T.shape_id]
                        self.args[offset][ivert] =  torch.mm(R, pos_local.unsqueeze(1)).squeeze(1) + centers[T.shape_id]
                    has_normal = self.args_0[offset + 3] is not None
                    if has_normal:
                        for ivert in range(num_verts):
                            self.args[offset + 3][ivert] =  torch.mm(R, self.args[offset+3][ivert].unsqueeze(1)).squeeze(1)
                elif T.type_name == "SHAPE_GLOBAL_ROTATE":
                    offset = self.offsets["shape"][T.shape_id]
                    num_verts  = self.args_0[offset].size(0)
                    R = transform.gen_rotation_matrix3x3(T.vec_vel, var_clamped[ivar])
                    for ivert in range(num_verts):
                        self.args[offset][ivert] = torch.mm(R, self.args[offset][ivert].unsqueeze(1)).squeeze(1)
                    centers[T.shape_id] = torch.mm(R, centers[T.shape_id].unsqueeze(1)).squeeze(1)
                    has_normal = self.args_0[offset + 3] is not None
                    if has_normal:
                        for ivert in range(num_verts):
                            self.args[offset + 3][ivert] =  torch.mm(R, self.args[offset+3][ivert].unsqueeze(1)).squeeze(1)
                elif T.type_name == "SHAPE_SCALE":
                    offset = self.offsets["shape"][T.shape_id]
                    num_verts  = self.args_0[offset].size(0)
                    for ivert in range(num_verts):
                        self.args[offset][ivert] = (T.scl*var_clamped[ivar]) * (self.args_0[offset][ivert] - centers[T.shape_id]) + centers[T.shape_id]
                elif T.type_name == "SHAPE_BLENDING":
                    offset = self.offsets['shape'][T.shape_id]
                    num_verts = self.args_0[offset].size(0)
                    self.args[offset] += var_clamped[ivar] * (T.vert_target - self.args[offset])
                    has_normal = self.args_0[offset + 3] is not None
                    if has_normal:
                        self.args[offset + 3] += var_clamped[ivar] * (T.norm_target - self.args[offset + 3])
                # BSDF related
                elif T.type_name == "BSDF_VARY":
                    offset = self.offsets["bsdf"][T.bsdf_id]
                    assert(self.args[offset] == T.bsdf_type)
                    offset += 1
                    if T.bsdf_type == "diffuse":
                        self.args[offset]             += T.d_reflectance * var_clamped[ivar]
                    elif T.bsdf_type == "twosided":
                        self.args[offset]             += T.nested * var_clamped[ivar]
                    elif T.bsdf_type == "phong":
                        self.args[offset]             += T.d_diffuse * var_clamped[ivar]
                        self.args[offset + 1]         += T.d_specular * var_clamped[ivar]
                        self.args[offset + 2]         += T.d_exponent * var_clamped[ivar]
                    elif T.bsdf_type == "roughdielectric":
                        self.args[offset]             += T.d_alpha * var_clamped[ivar]
                        eta = self.args_0[offset+1]/self.args_0[offset+2]
                        eta += T.d_eta * var_clamped[ivar]
                        self.args[offset + 1]         = eta * self.args[offset + 2]
                        self.args[offset + 3]         += T.d_spectrum * var_clamped[ivar] 
                    elif T.bsdf_type == "roughconductor":
                        self.args[offset]             += T.d_alpha * var_clamped[ivar]
                        self.args[offset + 1]         += T.d_k * var_clamped[ivar]
                        self.args[offset + 2]         += T.d_eta * var_clamped[ivar]
                # Medium related
                elif T.type_name == "MEDIUM_VARY":
                    offset = self.offsets["medium"][T.med_id]
                    assert(self.args[offset] == T.med_type)
                    offset += 1
                    if T.med_type == "homogeneous":
                        self.args[offset]             += T.d_sigmaT * var_clamped[ivar]
                        self.args[offset + 1]         += T.d_albedo * var_clamped[ivar]
                    elif T.med_type == "heterogeneous":
                        if not isinstance(self.args_0[offset + 1], str):
                            self.args[offset + 1]         += T.d_albedo * var_clamped[ivar]
                        self.args[offset + 3]         += T.d_scalar * var_clamped[ivar]
                        R   = transform.gen_rotation_matrix4x4(T.vec_rotate, var_clamped[ivar])
                        Tr  = transform.gen_translate_matrix(centers_vol[T.med_id])
                        Tr_ = transform.gen_translate_matrix(-centers_vol[T.med_id])
                        M   = self.args[offset + 2]
                        M   = torch.mm(Tr_, M)
                        M   = torch.mm(  R, M)
                        M   = torch.mm( Tr, M)
                        M   = torch.mm(transform.gen_translate_matrix(T.vec_translate * var_clamped[ivar]), M)
                        self.args[offset + 2] = M.clone().contiguous()
                        centers_vol[T.med_id] += T.vec_translate * var_clamped[ivar]
                # Phase related
                elif T.type_name == "PHASE_VARY":
                    offset = self.offsets["phase"][T.phase_id]
                    assert(self.args[offset] == T.phase_type)
                    offset += 1
                    self.args[offset]             += T.d_g * var_clamped[ivar]
                # Emitter related
                elif T.type_name == "EMITTER_VARY":
                    offset = self.offsets["emitter"][T.emitter_id]
                    assert(self.args[offset] == T.emitter_type)
                    offset += 2
                    self.args[offset]             += T.d_intensity * var_clamped[ivar]
                    self.args[offset + 1]         += T.d_kappa * var_clamped[ivar]
                elif T.type_name == "EMITTER_INVSCALE":
                    offset = self.offsets["emitter"][T.emitter_id]
                    assert(self.args[offset] == T.emitter_type)
                    offset += 2
                    self.args[offset]             = self.args_0[offset] / torch.pow(T.scl*var_clamped[ivar], 2.0)

            if not clamped[ivar]:
                for T in Tset:
                    # Camera related
                    if T.type_name == "CAMERA_TRANSLATE":
                        offset = self.offsets["camera"]
                        self.args[offset + 5][ivar] += T.vec_vel
                    elif T.type_name == "CAMERA_ROTATE":
                        offset = self.offsets["camera"]
                        axis_world = torch.mm(self.args_0[offset][0:3, 0:3], T.vec_vel.unsqueeze(1)).squeeze(1)
                        self.args[offset + 5][nder+ivar] = axis_world
                    # Shape related
                    elif T.type_name == "SHAPE_TRANSLATE":
                        offset = self.offsets["shape"][T.shape_id]
                        num_verts  = self.args_0[offset].size(0)
                        self.args[offset + 8][ivar*num_verts : (ivar+1)*num_verts] += T.vec_vel
                    elif T.type_name == "VERTEX_TRANSLATE":
                        offset = self.offsets["shape"][T.shape_id]
                        num_verts  = self.args_0[offset].size(0)
                        self.args[offset + 8][ivar*num_verts + T.vert_id]  += T.vec_vel
                    elif T.type_name == "SHAPE_ROTATE":
                        offset = self.offsets["shape"][T.shape_id]
                        num_verts  = self.args_0[offset].size(0)
                        for ivert in range(num_verts):
                            self.args[offset + 8][ivar*num_verts + ivert] += T.vec_vel.cross(self.args[offset][ivert] - centers[T.shape_id])
                        has_normal = self.args_0[offset + 3] is not None
                        if has_normal:
                            for ivert in range(num_verts):
                                self.args[offset + 8][(ivar+nder)*num_verts + ivert] += T.vec_vel.cross(self.args[offset+3][ivert])
                    elif T.type_name == "SHAPE_GLOBAL_ROTATE":
                        offset = self.offsets["shape"][T.shape_id]
                        num_verts  = self.args_0[offset].size(0)
                        for ivert in range(num_verts):
                            self.args[offset + 8][ivar*num_verts + ivert] += T.vec_vel.cross(self.args[offset][ivert])
                        has_normal = self.args_0[offset + 3] is not None
                        if has_normal:
                            for ivert in range(num_verts):
                                self.args[offset + 8][(ivar+nder)*num_verts + ivert] += T.vec_vel.cross(self.args[offset+3][ivert])
                    elif T.type_name == "SHAPE_SCALE":
                        offset = self.offsets["shape"][T.shape_id]
                        num_verts  = self.args_0[offset].size(0)
                        for ivert in range(num_verts):
                            self.args[offset + 8][ivar*num_verts + ivert] += T.scl * (self.args_0[offset][ivert] - centers[T.shape_id])
                    elif T.type_name == "SHAPE_BLENDING":
                        offset = self.offsets['shape'][T.shape_id]
                        num_verts = self.args_0[offset].size(0)
                        self.args[offset + 8][ivar*num_verts : (ivar+1)*num_verts] += T.vert_target - self.args_0[offset]
                        has_normal = self.args_0[offset + 3] is not None
                        if has_normal:
                            self.args[offset + 8][(ivar+nder)*num_verts : (ivar+nder+1)*num_verts] += T.norm_target - self.args_0[offset + 3]
                    # BSDF related
                    elif T.type_name == "BSDF_VARY":
                        offset = self.offsets["bsdf"][T.bsdf_id]
                        assert(self.args[offset] == T.bsdf_type)
                        offset += 1
                        if T.bsdf_type == "diffuse":
                            self.args[offset + 1][ivar]   += T.d_reflectance
                        if T.bsdf_type == "twosided":
                            self.args[offset + 1][ivar]   += T.twosided
                        elif T.bsdf_type == "phong":
                            self.args[offset + 3][ivar]   += T.d_diffuse
                            self.args[offset + 4][ivar]   += T.d_specular
                            self.args[offset + 5][ivar]   += T.d_exponent
                        elif T.bsdf_type == "roughdielectric":
                            self.args[offset + 4][ivar]   += T.d_alpha 
                            self.args[offset + 5][ivar]   += T.d_eta 
                            self.args[offset + 6][ivar]   += T.d_spectrum 
                        elif T.bsdf_type == "roughconductor":
                            self.args[offset + 3][ivar]   += T.d_alpha
                            self.args[offset + 4][ivar]   += T.d_k
                            self.args[offset + 5][ivar]   += T.d_eta
                    # Medium related
                    elif T.type_name == "MEDIUM_VARY":
                        offset = self.offsets["medium"][T.med_id]
                        assert(self.args[offset] == T.med_type)
                        offset += 1
                        if T.med_type == "homogeneous":
                            self.args[offset + 3][ivar]   += T.d_sigmaT
                            self.args[offset + 4][ivar]   += T.d_albedo
                        elif T.med_type == "heterogeneous":
                            self.args[offset + 5][ivar]   += T.d_albedo
                            self.args[offset + 6][ivar]   += T.d_scalar
                            self.args[offset + 7][ivar]   += T.vec_translate
                            assert torch.norm(self.args[offset + 8][ivar]) < 1e-6
                            self.args[offset + 8][ivar]    = T.vec_rotate
                    elif T.type_name == "PHASE_VARY":
                        offset = self.offsets["phase"][T.phase_id]
                        assert(self.args[offset] == T.phase_type)
                        offset += 1
                        self.args[offset + 1][ivar]   += T.d_g
                    elif T.type_name == "EMITTER_VARY":
                        offset = self.offsets["emitter"][T.emitter_id]
                        assert(self.args[offset] == T.emitter_type)
                        offset += 4
                        self.args[offset][ivar]         += T.d_intensity
                        self.args[offset + 1][ivar]     += T.d_kappa
                    elif T.type_name == "EMITTER_INVSCALE":
                        offset = self.offsets["emitter"][T.emitter_id]
                        assert(self.args[offset] == T.emitter_type)
                        offset += 2
                        assert torch.norm(self.args[offset + 1][ivar]) < 1e-6
                        self.args[offset + 1][ivar]      = -2.0 * T.scl * torch.pow(var_clamped[ivar], -3.0) * self.args_0[offset]
        return var_grad

    def render(self, integrator, opts, time_elapsed = None):
        dim_guide_direct   = list(self.grid_guide_direct.shape)
        dim_guide_indirect = list(self.grid_guide_indirect.shape)
        with_guiding = dim_guide_direct != [1, 1, 1] or dim_guide_indirect != [1, 1, 1]
        # single camera mode
        if self.cam2world_list is None:
            if not with_guiding:
                if not opts.quiet:
                    print("[SceneManager] render without guiding...")
                out, out_transient, tmp = pydtrr.render_scene_timed(integrator, opts, *(self.args))
            else:
                if not opts.quiet:
                    print("[SceneManager] render with guiding...")
                sceneData, out, out_transient = pydtrr.build_scene(opts, *(self.args))
                scene = sceneData[-1]
                scene.initPathSpaceEdges(dtrr.float_ptr(self.shape_weights.data_ptr()),
                                        dim_guide_direct,   dtrr.float_ptr(self.grid_guide_direct.data_ptr()),
                                        dim_guide_indirect, dtrr.float_ptr(self.grid_guide_indirect.data_ptr()))
                tmp = pydtrr.render(integrator, opts, sceneData, out, out_transient)
        else:
            out_per_cam = []
            tmp = 0.0
            for cam_id, cam2world in enumerate(self.cam2world_list):
                # updating the camera info
                self.args[self.offsets["camera"]] = cam2world
                print('Rendering camera [%d] ...' % cam_id)
                if not with_guiding:
                    if not opts.quiet:
                        print("[SceneManager] render without guiding...")
                    out, out_transient, tmp_per_cam = pydtrr.render_scene_timed(integrator, opts, *(self.args))
                    tmp += tmp_per_cam
                    out_per_cam.append(out)
                else:
                    if not opts.quiet:
                        print("[SceneManager] render with guiding...")
                    sceneData, out, out_transient = pydtrr.build_scene(opts, *(self.args))
                    scene = sceneData[-1]
                    scene.initPathSpaceEdges(dtrr.float_ptr(self.shape_weights.data_ptr()),
                                            dim_guide_direct,   dtrr.float_ptr(self.grid_guide_direct.data_ptr()),
                                            dim_guide_indirect, dtrr.float_ptr(self.grid_guide_indirect.data_ptr()))
                    tmp_per_cam = pydtrr.render(integrator, opts, sceneData, out, out_transient)
                    tmp += tmp_per_cam
                    out_per_cam.append(out)
            out = torch.cat(tuple(out_per_cam), 2)
        if time_elapsed is not None:
            assert isinstance(time_elapsed, list)
            time_elapsed.append(tmp)
        return out, out_transient
