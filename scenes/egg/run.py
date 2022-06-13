import dtrr
from pydtrr import SceneManager, SceneTransform
import pydtrr
import torch
import sys, os
import argparse
import numpy as np
from pydtrr.auxiliary import *

def set_direct_guiding(scene_manager, gspec, integrator, options, quiet = False):
    scene_manager.set_direct_guiding(gspec, integrator, options, quiet)

def main(args):
    ### Parameters
    scene, integrator = pydtrr.load_mitsuba('./scene.xml')
    duration = 1000 # from './scene.xml'
    integrator = dtrr.BidirectionalPathTracerADBiBd()
    # When defining unidirectional and bidirectional tracings for interior and boundary integrals resp.
    # similar to Zhang et al. 2020,
    # dtrr.BidirectionalPathTracerAD() performs bidirectional interior integral and unidirectional boundary integral,
    # and dtrr.BidirectionalPathTracerADBiBd() performs bidirectional interior integral and bidirectional boundary integral.
    
    max_bounces = 5
    spp = 16     # samples per pixel for interior integral
    sppe = 0     # samples per pixel for primary boundary integral
                 # (should be nonzero when there is a geometric change directly visible from the camera)
    sppse0 = spp # samples per pixel for secondary boundary integral (NEE in [Zhang et al. 2020])
    sppse1 = spp # samples per pixel for secondary boundary integral
    
    seed = 42
    
    
    dir_out = f'./result/'

    ### Run
    memory_mode = 1
    options = dtrr.RenderOptions(seed, spp*duration, max_bounces, sppe*duration, sppse0*duration, False, memory_mode)
    options.sppse1 = sppse1*duration
    options.grad_threshold = 5e9

    scene_args = pydtrr.serialize_scene(scene)
    xforms = [  [SceneTransform("SHAPE_TRANSLATE", torch.tensor([ 0.0, 100.0, 0.0], dtype=torch.float), 0),
                 SceneTransform("SHAPE_TRANSLATE", torch.tensor([ 0.0, 100.0, 0.0], dtype=torch.float), 1)]
             ]
    init_param = torch.tensor([0.0]*dtrr.nder, dtype=torch.float)
    scene_manager = SceneManager(scene_args, xforms, init_param)
    gspec_direct = [5000, 5, 5, 32]
    gspec_indirect = [1, [5000, 5, 5, 32], 5000, 5000, 1.0]
    guiding_quiet = True
    
    recorder = Recorder(dir_out)
    recorder.run(scene_manager, integrator, options, gspec_direct, gspec_indirect, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Script for Egg Scene',
            epilog='Shinyoung Yi (syyi@vclab.kaist.ac.kr)')
    parser.add_argument('-spp', metavar='samples per pixel', type=int, default=16, help='samples per pixel in integer')
    args = parser.parse_args()
    main(args)
