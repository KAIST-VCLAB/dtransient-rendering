import dtrr
from pydtrr import SceneManager, SceneTransform
import pydtrr
import torch
import sys, os, platform
import re
import argparse
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import OpenEXR as opexr
import Imath
import matplotlib.pyplot as plt
import time

def create_directory(dirname):
    if not os.path.exists(dirname):
        os.mkdir( dirname )

def time_to_string(time_elapsed):
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    if hours > 0:
        ret = "{:0>2}h {:0>2}m {:0>2.2f}s".format(hours, minutes, seconds)
    elif minutes > 0:
        ret = "{:0>2}m {:0>2.2f}s".format(minutes, seconds)
    else:
        ret = "{:0>2.2f}s".format(seconds)
    return ret

def fill_value2form(arr, format):
    attributes = {"min":lambda x:x.min(), "max":lambda x:x.max(), "mean":lambda x:x.mean(), "sum":lambda x:x.sum()}
    n_form = format.count("%f")
    values = []
    for attr in attributes:
        temp_list = [(m.start(), attributes[attr](arr)) for m in re.finditer(attr, format)]
        values += temp_list
    if n_form != len(values):
        raise ValueError("The number of formatting characters (%d) and the number of attribute names (%d) must be same." % (n_form, len(values)))
    values.sort(key=lambda x:x[0])
    return format % tuple([val[1] for val in values])

def downsample(mat, factor, dim=[]):
    if not dim:
    	dim = range(len(mat.shape))
    res = mat
    for axis in dim:
    	shape = list(res.shape)
    	shape = shape[:axis] + [factor, shape[axis]//factor] + shape[axis+1:]
    	res = res.reshape(shape).mean(axis)
    return res

def mats_abs_max(*args, **kargs):
    if 'factor' not in kargs:
    	factor = 1
    else:
        factor = kargs['factor']
    	
    lst = [downsample(abs(mat), factor, [0,1]).max() for mat in args]
    return max(lst)

def readEXR(img_path):
    """Converts OpenEXR image to numpy float array."""

    # Read OpenEXR file
    if not opexr.isOpenExrFile(img_path):
        raise ValueError(f'Image {img_path} is not a valid OpenEXR file')
    src = opexr.InputFile(img_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = src.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read into tensor
    tensor = np.zeros((size[1], size[0],3))
    for i, c in enumerate('RGB'):
        rgb32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32)
        tensor[:, :, i] = rgb32f.reshape(size[1], size[0])

    return tensor

def readSequence(filename, frame_i, frame_f, t_dim=0):
    # t_dim == 0: img_array T*W*H*C
    # t_dim == 2: img_array W*H*T*C
    for frame in range(frame_i, frame_f + 1):
        tmp_filename = filename % frame
        img = readEXR(tmp_filename)
        
        if frame == frame_i:
            size = img.shape
            img_array = np.zeros((frame_f - frame_i + 1, *size))
            
        img_array[frame-frame_i, :, :, :] = img
    if t_dim != 0:
        dims = [1,2,3]
        dims.insert(t_dim, 0)
        img_array = img_array.transpose(dims)
    return img_array

def readImageTrans(filename_ss, filename_tr, frame_i, frame_f, t_dim=0):
    img = readEXR(filename_ss)
    return (img, readSequence(filename_tr, frame_i, frame_f, t_dim))

def write_log(filename, title, img, img_trans, time=None):
    with open(filename, 'a') as file:
        file.write( "\n------------------------------\n")
        file.write(f'Log for {title}')
        if time != None:
            file.write(f', rendered time={time}\n')
        else:
            file.write('\n')
        
        file.write(fill_value2form(img, "image: mean=%f, min=%f, max=%f\n"))
        file.write(fill_value2form(img_trans, "transient: mean=%f, min=%f, max=%f\n"))

        file.write("total energy: image=%f, transient=%f\n" % (img.sum(), img_trans.sum()))

def writeTrans(file_trans, img_trans):
    if img_trans.shape[-1] > 3 :
        filename_split = file_trans.split('%')
        for i in range(img_trans.shape[-1]//3):
            temp_split = filename_split[:]
            temp_split[-2] += f'ch{i}_'
            writeTrans('%'.join(temp_split), img_trans[:,:,:,3*i:3*(i+1)])
        return
    
    for i_bin in range(img_trans.shape[2]):
        pydtrr.imwrite(img_trans[:,:,i_bin,:], file_trans % i_bin)


def write_all(file_img, file_trans, file_log, title, img, img_trans, time=None):
    pydtrr.imwrite(img, file_img)
    writeTrans(file_trans, img_trans)
        
    write_log(file_log, title, img, img_trans, time)



class Recorder:
    def __init__(self, dir_out, file_log=None):
        # Input
        self.dir_out = dir_out
        if file_log:
            self.file_log = file_log
        else:
            self.file_log = dir_out + 'log.txt'

        # Default
        self.subdir_grad = 'grad_img/'
        self.subdir_iterations = 'iterations/'
        self.subdir_plot = 'plot/'
        self.file_log_target = dir_out + 'log_target.txt'

    def dir_grad(self):
        return self.dir_out + self.subdir_grad
    def dir_iterations(self):
        return self.dir_out + self.subdir_iterations
    def dir_plot(self):
        return self.dir_out + self.subdir_plot
    def dir_iter_i(self, i_iter):
        return self.dir_iterations() + f'iter_{i_iter:04d}/'

    def create_output_directory(self):
        if not os.path.exists(self.dir_out):
            os.mkdir( self.dir_out )
        if not os.path.exists(self.dir_grad()):
            os.mkdir( self.dir_grad() )
        if not os.path.exists(self.dir_iterations()):
            os.mkdir( self.dir_iterations() )
        if not os.path.exists(self.dir_plot()):
            os.mkdir( self.dir_plot() )

    def create_iter_directory(self, i_iter):
        dirname = self.dir_iter_i(i_iter)
        if not os.path.exists(dirname):
            os.mkdir( dirname )

    def set_guiding(self, scene_manager, integrator, options, gspec_direct, gspec_indirect, guiding_quiet = False):
        if gspec_direct:
            scene_manager.set_direct_guiding(gspec_direct, integrator, options, guiding_quiet)
        if gspec_indirect:
            guide_type, indirect_param, num_cam_paths, num_light_paths, radius = gspec_indirect
            assert(guide_type >= 1 and guide_type < 4)
            assert(isinstance(indirect_param, tuple) or isinstance(indirect_param, list))
            scene_manager.set_indirect_guiding(guide_type, indirect_param, num_cam_paths, num_light_paths, radius, integrator, options, guiding_quiet)
            
    def run(self, scene_manager, integrator, options, gspec_direct, gspec_indirect, mode, **kargs):
        self.create_output_directory()
        with open(self.file_log, 'a') as file:
            file.write("Platform: %s\tIntegrator:%s\n"%(platform.node(), str(type(integrator))))
        
        elapsed_times = []
        options = dtrr.RenderOptions(options)
        if mode == 1:
            # No boundary integral
            options.sppe = 0
            options.sppse0 = 0
            options.sppse1 = 0
        elif mode == 2:
            self.set_guiding(scene_manager, integrator, options, gspec_direct, gspec_indirect)
            with open(self.file_log, 'a') as file:
                file.write("Direct guiding at:%s\n"%str(gspec_direct))
                file.write("Indirect guiding at:%s\n"%str(gspec_indirect))

        if mode == 1 or mode == 2:
            img_grad, img_grad_trans = scene_manager.render(integrator, options, elapsed_times) 

            # Print rendered scene
            img_scene = img_grad[0,:,:,:]
            img_scene_trans = img_grad_trans[0,:,:,:,:]

            write_all(self.dir_out + 'image.exr', self.dir_out + 'trans_%05d.exr', self.file_log, "scene", img_scene, img_scene_trans, elapsed_times[-1])

            if mode == 2:
                # Print scene derivatives
                for i in range(dtrr.nder):
                    img_deriv = img_grad[i+1, :, :, :]
                    trans_deriv = img_grad_trans[i+1, :,:,:,:]

                    file_img = self.dir_grad() + 'deriv%d.exr' % i
                    file_trans = self.dir_grad() + 'deriv%d'%i + '_trans_%05d.exr'
                    write_all(file_img, file_trans, self.file_log, f'deriv{i}', img_deriv, trans_deriv)
        
        elif mode == 0:
            ########## Parsing  ##########
            param_target = kargs['param_target']
            flag_render_target = 'spp_target' in kargs
            if flag_render_target:
                assert('rendered_target' not in kargs)
                spp_target = kargs['spp_target']
                with open(self.file_log, 'a') as file:
                    file.write("Render target with %sspp\n"%spp_target)
            else:
                rendered_target = kargs['rendered_target']
                if type(rendered_target) is list:
                    img_target_trans = readSequence(*rendered_target)
                    img_target_trans = torch.tensor(np.transpose(img_target_trans, (1, 2, 0, 3)))
                elif type(rendered_target) is torch.Tensor:
                    img_target_trans = rendered_target
                else:
                    raise TypeError(f'Argument rendered_target should be a list or torch.Tensor, but now is {type(rendered_target)}')
                with open(self.file_log, 'a') as file:
                    file.write("Rendered target %s\n"%rendered_target)
                
            if 'num_iters' in kargs:
                num_iters = kargs['num_iters']
            else:
                num_iters = 200
            
            if 'lr' in kargs:
                lr = kargs['lr']
            else:
                lr = 1e-2

            if 'ds_factor' in kargs:
                ds_factor = kargs['ds_factor']
            else:
                ds_factor = 1

            with open(self.file_log, 'a') as file:
                file.write("Optimization options: num_iters=%d, lr=%f, ds_factor=%d\n"% (num_iters, lr, ds_factor))
            ##############################

            print('[INFO] optimization for inverse rendering starts...')
            
            fig_n = dtrr.nder+1
            if fig_n > 5:
                fig_h = math.floor(math.sqrt(dtrr.nder+1))
                fig_w = math.ceil(fig_n/fig_h)
            else:
                fig_h = 1
                fig_w = fig_n
            #fig = plt.figure(figsize=(3*(dtrr.nder+1), 3))
            #gs1 = fig.add_gridspec(nrows=1, ncols=dtrr.nder+1)
            fig = plt.figure(figsize=(3*fig_w, 3*fig_h))
            gs1 = fig.add_gridspec(nrows=fig_h, ncols=fig_w)

            loss_record = [[]]
            for i in range(dtrr.nder):
                loss_record.append([])
            
            if flag_render_target:
                scene_manager.set_arguments( param_target )
                options_target = dtrr.RenderOptions(options)
                options_target.spp = spp_target
                options_target.sppe = 0
                options_target.sppse0 = 0
                options_target.sppse1 = 0

                img_target, img_target_trans = scene_manager.render(integrator, options_target, elapsed_times)
                img_target = img_target[0,:,:,:]
                img_target_trans = img_target_trans[0,:,:,:,:]

                pydtrr.imwrite(img_target, self.dir_out + 'image_target.exr')
                temp_dirs = [self.dir_out + x for x in ['image_target.exr', 'trans_target_%05d.exr']]
                write_all(*temp_dirs, self.file_log_target, 'target scene', img_target, img_target_trans, elapsed_times[-1])
                scene_manager.reset()

            lossFunc = pydtrr.ADLossFuncTrans.apply
            param = torch.tensor([0.0]*dtrr.nder, dtype=torch.float, requires_grad=True)
            optimizer = torch.optim.Adam( [param], lr=lr)
            grad_out_range = torch.tensor([0]*dtrr.nder, dtype=torch.float)

            file_loss  = open(self.dir_iterations() + 'iter_loss.log', 'w')
            file_param = open(self.dir_iterations() + 'iter_param.log', 'w')
            num_pyramid_lvl = 9
            weight_pyramid  = 4
            options.quiet = True
            times = []
            img_all = [] # None

            ### Pooling layer
            layer = torch.nn.AvgPool3d(ds_factor, ds_factor)

            for t in range(num_iters):
                print('[Iter %3d]' % t, end=' ')
                optimizer.zero_grad()
                options.seed = t + 1
                # print('[iter %d] re-compute guiding..' % t)
                start = time.time()
                self.set_guiding(scene_manager, integrator, options, gspec_direct, gspec_indirect, True)
                time_elapsed = time.time() - start
                hours, rem = divmod(time_elapsed, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Total preprocess time: {:0>2.2f}s".format(seconds))

                img_all = [] 
                img = lossFunc(scene_manager, integrator, options, param,
                            grad_out_range, torch.tensor([10000.0]*dtrr.nder, dtype=torch.float),  # out of range penalty (not well tested)
                            num_pyramid_lvl, weight_pyramid, -1, 0, times, img_all)

                #pydtrr.imwrite(img, self.dir_iterations() + ('iter_%d.exr' % t))
                #if img_all is not None:
                #    for i in range(dtrr.nder):
                #        pydtrr.imwrite(img_all[0][i + 1, :, :, :], self.dir_iterations() + ('/iter_%d_%d.exr' % (t, i)))
                if t == 0 or t%10 == 9:
                    self.create_iter_directory(t)
                    dirname = self.dir_iter_i(t)
                    writeTrans(dirname + '_%05d.exr', img)
                    for i in range(dtrr.nder):
                        writeTrans(dirname + 'deriv%d'%i + '_%05d.exr', img_all[0][i+1,:,:,:,:])

                # compute losses
                #img_loss = (img - img_target_trans).pow(2).mean()
                img_permute = img.permute(3, 2, 0, 1).unsqueeze(0)
                img_target_permute = img_target_trans.permute(3, 2, 0, 1).unsqueeze(0)
                if ds_factor == 1:
                    img_loss = (img_permute - img_target_permute).pow(2).mean()
                else:
                    img_loss = (layer(img_permute) - layer(img_target_permute)).pow(2).mean()
                opt_loss = np.sqrt(img_loss.detach().numpy())
                param_loss = (param - param_target).pow(2).sum().sqrt()
                print('render time: %s; opt. loss: %.3e; param. loss: %.3e' % (time_to_string(times[-1]), opt_loss, param_loss))

                # write image/param loss
                file_loss.write("%d, %.5e, %.5e, %.2e\n" % (t, opt_loss, param_loss, times[-1]))
                file_loss.flush()

                # write param values
                file_param.write("%d" % t)
                for i in range(dtrr.nder):
                    file_param.write(", %.5e" % param[i])
                file_param.write("\n")
                file_param.flush()

                # write temporal (time-directional) informations
                # tempseq = [ord; der0; ...; me; mse]
                tempseq = torch.zeros((dtrr.nder+3, img.size(2), img.size(3)))
                tempseq[:dtrr.nder+1, :, :] = img_all[0].sum((1,2))
                difftrans = (img - img_target_trans).detach()
                tempseq[dtrr.nder+1, :, :] = difftrans.mean((0,1))
                tempseq[dtrr.nder+2, :, :] = difftrans.pow(2).mean((0,1))
                torch.save(tempseq, f'{self.dir_iterations()}sequences_{t:04d}.pt')

                # plot the results
                # image loss
                loss_record[0].append(opt_loss)
                #ax = fig.add_subplot(gs1[0])
                ax = fig.add_subplot(gs1[0,0])
                ax.plot(loss_record[0], 'b')
                ax.set_title('Img. RMSE')
                #ax.set_xlim([0, num_iters])
                ax.set_yscale('log')
                # param record
                for i in range(dtrr.nder):
                    loss_record[i+1].append( param[i].detach()-param_target[i] )
                    #ax = fig.add_subplot(gs1[i+1])
                    ax = fig.add_subplot(gs1[(i+1)//fig_w,(i+1)%fig_w])
                    ax.plot(loss_record[i+1], 'b')
                    ax.set_title('Img. RMSE')
                    #ax.set_xlim([0, num_iters])
                    rng = max( abs(loss_record[i+1][0])*2, 3*lr)
                    ax.set_ylim([-rng, rng])
                    ax.set_title( 'Param. %d'%(i+1) )
                plt.savefig(self.dir_plot()+'frame_{:03d}.png'.format(t), bbox_inches='tight')
                plt.clf()

                img_loss.backward()
                optimizer.step()

                grad_out_range = scene_manager.set_arguments(param)
            file_loss.close()
            file_param.close()

