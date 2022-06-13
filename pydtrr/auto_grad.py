import pydtrr
import torch
import numpy as np
from dtrr import nder
import math
import scipy
import scipy.ndimage



def downsample(input):
	if input.size(0) % 2 == 1:
		input = torch.cat((input, torch.unsqueeze(input[-1,:], 0)), dim=0)
	if input.size(1) % 2 == 1:
		input = torch.cat((input, torch.unsqueeze(input[:,-1], 1)), dim=1)
	return (input[0::2, 0::2, :] + input[1::2, 0::2, :] + input[0::2, 1::2, :] + input[1::2, 1::2, :]) * 0.25

class ADLossFunc(torch.autograd.Function):

	@staticmethod
	def forward(ctx, scene_manager, integrator, options, input, out_of_range = torch.tensor([0]*nder, dtype=torch.float),
																penalty_scale = torch.tensor([1]*nder, dtype=torch.float),
																pyramid_level = 1,
																pyramid_scale = 4.0,
																index_iter = -1,
																clamping = 0,
																time_elapsed = None,
																img_all = None):
		# img = pydtrr.render_scene(integrator, options, *(scene_manager.args))
		img = scene_manager.render(integrator, options, time_elapsed)
		if index_iter > -1:
			torch.save(img, 'pt_iter%d.pt'%index_iter)
		ret = img[0, :, :, :]
		ctx.save_for_backward(img[1:, :, :,:],
							  torch.tensor([pyramid_level], dtype=torch.int),
							  torch.tensor([pyramid_scale], dtype=torch.float),
							  out_of_range,
							  penalty_scale,
							  torch.tensor([clamping], dtype=torch.int))
		if img_all is not None:
			assert isinstance(img_all, list)
			data = img.clone().detach()
			if not img_all:
				img_all.append(data)
			else:
				img_all[0] = data
		return ret

	@staticmethod
	def backward(ctx, grad_input):
		ret_list = [None, None, None]
		derivs, lvl, pyramid_scale, out_of_range, penalty_scale, clamping = ctx.saved_tensors
		lvl = int(min( math.log(derivs.size(1), 2)+1, math.log(derivs.size(2),2)+1, lvl))
		ret = torch.tensor([0]*nder, dtype=torch.float)
		grad_curr = []
		for ider in range(nder):
			grad_curr.append(derivs[ider, :, :, :])
		for i in range(lvl):
			for ider in range(nder):
				if abs(out_of_range[ider].item()) > 1e-4:
					print("param #%d is out of range..." % ider)
					ret[ider] = out_of_range[ider] * penalty_scale[ider]
				else:
					if clamping.data[0] == 0:
						product = grad_curr[ider] * grad_input
						# pydtrr.imwrite( grad_input, 'diff_image%d.exr' % i)
						# pydtrr.imwrite(-grad_input, 'diff_image%d_neg.exr' % i)
						# pydtrr.imwrite( grad_curr[ider], 'deriv_image%d.exr' % i)
						# pydtrr.imwrite(-grad_curr[ider], 'deriv_image%d_neg.exr' % i)
						# pydtrr.imwrite( product, 'product_image%d.exr' % i)
						# pydtrr.imwrite(-product, 'product_image%d_neg.exr' % i)
						val = product.sum()
						# print('lvl = %d, deriv = %.2e' % (i, pow(pyramid_scale[0], i)*val))
						ret[ider] += pow(pyramid_scale[0], i) * val / lvl
					else:
						clamped = grad_input.clone()
						clamped[grad_input > 2.0]  = 0.0
						clamped[grad_input < -2.0] = 0.0
						ret[ider] += pow(pyramid_scale[0], i) * (grad_curr[ider] * clamped).sum() / lvl
				if i < lvl - 1:
					grad_curr[ider] = downsample(grad_curr[ider])
			if i < lvl - 1:
				grad_input = downsample(grad_input)
		ret_list.append(ret)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		return tuple(ret_list)

class ADLossFunc2(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, img_ori, img_grad, pyramid_level = 1, pyramid_scale = 4.0):
		ctx.save_for_backward(img_grad,
							  torch.tensor([pyramid_level], dtype=torch.int),
							  torch.tensor([pyramid_scale], dtype=torch.float))
		return img_ori

	def backward(ctx, grad_input):
		img_grad, lvl, pyramid_scale = ctx.saved_tensors
		lvl = int(min( math.log(img_grad.size(1), 2)+1, math.log(img_grad.size(2),2)+1, lvl))
		num_deriv = img_grad.shape[0]
		ret = torch.tensor([0]*num_deriv, dtype=torch.float)
		grad_curr = []
		for ider in range(num_deriv):
			grad_curr.append(img_grad[ider, :, :, :])

		for i in range(lvl):
			for ider in range(num_deriv):
				product = grad_input * grad_curr[ider]
				ret[ider] += pow(pyramid_scale[0], i) * product.sum() / lvl
				if i < lvl - 1:
					grad_curr[ider] = downsample(grad_curr[ider])
			if i < lvl - 1:
				grad_input = downsample(grad_input)

		return tuple( [ret, None, None, None, None] )
		
class ADLossFuncTrans(torch.autograd.Function):

	@staticmethod
	def forward(ctx, scene_manager, integrator, options, input, out_of_range = torch.tensor([0]*nder, dtype=torch.float),
																penalty_scale = torch.tensor([1]*nder, dtype=torch.float),
																pyramid_level = 1,
																pyramid_scale = 4.0,
																index_iter = -1,
																clamping = 0,
																time_elapsed = None,
																img_all = None):
		# img = pydtrr.render_scene(integrator, options, *(scene_manager.args))
		img, trans = scene_manager.render(integrator, options, time_elapsed)
		if index_iter > -1:
			torch.save(trans, 'pt_iter%d.pt'%index_iter)
		ret = trans[0, :, :, :, :]
		ctx.save_for_backward(trans[1:, :, :, :,:],
							  torch.tensor([pyramid_level], dtype=torch.int),
							  torch.tensor([pyramid_scale], dtype=torch.float),
							  out_of_range,
							  penalty_scale,
							  torch.tensor([clamping], dtype=torch.int))
		if img_all is not None:
			assert isinstance(img_all, list)
			data = trans.clone().detach()
			if not img_all:
				img_all.append(data)
			else:
				img_all[0] = data
		return ret

	@staticmethod
	def backward(ctx, grad_input):
		ret_list = [None, None, None]
		derivs, lvl, pyramid_scale, out_of_range, penalty_scale, clamping = ctx.saved_tensors
		lvl = int(min( math.log(derivs.size(1), 2)+1, math.log(derivs.size(2),2)+1, lvl))
		ret = torch.tensor([0]*nder, dtype=torch.float)
		grad_curr = []
		for ider in range(nder):
			grad_curr.append(derivs[ider, :, :, :])
		for i in range(lvl):
			for ider in range(nder):
				if abs(out_of_range[ider].item()) > 1e-4:
					print("param #%d is out of range..." % ider)
					ret[ider] = out_of_range[ider] * penalty_scale[ider]
				else:
					if clamping.data[0] == 0:
						product = grad_curr[ider] * grad_input
						# pydtrr.imwrite( grad_input, 'diff_image%d.exr' % i)
						# pydtrr.imwrite(-grad_input, 'diff_image%d_neg.exr' % i)
						# pydtrr.imwrite( grad_curr[ider], 'deriv_image%d.exr' % i)
						# pydtrr.imwrite(-grad_curr[ider], 'deriv_image%d_neg.exr' % i)
						# pydtrr.imwrite( product, 'product_image%d.exr' % i)
						# pydtrr.imwrite(-product, 'product_image%d_neg.exr' % i)
						val = product.sum()
						# print('lvl = %d, deriv = %.2e' % (i, pow(pyramid_scale[0], i)*val))
						ret[ider] += pow(pyramid_scale[0], i) * val / lvl
					else:
						clamped = grad_input.clone()
						clamped[grad_input > 2.0]  = 0.0
						clamped[grad_input < -2.0] = 0.0
						ret[ider] += pow(pyramid_scale[0], i) * (grad_curr[ider] * clamped).sum() / lvl
				if i < lvl - 1:
					grad_curr[ider] = downsample(grad_curr[ider])
			if i < lvl - 1:
				grad_input = downsample(grad_input)
		ret_list.append(ret)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		ret_list.append(None)
		return tuple(ret_list)
