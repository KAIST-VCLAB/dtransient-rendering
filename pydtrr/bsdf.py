import pydtrr
import torch

class BSDF_TexturedDiffuse:
    def __init__(self, texture_diffuse_reflectance):
        self.width = texture_diffuse_reflectance.shape[0]
        self.height = texture_diffuse_reflectance.shape[1]
        self.texture_diffuse_reflectance = texture_diffuse_reflectance
        self.type = 'texturediffuse'

class BSDF_twosided:
    def __init__(self, nested):
        self.nested = nested
        self.type = 'twosided'

class BSDF_diffuse:
    def __init__(self,
                 diffuse_reflectance):
        self.diffuse_reflectance = diffuse_reflectance
        self.type = 'diffuse'

class BSDF_null:
    def __init__(self):
        self.type = 'null'

class BSDF_Phong:
    def __init__(self,
                 diffuse_reflectance, specular_reflectance, exponent):
        self.diffuse_reflectance = diffuse_reflectance
        self.specular_reflectance = specular_reflectance
        self.exponent = exponent
        self.type = 'phong'

class BSDF_roughdielectric:
    def __init__(self, alpha, intIOR, extIOR, spectrum): 
        self.alpha = alpha
        self.intIOR = intIOR
        self.extIOR = extIOR
        self.spectrum = spectrum
        self.type = 'roughdielectric'

class BSDF_roughconductor:
    def __init__(self, alpha, k, eta):
        self.alpha = alpha
        self.k = k
        self.eta = eta
        self.type = 'roughconductor'

