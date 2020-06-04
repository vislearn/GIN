from math import exp

import torch
import torch.nn as nn

class GINCouplingBlock(nn.Module):
    '''Coupling Block following the GIN design. The difference from the RealNVP coupling blocks
    is that it uses a single subnetwork (like the GLOW coupling blocks) to jointly predict [s_i, t_i], 
    instead of two separate subnetworks, and the Jacobian determinant is constrained to be 1. 
    This constrains the block to be volume-preserving. Volume preservation is achieved by subtracting
    the mean of the output of the s subnetwork from itself. 
    Note: this implementation differs slightly from the originally published implementation, which 
    scales the final component of the s subnetwork so the sum of the outputs of s is zero. There was
    no difference found between the implementations in practice, but subtracting the mean guarantees 
    that all outputs of s are at most ±exp(clamp), which might be more stable in certain cases.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    clamp:              Soft clamping for the multiplicative component. The amplification or attenuation
                        of each input dimension can be at most ±exp(clamp).'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, clamp=5.):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            F"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s1 = subnet_constructor(self.split_len1 + condition_length, self.split_len2*2)
        self.s2 = subnet_constructor(self.split_len2 + condition_length, self.split_len1*2)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r2 = self.s2(torch.cat([x2, *c], 1) if self.conditional else x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            s2 = self.log_e(s2)
            s2 -= s2.mean(1, keepdim=True)
            y1 = torch.exp(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, *c], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            s1 = self.log_e(s1)
            s1 -= s1.mean(1, keepdim=True)
            y2 = torch.exp(s1) * x2 + t1
            
            self.last_jac = (  torch.sum(s1, dim=tuple(range(1, self.ndims+1)))
                             + torch.sum(s2, dim=tuple(range(1, self.ndims+1))))

        else: # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, *c], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            s1 = self.log_e(s1)
            s1 -= s1.mean(1, keepdim=True)
            y2 = (x2 - t1) * torch.exp(-s1)

            r2 = self.s2(torch.cat([y2, *c], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            s2 = self.log_e(s2)
            s2 -= s2.mean(1, keepdim=True)
            y1 = (x1 - t2) * torch.exp(-s2)
            self.last_jac = (- torch.sum(s1, dim=tuple(range(1, self.ndims+1)))
                             - torch.sum(s2, dim=tuple(range(1, self.ndims+1))))

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims
        

