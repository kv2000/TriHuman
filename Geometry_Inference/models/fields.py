"""
@File: fields.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: Some are borrowed from NeuS(https://github.com/Totoro97/NeuS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=[4],
                 multires=0,
                 bias=0.5,
                 scale=1,
                 to_embed_d_in = 0,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):

        super(SDFNetwork, self).__init__()

        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
    
        self.embed_fn_fine = None

        self.to_embed_d_in = to_embed_d_in
        self.non_embed_d_in = d_in - self.to_embed_d_in
        
        self.num_layers = len(self.dims)
        self.skip_in = skip_in
        self.scale = scale
        self.multires = multires
        self.weight_norm = weight_norm

        if (multires > 0) and (self.to_embed_d_in > 0):
            embed_fn, input_ch = get_embedder(self.multires, input_dims=self.to_embed_d_in)
            self.embed_fn_fine = embed_fn
            self.dims[0] = input_ch + self.non_embed_d_in
        
        print('self.to_embed_d_in,self.non_embed_d_in', d_in, self.to_embed_d_in, self.non_embed_d_in, self.multires)
        print('embedded dims',self.dims)
        
        for l in range(0, self.num_layers - 1):
            
            out_dim = self.dims[l + 1]
            #if l + 1 in self.skip_in:   
            #    out_dim = self.dims[l + 1] + self.dims[0]
            
            in_dim = self.dims[l]
            if l in self.skip_in:   
                in_dim = self.dims[l] + self.dims[0]
                        
            lin = nn.Linear(in_dim, out_dim)

            if geometric_init:
                
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(self.dims[0] - 3):], 0.0)
                
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        # 0 1 2
        to_embed = inputs[:,:self.to_embed_d_in]
        non_embed = inputs[:,self.to_embed_d_in:]
        
        if self.embed_fn_fine is not None:
            to_embed = self.embed_fn_fine(to_embed)

        fin_inputs = torch.cat((
            to_embed, non_embed
        ),dim=1)
        
        x = fin_inputs
                
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, fin_inputs], 1) / np.sqrt(2)
            
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([
            x[:, :1] / self.scale, x[:, 1:]
        ], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        return gradients.unsqueeze(1)

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                d_feature=256,
                mode='idr',
                d_in=9,
                d_out=3,
                d_hidden=256,
                n_layers=4,
                weight_norm=True,
                multires_view=4,
                squeeze_out=True
            ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'trans_idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors, trans_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'idr_no_pts':
            rendering_input = torch.cat([view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'idr_no_pts_no_normal':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        
        return x

class MotionNetwork(nn.Module):
    def __init__(self,
                d_in=85,
                d_out=16,
                d_hidden=128,
                n_layers=4,
                weight_norm=True,
                squeeze_out=True
            ):
        
        super().__init__()

        self.squeeze_out = squeeze_out
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        print(
            '++++++ motion network dims:', dims
        )

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, x):
        
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        
        return x

class TranslationNetwork(nn.Module):
    def __init__(self,
                d_in=3,
                d_out=16,
                d_hidden=128,
                n_layers=4,
                weight_norm=True,
                multires_view = 5
            ):
        
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
            
        print(
            '++++++ translation network dims:', dims
        )

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, x):

        if self.embedview_fn is not None:
            x = self.embedview_fn(x)
        
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 1:
                x = self.relu(x)
        
        return x

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).to(self.variance.device) * torch.exp(self.variance * 10.0)

if __name__ == '__main__':
    
    print('wootwootwo')
    
    mock_vec = torch.zeros([1,3]).cuda()
    
    trans_net = TranslationNetwork(
        d_in=3,
        d_out=16,
        d_hidden=64,
        n_layers=2,
        weight_norm=True,
        multires_view=6
    )
    