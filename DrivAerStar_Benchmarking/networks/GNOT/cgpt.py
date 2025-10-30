#!/usr/bin/env python
#-*- coding:utf-8 _*-
import os
import dgl
import torch
import pyvista as pv
import torch.nn as nn
import lightning as L
from utils import register_class
from types import SimpleNamespace
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import GELU, ReLU, Tanh, Sigmoid



class TestLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(TestLoss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# from utils import MultipleTensors
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)


    def __getitem__(self, item):
        return self.x[item]

# from models.mlp import MLP

import torch.nn as nn
import torch.nn.functional as F

import dgl

ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU()}

'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu'):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])



    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x
            # x = self.act(self.bns[i](self.linears[i](x))) + x

        x = self.linear_post(x)
        return x

class GPTConfig():
    """ base GPT config, params common to all GPT versions """
    def __init__(self,attn_type='linear', embd_pdrop=0.0, resid_pdrop=0.0,attn_pdrop=0.0, n_embd=128, n_head=1, n_layer=3, block_size=128, n_inner=512,act='gelu', branch_sizes=1,n_inputs=1):
        self.attn_type = attn_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_embd = n_embd  # 64
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_inner = 4 * self.n_embd
        self.act = act
        self.branch_sizes = branch_sizes
        self.n_inputs = n_inputs


'''
    X: N*T*C --> N*(4*n + 3)*C 
'''
def horizontal_fourier_embedding(X, n=3):
    freqs = 2**torch.linspace(-n, n, 2*n+1).to(X.device)
    freqs = freqs[None,None,None,...]
    X_ = X.unsqueeze(-1).repeat([1,1,1,2*n+1])
    X_cos = torch.cos(freqs * X_)
    X_sin = torch.sin(freqs * X_)
    X = torch.cat([X.unsqueeze(-1), X_cos, X_sin],dim=-1).view(X.shape[0],X.shape[1],-1)
    return X


class LinearAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super(LinearAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(y).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)


        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)   #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)       # normalized
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)  #
            D_inv = 1. / T2                                           # galerkin
        elif self.attn_type == "l2":                                   # still use l1 normalization
            q = q / q.norm(dim=-1,keepdim=True, p=1)
            k = k / k.norm(dim=-1,keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y



class LinearCrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super(LinearCrossAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.keys = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_inputs = config.n_inputs

        self.attn_type = 'l1'

    '''
        Linear Attention and Linear Cross Attention (if y is provided)
    '''
    def forward(self, x, y=None, layer_past=None):
        y = x if y is None else y
        B, T1, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.softmax(dim=-1)
        out = q
        for i in range(self.n_inputs):
            _, T2, _ = y[i].size()
            k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
            k = k.softmax(dim=-1)  #
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
            out = out +  1 * (q @ (k.transpose(-2, -1) @ v)) * D_inv


        # output projection
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out




'''
    Self and Cross Attention block for CGPT, contains  a cross attention block and a self attention block
'''
class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(CrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(config.n_inputs)])
        self.n_inputs = config.n_inputs
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.ln5 = nn.LayerNorm(config.n_embd)

        # self.ln6 = nn.LayerNorm(config.n_embd)      ## for ab study
        if config.attn_type == 'linear':
            print('Using Linear Attention')
            self.selfattn = LinearAttention(config)
            self.crossattn = LinearCrossAttention(config)
            # self.selfattn_branch = LinearAttention(config)
        else:
            raise NotImplementedError

        if config.act == 'gelu':
            self.act = GELU
        elif config.act == "tanh":
            self.act = Tanh
        elif config.act == 'relu':
            self.act = ReLU
        elif config.act == 'sigmoid':
            self.act = Sigmoid

        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)
        self.mlp1 = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        )


    def ln_branchs(self, y):
        return MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])


    def forward(self, x, y):
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))
        x = x + self.mlp1(self.ln3(x))
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        x = x + self.mlp2(self.ln5(x))

        return x





'''
    Cross Attention GPT neural operator
    Trunck Net: geom'''
class CGPTNO(nn.Module):
    def __init__(self,
                 trunk_size=2,
                 branch_sizes=None,
                 output_size=3,
                 n_layers=2,
                 n_hidden=64,
                 n_head=1,
                 n_inner=4,
                 mlp_layers=2,
                 attn_type='linear',
                 act = 'gelu',
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 horiz_fourier_dim = 0,
                 ):
        super(CGPTNO, self).__init__()

        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim>0 else trunk_size
        # self.branch_sizes = [bsize * (4*horiz_fourier_dim + 3) for bsize in branch_sizes] if horiz_fourier_dim > 0 else branch_sizes
        self.branch_sizes = branch_sizes

        self.output_size = output_size

        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)
        if branch_sizes:
            self.n_inputs = len(branch_sizes)
            self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers, act=act) for bsize in self.branch_sizes])
        else:
            self.n_inputs = 0

        self.gpt_config = GPTConfig(attn_type=attn_type, embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout,
                                    attn_pdrop=attn_dropout, n_embd=n_hidden, n_head=n_head, n_layer=n_layers,
                                    block_size=128, act=act, branch_sizes=branch_sizes, n_inputs=self.n_inputs,
                                    n_inner=n_inner)

        self.blocks = nn.Sequential(*[CrossAttentionBlock(self.gpt_config) for _ in range(self.gpt_config.n_layer)])

        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)

        # self.apply(self._init_weights)

        self.__name__ = 'CGPT'



    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, lr=1e-3, betas=(0.9,0.999), weight_decay=0.00001, no_decay_extras=[]):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        for nd in no_decay_extras:
            no_decay.add(nd)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def forward(self, g, u_p, inputs):
        # gs = dgl.unbatch(g)
        # x = pad_sequence([_g.ndata['x'] for _g in gs]).permute(1, 0, 2)  # B, T1, F
        x = torch.cat([x, u_p.unsqueeze(1).repeat([1, x.shape[1], 1])], dim=-1)
        if self.horiz_fourier_dim > 0:
            x = horizontal_fourier_embedding(x, self.horiz_fourier_dim)
            # z = horizontal_fourier_embedding(z, self.horiz_fourier_dim)
        x = self.trunk_mlp(x)
        if self.n_inputs:
            z = MultipleTensors([self.branch_mlps[i](inputs[i]) for i in range(self.n_inputs)])
        else:
            z = MultipleTensors([x])
        for block in self.blocks:
            x = block(x, z)
        x_out = self.out_mlp(x)
        # x_out = torch.cat([x[i, :num] for i, num in enumerate(g.batch_num_nodes())],dim=0)
        return x_out


class CGPT(nn.Module):
    def __init__(self,
                 trunk_size=2,
                 branch_sizes=None,
                 space_dim=2,
                 output_size=3,
                 n_layers=2,
                 n_hidden=64,
                 n_head=1,
                 n_experts = 2,
                 n_inner = 4,
                 mlp_layers=2,
                 attn_type='linear',
                 act = 'gelu',
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 horiz_fourier_dim = 0,
                 ):
        super(CGPT, self).__init__()

        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim>0 else trunk_size
        self.branch_sizes = [bsize * (4*horiz_fourier_dim + 3) for bsize in branch_sizes] if horiz_fourier_dim > 0 else branch_sizes
        self.n_inputs = len(self.branch_sizes)
        self.output_size = output_size
        self.space_dim = space_dim
        self.gpt_config = GPTConfig(attn_type=attn_type,embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout, attn_pdrop=attn_dropout,n_embd=n_hidden, n_head=n_head, n_layer=n_layers,
                                       block_size=128,act=act, branch_sizes=branch_sizes,n_inputs=len(branch_sizes),n_inner=n_inner)

        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)
        self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers,act=act) for bsize in self.branch_sizes])


        self.blocks = nn.Sequential(*[CrossAttentionBlock(self.gpt_config) for _ in range(self.gpt_config.n_layer)])

        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)

        # self.apply(self._init_weights)

        self.__name__ = 'MIOEGPT'

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs, fake_input=None):
        # import pdb;pdb.set_trace()
        x = inputs[:,:,:3]
        # x = pad_sequence([x[:,:,:3] for x in inputs]).permute(1, 0, 2)  # B, T1, F
        x = self.trunk_mlp(x)
        z = MultipleTensors([self.branch_mlps[i](inputs[i]) for i in range(self.n_inputs)])

        for block in self.blocks:
            x = block(x, z)
        x = self.out_mlp(x)

        # x_out = torch.cat([x[i, :num] for i, num in enumerate(g.batch_num_nodes())],dim=0)
        x_out = x
        return x_out


@register_class(name=['CGPT'])
class CGPT_LightningModule(L.LightningModule):
    def __init__(self,
                 normalizer_y=None,
                 normalizer_x=None,
                 point_normalizer_y=None,
                 point_normalizer_x=None,
                 **network_args,
                 ) -> None:
        super().__init__()
        self.__name__ = 'GNOT CGPT'
        self.params = SimpleNamespace(**network_args["hyperparameter"])
        self.save_hyperparameters(ignore=['normalizer_x',
                                          'normalizer_y',
                                          'point_normalizer_y',
                                          'point_normalizer_x'])

        self.use_wandb = 1
        print('='*20, 'Parameters', '='*20)
        print(self.params)
        self.lr = self.params.lr
        print('='*50)
        if normalizer_y: self.normalizer_y = normalizer_y

        if normalizer_x:  self.normalizer_x = normalizer_x

        if point_normalizer_x:  self.point_normalizer_x = point_normalizer_x

        if point_normalizer_y:  self.point_normalizer_y = point_normalizer_y

        self.example_input_array = torch.ones([4, 10086, 7])

        self.loss = TestLoss()
        self.model = CGPT(**network_args["model"])

    def set_normalizer(self,normalizer_x,normalizer_y):
        self.x_normalizer = normalizer_x
        self.x_normalizer.cuda()
        self.y_normalizer = normalizer_y
        self.y_normalizer.cuda()

    def set_test_args(self, writer_path):
        self.writer_path = writer_path
        self.test_step_outputs = []
        
    def forward(self, x):
        out = self.model(x, None).squeeze(-1)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)

        out = self.y_normalizer.decode(out)
        y = self.y_normalizer.decode(y)

        loss = self.loss(out, y)    
        # if self.use_wandb:
        #     wandb.log({'train_loss': loss})

        self.log('train_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x, None).squeeze(-1)
        out = self.y_normalizer.decode(out)
        y = self.y_normalizer.decode(y)
        loss = self.loss(out, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        results = {'val_loss': loss}
        return results

    def test_step(self, batch, batch_idx):
        x, y, vtk_paths = batch
        out = self.model(x, None).squeeze(-1)
        out = self.y_normalizer.decode(out)
        y = self.y_normalizer.decode(y)
        loss = self.loss(out, y)

        # vtk
        output_meshs = []
        for i in range(len(vtk_paths)):
            vtk_path = vtk_paths[i]
            mesh = pv.read(vtk_path)
            # print("read vtk file:", vtk_path)
            cell_pressure = mesh.cell_data.get('Pressure').flatten()
            cell_wall_ishear_stress_i = mesh.cell_data.get('WallShearStressi').flatten()
            cell_wall_shear_stress_j = mesh.cell_data.get('WallShearStressj').flatten()
            cell_wall_shear_stress_k = mesh.cell_data.get('WallShearStressk').flatten()

            test_pressure = out[i, :, 0].cpu().numpy()
            test_wall_ishear_stress_i = out[i, :, 1].cpu().numpy()
            test_wall_shear_stress_j = out[i, :, 2].cpu().numpy()
            test_wall_shear_stress_k = out[i, :, 3].cpu().numpy()
            
            diff_pressure = test_pressure - cell_pressure
            diff_wall_ishear_stress_i = test_wall_ishear_stress_i - cell_wall_ishear_stress_i
            diff_wall_shear_stress_j = test_wall_shear_stress_j - cell_wall_shear_stress_j
            diff_wall_shear_stress_k = test_wall_shear_stress_k - cell_wall_shear_stress_k
        
            mesh.cell_data['test_Pressure'] = test_pressure
            mesh.cell_data['test_WallShearStressi'] = test_wall_ishear_stress_i
            mesh.cell_data['test_WallShearStressj'] = test_wall_shear_stress_j
            mesh.cell_data['test_WallShearStressk'] = test_wall_shear_stress_k
            mesh.cell_data['diff_Pressure'] = diff_pressure
            mesh.cell_data['diff_WallShearStressi'] = diff_wall_ishear_stress_i
            mesh.cell_data['diff_WallShearStressj'] = diff_wall_shear_stress_j
            mesh.cell_data['diff_WallShearStressk'] = diff_wall_shear_stress_k

            output_meshs.append(mesh)

        results = {'output_meshs': output_meshs ,'vtk_paths': vtk_paths}
        self.test_step_outputs .append(results)
        return results
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        all_output_meshs = []
        all_vtk_paths = []

        # output_meshsvtk_paths
        for result in outputs:
            all_output_meshs.extend(result['output_meshs'])
            all_vtk_paths.extend(result['vtk_paths'])

        # vtk
        for i in range(len(all_output_meshs)):
            mesh_i = all_output_meshs[i]
            vtk_paths = all_vtk_paths[i]
            writer_path = self.writer_path
            if not os.path.exists(os.path.join(writer_path, 'output')):
                os.makedirs(os.path.join(writer_path, 'output')) 
            mesh_i.save(os.path.join(writer_path, 'output', os.path.basename(vtk_paths)))
            print("save vtk file:", os.path.join(writer_path, 'output', os.path.basename(vtk_paths)))
       
        # return all_output_meshs, all_vtk_paths

    def predict_step(self, batch, batch_idx):
        pass
    
    def predict_epoch_end(self, outputs):
        preds = torch.cat(outputs)
        print("Prediction Results DataFrame:\n", preds.mean())

    def configure_optimizers(self):
        # 
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        # 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # epoch
                'frequency': 1
            }
        }

    def on_before_optimizer_step(self, optimizer):
        # 
        if self.params.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.params.max_grad_norm)
