import pdb

from torch import nn
import torch
import warnings
import math

from model.transformer.weight_init import xavier_init, constant_init
from multi_scale_deformable_attn_3D_custom_function import MultiScaleDeformableAttn3DCustomFunction_fp32

class SpatialSelfAttention3D(nn.Module):
    def __init__(self, args, dtype = torch.float, num_queue = 2):
        super(SpatialSelfAttention3D, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(0.1)
        self.dims = args.dim_num
        self.batch_first = False
        self.num_head = args.head_num
        self.num_levels = args.self_level
        self.all_sampling_points = args.self_sampling_points
        if self.dims % self.num_head != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {self.dims} and {self.num_head}')
        dim_per_head = self.dims // self.num_head
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')
        self.im2col_step = 64
        self.num_queue = num_queue
        self.sampling_offsets = nn.Linear(self.dims * self.num_queue, self.num_queue * self.num_head * self.num_levels * self.all_sampling_points * 3)
        self.attention_weights = nn.Linear(self.dims * self.num_queue, self.num_queue * self.num_head * self.num_levels * self.all_sampling_points)
        self.value_proj = nn.Linear(self.dims, self.dims)
        self.output_proj = nn.Linear(self.dims, self.dims)
        self.init_weights()


    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_head,
            dtype=torch.float32) * (2.0 * math.pi / self.num_head)
        grid_init = torch.stack([thetas.sin(), thetas.cos(), thetas*0], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_head, 1, 1,
            3).repeat(1, self.num_levels*self.num_queue, self.all_sampling_points, 1)
        for i in range(self.all_sampling_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self, query, key, value,
                identity = None,
                query_pose = None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                ):
        bs, num_query, dims = query.shape
        value = torch.stack([query, query], 1).reshape(bs*2, num_query, dims)
        # print("0: ", query)
        if identity == None:
            identity = query
        if query_pose is not None:
            query = query + query_pose
        # print("1: ", query)
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]* spatial_shapes[:, 2]).sum() == num_value
        assert self.num_queue == 2


        query = torch.cat([value[:bs], query], -1)
        "V = kv * x"
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(bs*self.num_queue,
                              num_value, self.num_head, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_head,  self.num_queue, self.num_levels, self.all_sampling_points, 3)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_head, self.num_queue, self.num_levels * self.all_sampling_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_head,
                                                   self.num_queue,
                                                   self.num_levels,
                                                   self.all_sampling_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_queue, num_query, self.num_head, self.num_levels, self.all_sampling_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_queue, num_query, self.num_head, self.num_levels, self.all_sampling_points, 3)

        assert reference_points.shape[-1] == 3
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0],spatial_shapes[..., 2]], -1)#hwz
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]
        MSDA_function = MultiScaleDeformableAttn3DCustomFunction_fp32
        output = MSDA_function.apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = output.permute(1, 2, 0)
        output = output.view(num_query, dims, bs, self.num_queue)
        output = output.mean(-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        # print("output: ", output)
        return self.dropout(output) + identity
