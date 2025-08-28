from torch import nn
import torch
import warnings
import math

from model.transformer.weight_init import xavier_init, constant_init
from model.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch
import pdb

class SpatialSatCrossAttention(nn.Module):
    def __init__(self, args, dtype = torch.float):
        super(SpatialSatCrossAttention, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(0.1)
        self.dims = args.dim_num
        self.output_proj = nn.Linear(self.dims, self.dims)
        self.batch_first = False
        self.init_weight()
        self.sample_num = args.height_num
        self.num_cam = args.number_of_cam
        self.deformable_attention = DeformableSatAttention(args)

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, query, key, value,
                res=None,
                slots = None,
                query_pos = None,
                inp_residual = None,
                ref_points=None,
                vox_mask=None,
                spatial_shapes=None,
                level_start_index=None):

        if res is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.shape
        D = ref_points.size(3)
        "find the vox grid  visiable in camera"
        indexes = []
        for i, mask_per_img in enumerate(vox_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros([bs, self.num_cam, max_len, self.dims])
        reference_points_rebatch = ref_points.new_zeros([bs, self.num_cam, max_len, D, 2])
        for j in range(bs):
            for i, reference_points_per_img in enumerate(ref_points):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        cam, l, bs, dims = key.shape
        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cam, l, self.dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cam, l, self.dims)
        queries = self.deformable_attention(queries_rebatch.view(bs * self.num_cam, max_len, self.dims),
                                 key,
                                 value,
                                 reference_points=reference_points_rebatch.view(bs * self.num_cam, max_len, D, 2),
                                 spatial_shapes=spatial_shapes,
                                 level_start_index=level_start_index).view(bs, self.num_cam, max_len, self.dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = vox_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


class DeformableSatAttention(nn.Module):
    def __init__(self, args, dtype = torch.float):
        super(DeformableSatAttention, self).__init__()
        self.args = args
        self.dims = args.dim_num
        self.head_num = args.head_num
        self.num_levels = args.sat_level
        self.output_proj = None
        self.all_sampling_points = args.all_sampling_points
        self.im2col_step = 64
        dim_per_head = self.dims // self.head_num
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

        self.sampling_offsets = nn.Linear(
            self.dims, self.head_num * self.num_levels * self.all_sampling_points * 2)
        self.attention_weights = nn.Linear(self.dims,
                                           self.head_num * self.num_levels * self.all_sampling_points)
        self.value_proj = nn.Linear(self.dims, self.dims)
        self.init_weights()


    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.head_num,
            dtype=torch.float32) * (2.0 * math.pi / self.head_num)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.head_num, 1, 1,
            2).repeat(1, self.num_levels, self.head_num, 1)
        for i in range(self.head_num):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True


    def forward(self, query, key, value,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None):


        if query_pos is not None:
            query = query + query_pos
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        "V = kv * x"
        # print("max_value: ", torch.max(value))
        # print("value: ", value)
        # print("max_weight: ", torch.max(self.value_proj.weight))
        # print("weight: ", self.value_proj.weight)
        value = self.value_proj(value)
        # print(value)
        # print("value 1: ", value)
        "add mask for the value"
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.head_num, -1)
        "generate the sampling_offsets and attention_weights by linear process the query"
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.head_num, self.num_levels, self.all_sampling_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.head_num, self.num_levels * self.all_sampling_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.head_num, self.num_levels, self.all_sampling_points)
        "Do the attention"
        """
        For each BEV query, it owns `num_points` in 3D space that having different heights.
        After proejcting, each BEV query has `num_points` reference points in each 2D image.
        For each referent point, we sample 'num_shift_points` sampling points.
        For `one` reference points,  it has overall `num_points * num_shift_points` sampling points.
        """
        "exchange the first and second columns of spatial_shapes"
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        bs, num_query, num_points, uv = reference_points.shape
        reference_points = reference_points[:, :, None, None, None, :, :]
        "normalize the offsets, for example, if the offsets is (u, v), then the output is (u/width, v/height) "
        sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        bs, num_query, num_heads, num_levels, all_sampling_points, uv = sampling_offsets.shape
        sampling_offsets = sampling_offsets.view(
            bs, num_query, num_heads, num_levels, all_sampling_points // num_points, num_points, uv)
        sampling_locations = reference_points + sampling_offsets
        bs, num_query, num_heads, num_levels, num_shift_points, num_points, uv = sampling_locations.shape
        assert all_sampling_points == num_shift_points * num_points
        sampling_locations = sampling_locations.view(bs, num_query, num_heads, num_levels, all_sampling_points, uv)

        "feed the value, spatial shape, sampling localization, and attention weight into Multi-scale-attention module"
        MSDA_function = MSDeformAttnFunction
        # print("value: ", value)
        # print("spatial_shapes: ", spatial_shapes)
        # print("level_start_index: ", level_start_index)
        # print("sampling_locations: ", sampling_locations)
        # print("attention_weights: ", attention_weights)
        output = MSDA_function.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)
        # print(attention_weights)
        return output
