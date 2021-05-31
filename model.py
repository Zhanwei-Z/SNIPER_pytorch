import torch.nn as nn
import torch
import torch.nn.functional as F
from convlstm import ConvLSTM


class Evolution(nn.Module):

    def __init__(self, dr2, **kwargs):
        super(Evolution, self).__init__(**kwargs)
        self.dr2 = dr2
        self.w1 = nn.Parameter(torch.ones(2 * self.dr2, self.dr2))

    def forward(self, all_data_static, thre_nc, all_data_dynamic_now):
        all_data_dynamic = [torch.unsqueeze(all_data_dynamic_now, dim=0)]

        all_data_dynamic_diff = []

        for i in range(1, len(thre_nc)):
            all_data_dynamic_now_diff = all_data_dynamic_now
            all_data_dynamic_now = torch.sigmoid(
                torch.matmul(torch.cat([all_data_dynamic_now, all_data_static[i]], axis=-1), self.w1) \
                * thre_nc[i].repeat(1, 1, self.dr2) + all_data_dynamic_now \
                * (1 - thre_nc[i]).repeat(1, 1, self.dr2))
            all_data_dynamic_now_diff = all_data_dynamic_now - all_data_dynamic_now_diff
            all_data_dynamic_diff.append(torch.unsqueeze(all_data_dynamic_now_diff, 0))
            all_data_dynamic.append(torch.unsqueeze(all_data_dynamic_now, 0))

        all_data_dynamic = torch.cat(all_data_dynamic, axis=0)
        all_data_dynamic_diff = torch.cat(all_data_dynamic_diff, axis=0)
        return all_data_dynamic, all_data_dynamic_now, all_data_dynamic_diff


class Attention(nn.Module):

    def __init__(self, dr2, len_recent_time, number_region, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dr2 = dr2
        self.len_recent_time = len_recent_time
        self.number_region = number_region
        self.wq = nn.Parameter(torch.zeros(self.dr2, self.dr2))
        self.wk = nn.Parameter(torch.zeros(self.dr2, self.dr2))
        self.wd_s = nn.Parameter(torch.zeros(self.dr2, self.dr2))

    def forward(self, data, neigh_index):  # len,time,regions,feas
        dataneigh = data.permute(2, 0, 1, 3).contiguous()[
            neigh_index]  # regions,len,time,feas->regions,neigh,len,time,feas
        dataneigh = dataneigh.permute(2, 3, 0, 1, 4).contiguous()  # len,time,regions,neigh,feas
        data = torch.unsqueeze(data, dim=3)
        data = torch.matmul(data, self.wq)
        dataneigh = torch.matmul(dataneigh, self.wk)
        out = torch.matmul(F.softmax(torch.matmul(data, dataneigh.transpose(-2, -1).contiguous()), dim=-1), dataneigh)
        out = data + out
        out = torch.sigmoid(torch.matmul(torch.squeeze(out, dim=3), self.wd_s))
        return out


class MultiAttention(nn.Module):

    def __init__(self, num_sp, dr2, len_recent_time, number_region, **kwargs):
        super(MultiAttention, self).__init__(**kwargs)
        self.dr2 = dr2
        self.num_sp = num_sp
        self.attention_layers_poi = nn.ModuleList()
        for i in range(self.num_sp):
            self.attention_layers_poi.append(Attention(self.dr2, len_recent_time, number_region))
        self.attention_layers_road = nn.ModuleList()
        for i in range(self.num_sp):
            self.attention_layers_road.append(Attention(self.dr2, len_recent_time, number_region))
        self.attention_layers_record = nn.ModuleList()
        for i in range(self.num_sp):
            self.attention_layers_record.append(Attention(self.dr2, len_recent_time, number_region))

        self.w_poi = nn.Parameter(torch.zeros(self.dr2, ))
        self.w_road = nn.Parameter(torch.zeros(self.dr2, ))
        self.w_record = nn.Parameter(torch.zeros(self.dr2, ))

    def forward(self, all_data, neigh_poi_index, neigh_road_index, neigh_record_index):  #
        all_data_static_poi = all_data
        all_data_static_road = all_data
        all_data_static_record = all_data
        for i in range(self.num_sp):
            all_data_static_poi = self.attention_layers_poi[i](all_data_static_poi, neigh_poi_index)
            all_data_static_road = self.attention_layers_road[i](all_data_static_road, neigh_road_index)
            all_data_static_record = self.attention_layers_record[i](all_data_static_record, neigh_record_index)
        out = torch.sigmoid(all_data_static_poi * self.w_poi + all_data_static_road * self.w_road + \
                            all_data_static_record * self.w_record)
        return out


class SNIPER(nn.Module):
    def __init__(self, dr, len_recent_time, number_sp, number_region,
                 neigh_poi_index, neigh_road_index, neigh_record_index, **kwargs):
        super(SNIPER, self).__init__(**kwargs)
        self.neigh_poi_index = neigh_poi_index
        self.neigh_road_index = neigh_road_index
        self.neigh_record_index = neigh_record_index
        self.evolution = Evolution(dr * 2)
        self.multiattention = nn.ModuleList()
        for i in range(2):
            self.multiattention.append(MultiAttention(number_sp, 2 * dr, len_recent_time, number_region))

        self.convlstm = ConvLSTM(input_dim=4 * dr,
                                 hidden_dim=[1],
                                 kernel_size=(1, 1),
                                 num_layers=1,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=False)
        self.final_layer = nn.Linear(in_features=number_region, out_features=number_region, bias=True)
        self.final_layer.bias.data.fill_(1)

    def forward(self, all_data):
        all_data_static, thre_nc, all_data_dynamic_now = all_data[0], all_data[1], all_data[2]
        all_data_dynamic, all_data_dynamic_now, all_data_dynamic_diff = self.evolution(all_data_static, thre_nc,
                                                                                       all_data_dynamic_now)
        all_data_dynamic = self.multiattention[0](all_data_dynamic, self.neigh_poi_index, self.neigh_road_index,
                                                  self.neigh_record_index)
        all_data_static = self.multiattention[1](all_data_static, self.neigh_poi_index, self.neigh_road_index,
                                                 self.neigh_record_index)
        all_data = torch.cat([all_data_dynamic, all_data_static], axis=-1)
        all_data = all_data.permute(0, 1, 3, 2).contiguous()
        all_data = torch.unsqueeze(all_data, dim=-1)
        all_data = self.convlstm(all_data)[1][0][0]
        all_data = torch.squeeze(all_data)
        all_data = torch.sigmoid(self.final_layer(all_data))
        # print(all_data.shape)
        return all_data, all_data_dynamic_now, all_data_dynamic_diff
