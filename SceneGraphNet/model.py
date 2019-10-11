import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from utils.default_settings import dic_id2type
import copy
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' to_torch Variable '''
def to_torch(n, torch_type=torch.FloatTensor, requires_grad=False, dim_0=1):
    n = torch.tensor(n, requires_grad=requires_grad).type(torch_type).to(device)
    n = n.view(dim_0, -1)
    return n

def get_gt_k_vec(node_list, cur_node, opt_parser):
    """
    Get cur_node's k-vec = category + dimension + position
    :param node_list:
    :param cur_node:
    :param opt_parser:
    :return:
    """

    if (node_list[cur_node]['type'] == 'root'):
        cat = 'wall'
        dim_vec = [0.0] * 3
        pos_vec = [0.0] * 3
    elif (node_list[cur_node]['type'] == 'wall'):
        cat = 'wall'
        dim_vec = node_list[cur_node]['self_info']['dim']
        pos_vec = node_list[cur_node]['self_info']['translation']
    else:
        if(len(node_list[cur_node]['self_info']['node_model_id']) > 2 and
                node_list[cur_node]['self_info']['node_model_id'][0:2] == 'EX'):
            cat = node_list[cur_node]['self_info']['node_model_id'][3:]
        else:
            cat = dic_id2type[node_list[cur_node]['self_info']['node_model_id']][1]
        dim_vec = node_list[cur_node]['self_info']['dim']
        pos_vec = node_list[cur_node]['self_info']['translation']

    cat_vec = [0.0] * (len(opt_parser.cat2id.keys()) + 1)
    cat_vec[int(opt_parser.cat2id[cat])] = 1.0

    return cat_vec + dim_vec + pos_vec


class AggregateMaxPoolEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(AggregateMaxPoolEnc, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, d),
            # nn.Tanh()
        )
        self.msg = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=[False]):
        if (cat_msg[0]):
            msg = self.msg(torch.cat((cur_d_vec, d_vec), dim=1))
        else:
            msg = d_vec

        d_vec = self.enc(msg) * w
        compare = torch.stack((pre_vec, d_vec), dim=2)
        d_vec, _ = torch.max(compare, 2)
        d_vec.view(d_vec.shape)
        return d_vec

class AggregateSumEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(AggregateSumEnc, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, d),
            # nn.Tanh()
        )
        self.msg = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=[False]):
        if (cat_msg[0]):
            msg = self.msg(torch.cat((cur_d_vec, d_vec), dim=1))
        else:
            msg = d_vec

        d_vec = pre_vec + self.enc(msg) * w
        return d_vec

class AggregateCatEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(AggregateCatEnc, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(d * 2, h),
            nn.ReLU(),
            nn.Linear(h, d),
            # nn.Tanh()
        )
        self.msg = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=[False]):
        if (cat_msg[0]):
            msg = self.msg(torch.cat((cur_d_vec, d_vec), dim=1))
        else:
            msg = d_vec

        feat = torch.cat((pre_vec, msg), dim=1)
        d_vec = self.enc(feat) * w
        return d_vec

class AggregateGRUEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(AggregateGRUEnc, self).__init__()

        self.w_x = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )
        self.w_h = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, d)
        )
        self.msg = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=[False]):
        if(cat_msg[0]):
            msg = self.msg(torch.cat((cur_d_vec, d_vec), dim=1))
        else:
            msg = d_vec

        ht = self.w_h(pre_vec) + self.w_x(msg) * w
        # ht = self.act(ht)
        return ht


class UpdateEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(UpdateEnc, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(d * 7, h),
            nn.ReLU(),
            nn.Linear(h, d),
            # nn.Tanh()
        )

    def forward(self, self_vec, p_sup_vec, c_sup_vec, p_sur_vec, c_sur_vec, n_vec, co_vec):
        feat = torch.cat((self_vec, p_sup_vec, c_sup_vec, p_sur_vec, c_sur_vec, n_vec, co_vec), dim=1)
        d_vec = self.enc(feat)
        return d_vec


class BoxEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(BoxEnc, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(k, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, k_vec):
        d_vec = self.enc(k_vec)
        return d_vec


class LearnedWeight(nn.Module):
    def __init__(self, k=54, h=300, dis_vec_dim=3):
        super(LearnedWeight, self).__init__()

        self.offset_enc = nn.Sequential(
            nn.Linear(dis_vec_dim, h),
            nn.ReLU(),
            nn.Linear(h, k)
        )

        self.k_offset_enc = nn.Sequential(
            nn.Linear(k * 3, h),
            nn.ReLU(),
            nn.Linear(h, 1),
            nn.Sigmoid()
        )

    def forward(self, k_vec1, k_vec2, offset_vec):
        offset_vec = self.offset_enc(offset_vec)
        k_offset_vec = torch.cat((k_vec1, k_vec2, offset_vec), dim=1)
        w = self.k_offset_enc(k_offset_vec)
        return w


class FullEnc(nn.Module):
    def __init__(self, k=55, d=100, h=300, aggregate_func='GRU'):
        super(FullEnc, self).__init__()

        if(aggregate_func == 'Sum'):
            AggregateEnc = AggregateSumEnc
        elif(aggregate_func == 'GRU'):
            AggregateEnc = AggregateGRUEnc
        elif(aggregate_func == 'MaxPool'):
            AggregateEnc = AggregateMaxPoolEnc
        elif (aggregate_func == 'CatRNN'):
            print('CatAggregate')
            AggregateEnc = AggregateCatEnc
        else:
            AggregateEnc = None
            print('Aggregation function selection error')
            exit(-1)

        self.aggregate_neighbor_enc = AggregateEnc(k, d, h)
        self.aggregate_child_supp_enc = AggregateEnc(k, d, h)
        self.aggregate_child_surr_enc = AggregateEnc(k, d, h)
        self.aggregate_parent_supp_enc = AggregateEnc(k, d, h)
        self.aggregate_parent_surr_enc = AggregateEnc(k, d, h)
        self.aggregate_cooc_enc = AggregateEnc(k, d, h)

        dis_vec_dim = 3

        self.learned_weight = LearnedWeight(k, h, dis_vec_dim=dis_vec_dim)

        self.aggregate_self_enc = UpdateEnc(k, d, h)

        self.box_enc = BoxEnc(k, d, h)

    def aggregate_neighbor_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
        return self.aggregate_neighbor_enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)

    def aggregate_child_supp_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
        return self.aggregate_child_supp_enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)

    def aggregate_child_surr_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
        return self.aggregate_child_surr_enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)

    def aggregate_parent_supp_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
        return self.aggregate_parent_supp_enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)

    def aggregate_parent_surr_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
        return self.aggregate_parent_surr_enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)

    def aggregate_cooc_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
        return self.aggregate_cooc_enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)

    def learned_weight_func(self, k_vec1, k_vec2, offset_vec):
        return self.learned_weight(k_vec1, k_vec2, offset_vec)


    def aggregate_self_func(self, self_vec, p_sup_vec, c_sup_vec, p_sur_vec, c_sur_vec, n_vec, co_vec):
        return self.aggregate_self_enc(self_vec, p_sup_vec, c_sup_vec, p_sur_vec, c_sur_vec, n_vec, co_vec)

    def cat_self_func(self, self_vec, p_sup_vec, c_sup_vec, p_sur_vec, c_sur_vec, n_vec, co_vec):
        return torch.cat((self_vec, p_sup_vec, c_sup_vec, p_sur_vec, c_sur_vec, n_vec, co_vec), dim=1)

    def box_enc_func(self, k_vec):
        return self.box_enc(k_vec)


def encode_tree_fold(fold, raw_node_list, rand_path, opt_parser):
    node_list = copy.deepcopy(raw_node_list)
    d_vec_dim = opt_parser.d_vec_dim

    encode_fold_list = []
    rand_path_node_name_order = []
    tree_leaf_node = rand_path[-1]

    def encode_node(node_list, leaf_node=tree_leaf_node, step=0):
        """
        Graph message passing by torchfold encoding
        :param node_list:
        :param leaf_node:
        :param step:
        :return:
        """

        # init d-vec for all nodes
        if(step == 0):

            # loop to get each node's k-vec and d-vec
            for cur_node in node_list.keys():

                # if leaf node, reset its k-vec to all-zeros to represent it is missing
                if(cur_node == leaf_node):
                    missing_cat = [0.0] * len(opt_parser.cat2id.keys()) + [1.0]
                    missing_dim_pos = [0.0] * 3 + node_list[cur_node]['self_info']['translation']
                    node_list[cur_node]['k-vec'] =  missing_cat + missing_dim_pos
                else:
                    node_list[cur_node]['k-vec'] = get_gt_k_vec(node_list, cur_node, opt_parser)

                node_list[cur_node]['k-vec'] = to_torch(node_list[cur_node]['k-vec'])
                node_list[cur_node]['d-vec'] = fold.add('box_enc_func', node_list[cur_node]['k-vec'])
                node_list[cur_node]['w'] = {}
                node_list[cur_node]['dis'] = {}

            # loop to get each pair of neighbor nodes' attention weight
            for cur_node in node_list.keys():
                for neighbor_node in node_list.keys():

                    dis_feat_vec = [node_list[cur_node]['self_info']['translation'][0] -
                                    node_list[neighbor_node]['self_info']['translation'][0],
                                    node_list[cur_node]['self_info']['translation'][1] -
                                    node_list[neighbor_node]['self_info']['translation'][1],
                                    node_list[cur_node]['self_info']['translation'][2] -
                                    node_list[neighbor_node]['self_info']['translation'][2]]
                    dis = np.sqrt(dis_feat_vec[0] ** 2 + dis_feat_vec[1] ** 2 + dis_feat_vec[2] ** 2)
                    dis_feat = dis_feat_vec

                    node_list[cur_node]['dis'][neighbor_node] = dis
                    node_list[cur_node]['w'][neighbor_node] = \
                        fold.add('learned_weight_func', node_list[cur_node]['k-vec'],
                                                                 node_list[neighbor_node]['k-vec'],
                                                                 to_torch(dis_feat))

        # graph message passing
        else:
            for cur_node in node_list.keys():
                cur_node_d_vec = node_list[cur_node]['pre-d-vec']

                # message from parents (supported-by, surrounded-by relation)
                aggregate_parent_d_vec ={'supp' : to_torch([0.0] * d_vec_dim), 'surr' : to_torch([0.0] * d_vec_dim)}
                for parent_node, parent_node_type in node_list[cur_node]['parents']:
                    parent_d_vec = node_list[parent_node]['pre-d-vec']
                    aggregate_parent_d_vec[parent_node_type] = \
                        fold.add('aggregate_parent_{}_func'.format(parent_node_type),
                                 parent_d_vec,
                                 aggregate_parent_d_vec[parent_node_type],
                                 cur_node_d_vec,
                                 to_torch([1.]),
                                 to_torch(opt_parser.cat_msg, torch.bool))

                # message from siblings (next-to relation)
                aggregate_neighbor_d_vec = to_torch([0.0] * d_vec_dim)
                for sibling_node_i, _ in node_list[cur_node]['siblings']:
                    sibling_node_d_vec = node_list[sibling_node_i]['pre-d-vec']
                    aggregate_neighbor_d_vec = fold.add('aggregate_neighbor_func',
                                                        sibling_node_d_vec,
                                                        aggregate_neighbor_d_vec,
                                                        cur_node_d_vec,
                                                        to_torch([1.]),
                                                        to_torch([opt_parser.cat_msg], torch.bool))

                # message from childs (supporting, surrounding relation)
                aggregate_child_d_vec = {'supp' : to_torch([0.0] * d_vec_dim), 'surr' : to_torch([0.0] * d_vec_dim)}
                for child_node_i, child_node_type_i in node_list[cur_node]['childs']:
                    child_node_d_vec = node_list[child_node_i]['pre-d-vec']
                    aggregate_child_d_vec[child_node_type_i] = \
                        fold.add('aggregate_child_{}_func'.format(child_node_type_i),
                                 child_node_d_vec,
                                 aggregate_child_d_vec[child_node_type_i],
                                 cur_node_d_vec,
                                 to_torch([1.]),
                                 to_torch([opt_parser.cat_msg], torch.bool))

                # message from loose neighbors (co-occurring relation)
                aggregate_cooc_d_vec = to_torch([0.0] * d_vec_dim)
                if(opt_parser.aggregate_in_order):
                    all_neighbor_nodes = list(node_list.keys())
                    all_neighbor_nodes.sort(key=lambda  x: node_list[cur_node]['dis'][x], reverse=True)
                else:
                    all_neighbor_nodes = node_list.keys()
                for neighbor_node in all_neighbor_nodes:
                    if (neighbor_node != cur_node):
                        w = node_list[cur_node]['w'][neighbor_node]
                        neighbor_node_d_vec = node_list[neighbor_node]['pre-d-vec']
                        aggregate_cooc_d_vec = fold.add('aggregate_cooc_func',
                                                            neighbor_node_d_vec,
                                                            aggregate_cooc_d_vec,
                                                            cur_node_d_vec,
                                                            w,
                                                            to_torch([opt_parser.cat_msg], torch.bool))

                node_list[cur_node]['d-vec'] = fold.add('aggregate_self_func',
                                                        node_list[cur_node]['pre-d-vec'],
                                                        aggregate_parent_d_vec['supp'],
                                                        aggregate_child_d_vec['supp'],
                                                        aggregate_parent_d_vec['surr'],
                                                        aggregate_child_d_vec['surr'],
                                                        aggregate_neighbor_d_vec,
                                                        aggregate_cooc_d_vec)
                node_list[cur_node]['cat-d-vec'] = fold.add('cat_self_func',
                                                            node_list[cur_node]['pre-d-vec'],
                                                            aggregate_parent_d_vec['supp'],
                                                            aggregate_child_d_vec['supp'],
                                                            aggregate_parent_d_vec['surr'],
                                                            aggregate_child_d_vec['surr'],
                                                            aggregate_neighbor_d_vec,
                                                            aggregate_cooc_d_vec)

        # end of func


    for i in range(opt_parser.K):
        encode_node(node_list, leaf_node=tree_leaf_node, step=i)
        for cur_node in node_list.keys():
            node_list[cur_node]['pre-d-vec'] = node_list[cur_node]['d-vec']


    # encode all d-vec along rand path
    if(opt_parser.decode_cat_d_vec == True):
        # use cat version d-vec for decoding
        encode_fold_list.append(node_list[tree_leaf_node]['cat-d-vec'])
    else:
        # use normal version d-vec for decoding
        encode_fold_list.append(node_list[tree_leaf_node]['d-vec'])
    rand_path_node_name_order.append(tree_leaf_node)

    return encode_fold_list, rand_path_node_name_order



class Root_to_leaf_Dec(nn.Module):
    def __init__(self, k=54, d=100, r=28, h=300):
        super(Root_to_leaf_Dec, self).__init__()

        self.dec = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, d)
        )

    def forward(self, parent_d_vec):
        dec_vec = self.dec(parent_d_vec)
        return dec_vec

class Cat_Root_to_leaf_Dec(nn.Module):
    def __init__(self, k=54, d=100, r=28, h=300):
        super(Cat_Root_to_leaf_Dec, self).__init__()

        self.dec = nn.Sequential(
            nn.Linear(d * 7, h),
            nn.ReLU(),
            nn.Linear(h, d)
        )

    def forward(self, parent_d_vec):
        dec_vec = self.dec(parent_d_vec)
        return dec_vec

class BoxDec(nn.Module):
    def __init__(self, k=54, d=100, r=28, h=300):
        super(BoxDec, self).__init__()

        self.dec = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, k)
        )

    def forward(self, d_vec):
        k_vec = self.dec(d_vec)
        return k_vec

class FullDec(nn.Module):
    def __init__(self, k=55, d=100, r=28, h=300, root_d=350, root_h=1050):
        super(FullDec, self).__init__()

        self.root_to_leaf_dec = Root_to_leaf_Dec(k, d, r, h)
        self.cat_root_to_leaf_dec = Cat_Root_to_leaf_Dec(k, d, r, h)
        self.box_dec = BoxDec(k, d, r, h)

    def root_dec_func(self, parent_d_vec):
        return self.root_to_leaf_dec(parent_d_vec)

    def cat_root_dec_func(self, parent_d_vec):
        return self.cat_root_to_leaf_dec(parent_d_vec)

    def box_dec_func(self, d_vec):
        return self.box_dec(d_vec)

    def forward(self, d_vec):
        box_d_vec = self.root_to_leaf_dec(d_vec)
        k_vec = self.box_dec(box_d_vec)

        return k_vec

def decode_tree_fold(fold, d_vec, opt_parser=None):
    if(opt_parser != None and opt_parser.decode_cat_d_vec == True):
        leaf_node_d_vec = fold.add('cat_root_dec_func', d_vec)
    else:
        leaf_node_d_vec = fold.add('root_dec_func', d_vec)
    return fold.add('box_dec_func', leaf_node_d_vec)
