import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from utils.default_settings import *
from utils.utl import weight_init
import copy
import SceneGraphNet.model as model
from SceneGraphNet.model import to_torch, device
from tensorboardX import SummaryWriter
from random import shuffle
import torchfold
import time


class train_model():
    def __init__(self, opt_parser):
        """
        Initialize training process
        :param opt_parser: args parser
        """

        ''' Input args parser '''
        self.opt_parser = opt_parser
        self.model = model

        ''' Initialize model '''
        k_size = k_size_dic[opt_parser.room_type]

        # encoder
        self.full_enc = self.model.FullEnc(k=k_size, d=opt_parser.d_vec_dim, h=opt_parser.h_vec_dim,
                                           aggregate_func=opt_parser.aggregation_func)
        self.full_enc.apply(weight_init)
        self.full_enc.to(device)

        # decoder
        self.full_dec = self.model.FullDec(k=k_size, d=opt_parser.d_vec_dim, h=opt_parser.h_vec_dim)
        self.full_dec.apply(weight_init)
        self.full_dec.to(device)

        ''' Setup Optimizer '''
        self.opt = {}
        self.opt['full_enc'] = optim.Adam(self.full_enc.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)
        self.opt['full_dec'] = optim.Adam(self.full_dec.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

        ''' Load pre-trained model '''
        self.pretrained_epoch = 0
        if opt_parser.ckpt != '':
            ckpt = torch.load(opt_parser.ckpt)

            def update_partial_dict(model, pretrained_ckpt):
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_ckpt.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                return model_dict

            self.full_enc.load_state_dict(update_partial_dict(self.full_enc, ckpt['full_enc_state_dict']))
            self.full_dec.load_state_dict(update_partial_dict(self.full_dec, ckpt['full_dec_state_dict']))
            if(opt_parser.load_model_along_with_optimizer):
                self.opt['full_enc'].load_state_dict(ckpt['full_enc_opt'])
                self.opt['full_dec'].load_state_dict(ckpt['full_dec_opt'])
            self.pretrained_epoch = ckpt['epoch']
            print("=========== LOAD PRE TRAINED MODEL ENCODER: " + opt_parser.ckpt + " ==================")

        ''' Loss function '''
        self.LOSS_CLS = torch.nn.CrossEntropyLoss()
        self.LOSS_L2 = torch.nn.MSELoss()

        ''' Output Summary in tensorboard '''
        if (opt_parser.write):
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, opt_parser.name))

        ''' load valid rooms '''
        f = open(os.path.join(pkl_dir, '{}_data.json'.format(opt_parser.room_type)))
        self.valid_rooms = json.load(f)
        self.valid_rooms_train = self.valid_rooms[0:opt_parser.num_train_rooms]
        self.valid_rooms_test = self.valid_rooms[opt_parser.num_train_rooms:opt_parser.num_train_rooms + opt_parser.num_test_rooms]

        self.MAX_ACC = -1.0
        self.MAX_ACC = -1.0
        self.MIN_LOSS = 1e10
        self.MIN_DIM_LOSS = 1e10
        self.STATE = 'INIT'

    ''' useful functions '''
    def find_root_to_leaf_node_path(self, node_list, cur_node):
        """
        find a list of paths from tree root to tree leaf nodes (in a recursive way)
        :param node_list: tree node list
        :param cur_node: current root node
        :return:
        """
        root_to_leaf_path = []

        # middle node
        if (len(node_list[cur_node]['co-occurrence']) + len(node_list[cur_node]['surround']) +
                len(node_list[cur_node]['support']) > 0):
            for child in node_list[cur_node]['co-occurrence']:
                child_node_list = self.find_root_to_leaf_node_path(node_list, child)
                p = [[cur_node] + c for c in child_node_list]
                root_to_leaf_path += p
            for child in node_list[cur_node]['support']:
                child_node_list = self.find_root_to_leaf_node_path(node_list, child)
                p = [[cur_node] + c for c in child_node_list]
                root_to_leaf_path += p
            for child in node_list[cur_node]['surround']:
                for key in child.keys():
                    child_node_list = self.find_root_to_leaf_node_path(node_list, key)
                    child_node_list += self.find_root_to_leaf_node_path(node_list, child[key])
                    p = [[cur_node] + c for c in child_node_list]
                    root_to_leaf_path += p
            return root_to_leaf_path
        # leaf node
        else:
            return [[cur_node]]

    def find_selected_node_list(self, node_list, cur_node):
        """
        find all nodes (or keys) under the subtree with root node=cur_node
        :param node_list: entire tree node list
        :param cur_node: current root node
        :return:
        """
        sub_keys = []

        # middle node
        if (len(node_list[cur_node]['co-occurrence']) + len(node_list[cur_node]['surround']) +
                len(node_list[cur_node]['support']) > 0):
            for child in node_list[cur_node]['co-occurrence']:
                child_sub_keys = self.find_selected_node_list(node_list, child)
                sub_keys += child_sub_keys

            for child in node_list[cur_node]['support']:
                child_sub_keys = self.find_selected_node_list(node_list, child)
                sub_keys += child_sub_keys

            for child in node_list[cur_node]['surround']:
                for key in child.keys():
                    child_sub_keys = self.find_selected_node_list(node_list, key)
                    child_sub_keys += self.find_selected_node_list(node_list, child[key])
                    sub_keys += child_sub_keys
            sub_keys += [cur_node]
            return sub_keys
        # leaf node
        else:
            return [cur_node]

    def find_parent_sibling_child_list(self, node_list):
        """
        find parent / sibling / child nodes for each node in node_list
        :param node_list:
        :return:
        """

        # init
        for node in node_list.keys():
            for relation in ['parents', 'siblings', 'childs']:
                node_list[node][relation] = []

        # parents and childs
        for node in node_list.keys():
            for c in node_list[node]['co-occurrence']:
                node_list[c]['parents'] += [(node, 'cooc')]
                node_list[node]['childs'] += [(c, 'cooc')]
            for c in node_list[node]['support'].copy():
                if (c in node_list.keys()):
                    node_list[c]['parents'] += [(node, 'supp')]
                    node_list[node]['childs'] += [(c, 'supp')]
            for c in node_list[node]['surround'].copy():
                for key in c.keys():
                    node_list[key]['parents'] += [(node, 'surr')]
                    node_list[node]['childs'] += [(key, 'surr')]
                    node_list[c[key]]['parents'] += [(node, 'surr')]
                    node_list[node]['childs'] += [(c[key], 'surr')]

        # siblings (next-to relation)
        for node in node_list.keys():
            for parent_node, _ in node_list[node]['parents']:
                node_list[node]['siblings'] += node_list[parent_node]['childs']

        return node_list

    def __preprocess_root_wall_nodes__(self, node_list):
        """
        # simple preprocess for root and wall nodes
        :param node_list:
        :return:
        """

        node_list['root']['self_info'] = {'dim': [0.0, 0.0, 0.0], 'translation': [0.0, 0.0, 0.0],
                                          'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}
        x_min = node_list['wall_0']['self_info']['translation'][0]
        x_max = node_list['wall_2']['self_info']['translation'][0]
        y_min = node_list['wall_3']['self_info']['translation'][2]
        y_max = node_list['wall_1']['self_info']['translation'][2]
        x_mean = 0.5 * (x_min + x_max)
        y_mean = 0.5 * (y_min + y_max)
        for wall_node in ['wall_0', 'wall_1', 'wall_2', 'wall_3']:
            node_list[wall_node]['self_info']['translation'][0] -= x_mean
            node_list[wall_node]['self_info']['translation'][2] -= y_mean

        # for root and wall nodes, switch cooc to support relation
        # (we simply assume all wall nodes are 'supported' by root node, all other nodes are 'supported' by wall nodes)
        if ('support' not in node_list['root'].keys()):
            node_list['root']['surround'] = []
            node_list['root']['support'] = copy.deepcopy(node_list['root']['co-occurrence'])
            node_list['root']['co-occurrence'] = []
        for cur_node in ['wall_0', 'wall_1', 'wall_2', 'wall_3']:
            if (len(node_list[cur_node]['co-occurrence']) > 0):
                node_list[cur_node]['surround'] = []
                node_list[cur_node]['support'] = copy.deepcopy(node_list[cur_node]['co-occurrence'])
                node_list[cur_node]['co-occurrence'] = []

        return node_list

    def _training_pass(self, valid_rooms, epoch, is_training=True):
        """
        Single training pass
        :param valid_rooms: choice of =[self.valid_rooms_train, self.valid_rooms_test]
        :param epoch: current epoch
        :param is_training: train or test pass
        :return:
        """

        ''' epoch and args '''
        epoch += self.pretrained_epoch
        opt_parser =self.opt_parser

        ''' current training state '''
        if (is_training):
            self.STATE = 'TRAIN'
            self.full_enc.train()
            self.full_dec.train()
        else:
            self.STATE = 'EVAL'
            self.full_enc.eval()
            self.full_dec.eval()

        ''' init loss / accuracy '''
        loss_cat_per_epoch, acc_cat_per_epoch, loss_dim_per_epoch, num_node_per_epoch, dim_acc_per_epoch = 0.0, {1:0.0, 3:0.0, 5:0.0}, 0.0, 0.0, 0.0

        ''' shuffle room list and create training batches '''
        shuffle(valid_rooms)
        room_indices = list(range(len(valid_rooms)))
        room_idx_batches = [room_indices[i: i + opt_parser.batch_size] for i in
                            range(0, len(valid_rooms), opt_parser.batch_size)]

        ''' Batch loop '''
        for batch_i, batch in enumerate(room_idx_batches):

            batch_rooms = [valid_rooms[i] for i in batch]

            """ ==================================================================
                                        Encoder Part
            ================================================================== """
            # init torchfold
            enc_fold = torchfold.Fold()
            enc_fold_nodes = []
            enc_rand_path_order = []
            enc_rand_path_root_to_leaf_order = []

            # loop for rooms
            for room_i, room in enumerate(batch_rooms):

                node_list = self.__preprocess_root_wall_nodes__(room['node_list'])

                # adapt acceleration for large graphs (by splitting into sub-graphs)
                consider_path_type = ['root']
                root_to_split = False

                if(opt_parser.adapt_training_on_large_graph):
                    if (len(node_list.keys()) >= int(opt_parser.max_scene_nodes)):
                        consider_path_type = node_list['root']['support']
                        root_to_split = True

                # loop for sub-graphs
                for sub_tree_root_node in consider_path_type:

                    # find sub-graph's root to leaf node path
                    subtree_to_leaf_path = self.find_root_to_leaf_node_path(node_list, cur_node=sub_tree_root_node)

                    # skip unreasonable paths
                    subtree_to_leaf_path = [p for p in subtree_to_leaf_path if len(p) >= 2 and len(p) < opt_parser.max_scene_nodes]
                    subtree_to_leaf_path = [p for p in subtree_to_leaf_path if 'wall' not in p[-1].split('_')[0]]
                    if(len(subtree_to_leaf_path) == 0):
                        continue

                    # find node list for sub-graphs
                    sub_keys = list(set(self.find_selected_node_list(node_list, sub_tree_root_node)))
                    if(root_to_split):
                        sub_keys += ['root']
                    sub_node_list = dict((k, node_list[k]) for k in sub_keys if k in node_list.keys())

                    # update parents, childs, siblings for each node
                    sub_node_list = self.find_parent_sibling_child_list(sub_node_list)

                    # exclude examples with too many sub tree nodes
                    if(len(sub_node_list.keys()) >= int(opt_parser.max_scene_nodes)):
                        print('skip too large sub-scene:', len(sub_node_list.keys()), '>', opt_parser.max_scene_nodes)
                        continue

                    subtree_to_leaf_path.sort()
                    # loop for each root-to-leaf path
                    for rand_path_idx, rand_path in enumerate(subtree_to_leaf_path):

                        rand_path_fold, rand_path_node_name_order = self.model.encode_tree_fold(enc_fold, sub_node_list, rand_path, opt_parser)
                        enc_fold_nodes += rand_path_fold
                        enc_rand_path_order += [[room_i, sub_tree_root_node] + rand_path_node_name_order]
                        enc_rand_path_root_to_leaf_order += [rand_path]

            # if batch size is too small, sometimes there is no valid training instance.
            if(len(enc_fold_nodes) == 0):
                print('surprisingly this batch has no valid training trees!')
                continue


            # torch-fold train encoder
            enc_fold_nodes = enc_fold.apply(self.full_enc, [enc_fold_nodes])
            enc_fold_nodes = torch.split(enc_fold_nodes[0], 1, 0)

            """ ==================================================================
                                        Decoder Part
            ================================================================== """
            # Ground-truth leaf node vec
            leaf_node_gt = []

            # # FOLD
            dec_fold = torchfold.Fold()
            dec_fold_nodes = []

            # loop for all encoded vectors
            for i, rand_path_order in enumerate(enc_rand_path_order):

                # find room-node-list
                room_i = rand_path_order[0]
                node_list = batch_rooms[room_i]['node_list']

                # decode to k-vec and add Ops to fold
                dec_fold_nodes.append(self.model.decode_tree_fold(dec_fold, enc_fold_nodes[i], opt_parser))
                leaf_node_gt += [self.model.get_gt_k_vec(node_list, enc_rand_path_root_to_leaf_order[i][-1], opt_parser)]  # leaf node ground-truth k-vec

            # torch-fold decoder
            dec_fold_nodes = dec_fold.apply(self.full_dec, [dec_fold_nodes])
            leaf_node_pred = dec_fold_nodes[0]

            """ ==================================================================
                                      Loss / Accuray Part
            ================================================================== """
            size_pos_dim = 6

            leaf_node_cat_gt = [c[:-size_pos_dim].index(1) for c in leaf_node_gt]
            leaf_node_cat_gt = to_torch(leaf_node_cat_gt, torch_type=torch.LongTensor, dim_0=len(leaf_node_gt)).view(-1)

            leaf_node_dim_gt = [c[-size_pos_dim:-size_pos_dim+3] for c in leaf_node_gt]
            leaf_node_dim_gt = to_torch(leaf_node_dim_gt, torch_type=torch.FloatTensor, dim_0=len(leaf_node_gt))

            loss_cat = self.LOSS_CLS(leaf_node_pred[:, :-size_pos_dim], leaf_node_cat_gt)
            loss_dim = self.LOSS_L2(leaf_node_pred[:, -size_pos_dim:-size_pos_dim+3], leaf_node_dim_gt) * 1000

            # report scores
            loss_cat_per_batch = loss_cat.data.cpu().numpy()
            loss_dim_per_batch = loss_dim.data.cpu().numpy()
            num_node_per_batch = len(leaf_node_gt) * 1.0

            # accuracy (top k)
            acc_cat_per_batch = {}
            for k in [1, 3, 5]:
                _, pred = leaf_node_pred[:, :-size_pos_dim].topk(k, 1, True, True)
                pred = pred.t()
                correct = pred.eq(leaf_node_cat_gt.view(1, -1).expand_as(pred))
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                acc_cat_per_batch[k] = correct_k[0].cpu().numpy()

            # dimension (diagonal) percentage off
            diag_pred = np.sqrt(
                np.sum(leaf_node_pred[:, -size_pos_dim:-size_pos_dim+3].data.cpu().numpy() ** 2, axis=1))
            diag_gt = np.sqrt(
                np.sum(leaf_node_dim_gt.data.cpu().numpy() ** 2, axis=1))
            dim_acc_per_batch = np.sum(np.abs(diag_pred - diag_gt) / diag_gt)

            loss_cat_per_epoch += loss_cat_per_batch
            loss_dim_per_epoch += loss_dim_per_batch
            num_node_per_epoch += num_node_per_batch
            dim_acc_per_epoch += dim_acc_per_batch
            for key in acc_cat_per_epoch.keys():
                acc_cat_per_epoch[key] += acc_cat_per_batch[key]

            if (is_training):

                # Back-propagation
                for key in self.opt.keys():
                    self.opt[key].zero_grad()

                # only train object dimensions
                if(opt_parser.train_dim and not opt_parser.train_cat):
                    loss_dim.backward()
                # only train object categories
                elif(opt_parser.train_cat and not opt_parser.train_dim):
                    loss_cat.backward()
                # train both
                elif(opt_parser.train_cat and opt_parser.train_dim):
                    loss_cat.backward(retain_graph=True)
                    loss_dim.backward()
                else:
                    print('At least enable --train_cat or --train_dim.')
                    exit(-1)

                for key in self.opt.keys():
                    self.opt[key].step()

            if (opt_parser.verbose >= 0):
                print(self.STATE, opt_parser.name, epoch,
                      ': ({}/{}:{})'.format(batch_i, len(room_idx_batches), num_node_per_batch),
                      'CAT Loss: {:.4f}, Acc_1: {:.4f}, Acc_3: {:.4f}, Acc_5: {:.4f},Dim Loss: {:.8f}, dim acc: {:.2f}'.format(
                          loss_cat_per_batch / num_node_per_batch * 100.0,
                          acc_cat_per_batch[1] / num_node_per_batch,
                          acc_cat_per_batch[3] / num_node_per_batch,
                          acc_cat_per_batch[5] / num_node_per_batch,
                          loss_dim_per_batch / num_node_per_batch,
                          dim_acc_per_batch / num_node_per_batch))

        """ ==================================================================
                                  Report Part
        ================================================================== """

        print('========================================================')
        print(self.STATE, epoch, ': ',
              'CAT Loss: {:.4f}, Acc_1: {:.4f}, Acc_3: {:.4f}, Acc_5: {:.4f}, Dim Loss: {:.4f}, Dim acc: {:.4f}'.format(
                  loss_cat_per_epoch / num_node_per_epoch * 100.0,
                  acc_cat_per_epoch[1] / num_node_per_epoch,
                  acc_cat_per_epoch[3] / num_node_per_epoch,
                  acc_cat_per_epoch[5] / num_node_per_epoch,
                  loss_dim_per_epoch / num_node_per_epoch,
                  dim_acc_per_epoch / num_node_per_epoch))
        print('========================================================')

        ''' write avg to log '''
        if (opt_parser.write):
            self.writer.add_scalar('{}_LOSS_CAT'.format(self.STATE), loss_cat_per_epoch / num_node_per_epoch,
                              epoch)
            self.writer.add_scalar('{}_ACC_CAT'.format(self.STATE), acc_cat_per_epoch[1] / num_node_per_epoch,
                              epoch)
            self.writer.add_scalar('{}_ACC_3_CAT'.format(self.STATE), acc_cat_per_epoch[3] / num_node_per_epoch,
                              epoch)
            self.writer.add_scalar('{}_ACC_5_CAT'.format(self.STATE), acc_cat_per_epoch[5] / num_node_per_epoch,
                              epoch)
            self.writer.add_scalar('{}_LOSS_DIM'.format(self.STATE), loss_dim_per_epoch / num_node_per_epoch,
                              epoch)

        ''' save model '''
        if (not is_training):
            def save_model(save_type):
                torch.save({
                    'full_enc_state_dict': self.full_enc.state_dict(),
                    'full_dec_state_dict': self.full_dec.state_dict(),
                    'full_enc_opt': self.opt['full_enc'].state_dict(),
                    'full_dec_opt': self.opt['full_dec'].state_dict(),
                    'epoch': epoch
                }, '{}/Entire_model_{}.pth'.format(opt_parser.outf, save_type))

            # if model is better, save model checkpoint
            # min dim loss model
            if(loss_dim_per_epoch / num_node_per_epoch < self.MIN_DIM_LOSS):
                self.MIN_DIM_LOSS = loss_dim_per_epoch / num_node_per_epoch
                save_model('min_dim_loss')
            # max cat acc model (top-5 acc)
            if (acc_cat_per_epoch[5] / num_node_per_epoch > self.MAX_ACC):
                self.MAX_ACC = acc_cat_per_epoch[5] / num_node_per_epoch
                save_model('max_acc')
            # min cat loss model
            if (loss_cat_per_epoch / num_node_per_epoch < self.MIN_LOSS):
                self.MIN_LOSS = loss_cat_per_epoch / num_node_per_epoch
                save_model('min_loss')
            # always save the latest model
            save_model('last_epoch')

        return

    def train(self, epoch):
        st = time.time()
        self._training_pass(self.valid_rooms_train, epoch, is_training=True)
        print('time usage:', time.time() - st)

    def test(self, epoch, DEBUG_mode=False):
        with torch.no_grad():
            self._training_pass(self.valid_rooms_test, epoch, is_training=False)



