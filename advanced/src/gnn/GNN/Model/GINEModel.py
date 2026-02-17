import torch
from torch_geometric.nn import GINEConv, dense_mincut_pool, DMoNPooling, MLP, dense_diff_pool, DenseGINConv, BatchNorm, GraphNorm, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_adj, to_dense_batch



class GINE_Net(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim,           
                 edge_dim,          
                 num_layers_pre=2,      
                 num_layers_post=2, 
                 activation='RELU',  
                 average_nodes=None,    # Needed for dense pooling methods
                 pooling = None,          # Pooling method
                 pool_ratio=0.1,
                 model_mode = '1' 
    ):
        super(GINE_Net, self).__init__()


        self.num_layers_pre = num_layers_pre
        self.num_layers_post = num_layers_post
        self.hidden_channels = hidden_dim
        self.act = activation_resolver(activation)
        self.pooling = pooling
        self.pool_ratio = pool_ratio
        self.model_mode = model_mode
        # self.batch_size = 64

        self.embeding = torch.nn.Linear(input_dim, hidden_dim)

        self.conv_layers_pre = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            mlp = MLP([hidden_dim, 4*hidden_dim, hidden_dim], act=activation)
            self.conv_layers_pre.append(GINEConv(nn=mlp, train_eps=True, edge_dim = edge_dim))
            self.ffn_layers.append(MLP([hidden_dim, 4*hidden_dim, hidden_dim], act=activation))

        self.graphnorm = LayerNorm(hidden_dim)

        self.mlp_clf = MLP([hidden_dim, hidden_dim, hidden_dim//2, 2], 
                      act=activation,
                      norm=None,
                      dropout=0.5)
        

        ## 第二种架构下，后续需要添加node_assignment作为feature
        if self.model_mode == '2': 
            hidden_dim = hidden_dim + 1
            input_dim = hidden_dim
        
        pooled_nodes = int(pool_ratio * average_nodes)
        if pooling == 'mincut' or 'diffpool':
            self.pool = torch.nn.Linear(hidden_dim, pooled_nodes)
        elif pooling == 'dmon':
            self.pool = DMoNPooling(hidden_dim, pooled_nodes)
        else:
            self.pool = pooling

        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = MLP([hidden_dim, hidden_dim, hidden_dim], act=activation,norm=None)
            self.conv_layers_post.append(GINEConv(nn=mlp, train_eps=True, edge_dim = edge_dim))
            # self.conv_layers_post.append(DenseGINConv(nn=mlp, train_eps=False))

        
        self.mlp_reg = MLP([hidden_dim, hidden_dim, hidden_dim//2, 1], 
                      act=activation,
                      norm=None,
                      dropout=0.5)
        
    def forward(self, data, inference = False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edges_attr, data.batch
        # print(x.shape)
        x = self.embeding(x)
        for i,conv in enumerate(self.conv_layers_pre):
            output = self.graphnorm(x)
            output = conv(output, edge_index, edge_attr)
            x = x+output
            output = self.graphnorm(x)
            output = self.ffn_layers[i](output)
            x = x+output

        if self.model_mode == '1':
            x = self.mlp_clf(x)
            return x
        
        elif self.model_mode == '2':
            out_clf = self.mlp_clf(x)
            
            if inference is True:
                node_label = out_clf.argmax(dim=1)
            else:
                node_label = data.y

            x = torch.cat([x, node_label.unsqueeze(1)], dim=1)
            # x, mask = to_dense_batch(x, batch)
            # edge_index = to_dense_adj(edge_index, batch)

            # if self.pooling=='diffpool':
            #     s = self.pool(x)
            #     x, edge_index, l1, l2 = dense_diff_pool(x, edge_index, s, mask)
            #     aux_loss = 0.1*l1 + 0.1*l2
            # elif self.pooling=='mincut':
            #     s = self.pool(x)
            #     x, edge_index, l1, l2 = dense_mincut_pool(x, edge_index, s, mask)
            #     aux_loss = 0.5*l1 + l2
            # elif self.pooling=='dmon':  
            #     _, x, edge_index, l1, l2, l3 = self.pool(x, edge_index, mask)
            #     aux_loss = 0.3*l1 + 0.3*l2 + 0.3*l3

            for conv in self.conv_layers_post: 
                x = conv(x, edge_index, edge_attr)
                x = self.act(x)
            aux_loss = torch.tensor([0],device=x.device)
            # print(x.shape)
            x = x.view(data.iter.shape[0],-1,x.shape[1])
            # print(x.shape)
            x = torch.sum(x, dim=1)
            # print(x.shape)
            out_reg = self.mlp_reg(x)
            # print(out_reg.shape)
            return out_clf, out_reg, aux_loss
        else:
            raise ValueError('model_mode should be 1 or 2')

       




