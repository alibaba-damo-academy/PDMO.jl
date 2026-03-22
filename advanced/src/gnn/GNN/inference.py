import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNN.Model import GINE_Net, GCN_Net, GAT_Net
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.parallel import DataParallel
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, CosineAnnealingWarmRestarts

import os
import time
import json


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def get_statistic_features(t):

    arr = t.flatten()
    min_ele = 1.0e30
    max_ele = -1.0e30
    nnz = 0 
    sum_ele = 0.0
    prod_ele = 1.0
    
    for ele in arr: 
        if ele < min_ele: 
            min_ele = ele 
        
        if ele > max_ele: 
            max_ele = ele 
        
        if np.abs(ele) > 1e-6: 
            nnz += 1 

        sum_ele += np.log(np.abs(ele) + 1.0)
        prod_ele *= np.log(np.abs(ele) + 1.0)
        
    return np.array([
        np.mean(arr),
        np.std(arr),
        np.var(arr),
        np.mean(np.abs(arr)),
        np.median(arr),
        np.percentile(arr, 25),
        np.percentile(arr, 75),
        min_ele, 
        max_ele, 
        nnz, 
        sum_ele, 
        prod_ele,
    ])

def get_structure_features(t):
    matrix = t
    if matrix.shape[0] == matrix.shape[1]:
        return np.array([
            np.linalg.matrix_rank(matrix),
            np.trace(matrix),
            #np.linalg.cond(matrix),
            # np.linalg.det(matrix),
            # np.linalg.slogdet(matrix)[0],
            # np.linalg.slogdet(matrix)[1],   
            # np.linalg.eigvals(matrix)[0].min(),
            # np.linalg.eigvals(matrix)[0].max(),
        ])
    else:
        return np.array([
            np.linalg.matrix_rank(matrix),
            np.trace(matrix),
            #np.linalg.cond(matrix),
            # np.linalg.det(matrix),
            # np.linalg.slogdet(matrix)[0],
            # np.linalg.slogdet(matrix)[1],   
            # np.linalg.eigvals(matrix)[0].min(),
            # np.linalg.eigvals(matrix)[0].max(),
        ])

def get_geometric_features(t):
    matrix = t
    return np.array([
        np.linalg.norm(matrix, ord=1),
        np.linalg.norm(matrix, ord=2),
        np.linalg.norm(matrix, ord=np.inf),
    ])

def get_random_features(i):
    return np.array([i,np.random.rand()])


import json

def _parse_node_id(node_key):
    if "(" in node_key and node_key.endswith(")"):
        return int(node_key[node_key.find("(") + 1:-1])
    return int(node_key)

def get_feature_from_julia_data(data):
    num_nodes = len(data['nodes'])
    num_edges = len(data['edges'])
    nodes = data['nodes']
    edges = data['edges']
    node_order = data.get('node_order', sorted(nodes.keys()))
    edge_order = data.get('edge_order', sorted(edges.keys()))
    
    x = []
    edge_index = []
    edges_attr = []
    # edge_label = []
    # node_label = []

    pos = {}
    for i, key in enumerate(node_order):
        node = nodes[key]
        node_attr = [1 for j in range(20)]
        pos[key] = i
        # attr = []
        # Node_type as the indicator feature
        # if node['type'] == 'VARIABLE_NODE':
        #     node_attr.extend([0,1])
        #     # The first feature of the node is its dimension.
        #     node_attr.append(np.array(node['f']['q']).shape[0]) 

        #     # Filtered information about the matrix Q and q append to features 
        #     node_attr.extend(node['f']['Q'][0])
        #     node_attr.extend(node['f']['q'])
        #     node_attr.append(node['f']['r']) 

        # else:
        #     node_attr.extend([1,0])
        #     node_attr.append(np.array(node['g']['rhs']).shape[0])
        #     node_attr.extend([0,0,0,0,0,0,0,0,0,0,0,0])
        #     node_attr.extend([0,0])
        #     node_attr.extend([0,0,0])
        #     node_attr.extend([0,0,0,0,0,0,0,0,0,0,0,0])
        #     node_attr.extend([0,0,0])
        #     node_attr.append(0)
            #print("Is not bipartite itself")
            #print(node)   

        '''
        # The next feature is its domain size(measure) (optional).
        measure = 1.0
        for j in range(len(node['domian']['ub'])):
            measure *= node['domain']['ub'][j] - node['domain']['lb'][j]
        node_attr.append(measure)  

        # The next feature is its domain type (optional). 
        node_attr.append(node['domain']['type]) 
        '''

        # The next feature is the degree of the node.
        # node_attr.append(len(node['neighboring_edges']))

        # Random features to enhancement the expressive power of the model.
        # node_attr.extend(get_random_features(i))

        x.append(node_attr)
        # node_label.append(node['assigment'])


    for key in edge_order:
        edge = edges[key]
        edge_attr = []

        # # The first feature of the edge is its dimension.
        # edge_attr.append(1)

        # # The next feature of the edge is the minimum and maximum condition number of the two matrices.
        # cond_i = torch.linalg.cond(torch.tensor(edge['Q_i']))
        # cond_j = torch.linalg.cond(torch.tensor(edge['Q_j']))
        # edge_attr.append(min([cond_i, cond_j]))
        # edge_attr.append(max([cond_i, cond_j]))
        Q_i = []
        Q_j = []
        rhs = []
        
        constraint = edge['constraints']
        key_i = _parse_node_id(edge['node_i'])
        key_j = _parse_node_id(edge['node_j'])
        Q_i, Q_j = constraint[int(key_i)], constraint[int(key_j)]
        # print(constraint)
        # Filtered information about matrix Q and b append to features 
        # edge_attr.extend(Q_i)
        edge_attr.append(len(Q_i))
        edge_attr.extend(get_structure_features(np.array(Q_i)))
        edge_attr.extend(get_geometric_features(np.array(Q_i)))
        edge_attr.append(np.linalg.cond(np.array(Q_i)))
    

        # edge_attr.extend(Q_j)
        edge_attr.extend(get_structure_features(np.array(Q_j)))
        edge_attr.extend(get_geometric_features(np.array(Q_j)))
        edge_attr.append(np.linalg.cond(np.array(Q_j)))

        # edge_attr.extend(rhs)
        # edge_attr.extend(get_geometric_features(np.array(rhs)))


        if edge["type"] == "TWO_BLOCK_EDGE": 
            edge_attr.extend([1,0])
        else:
            edge_attr.extend([0,1])
            #print("Is not bipartite")
        # print(edge_attr)
        edges_attr.append(edge_attr)
        edges_attr.append(list(map(lambda x: -x, edge_attr)))
        # print(edge.keys())
        node_index_i = edge['node_i']
        node_index_j = edge['node_j']
        # print(node_index_i)
        edge_index.append([pos[node_index_i], pos[node_index_j]])
        edge_index.append([pos[node_index_j], pos[node_index_i]])
        
        # edge_split = []
        # #if 'split' in edge.keys():
        # #    edge_split.append(edge['split'])
        # #    if edge['split'] == 1:
        # #        edge_split.append(edge['assignment'])
        # #    else:
        # #        edge_split.append(0)
        # # if 'should_split' in edge.keys():
        # edge_split.append(edge['should_split'])
        # if edge['should_split'] == 1:
        #     edge_split.append(edge['assignment'])
        # else:
        #     edge_split.append(0)
        # edge_label.append(edge_split)
    # print(len(edge_index),len(edges_attr[3]))
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edges_attr = torch.tensor(edges_attr, dtype=torch.float)
    # node_label = torch.tensor(node_label, dtype=torch.float)
    # edge_label = torch.tensor(edge_label, dtype=torch.int)
    return x, edge_index, edges_attr #, node_label, edge_label, data['iter']

def read_graph_from_json(dir):
    with open(dir) as f:
        data = json.load(f)

    num_nodes = len(data['nodes'])
    num_edges = len(data['edges'])
    nodes = data['nodes']
    edges = data['edges']
    
    x = []
    edge_index = []
    edges_attr = []
    # edge_label = []
    # node_label = []

    pos = {}
    for i in range(num_nodes):
        key = 'VariableNode('+str(i+1)+')'
        node = nodes[key]
        node_attr = [1 for j in range(20)]
        pos[key] = i
        # attr = []
        # Node_type as the indicator feature
        # if node['type'] == 'VARIABLE_NODE':
        #     node_attr.extend([0,1])
        #     # The first feature of the node is its dimension.
        #     node_attr.append(np.array(node['f']['q']).shape[0]) 

        #     # Filtered information about the matrix Q and q append to features 
        #     node_attr.extend(node['f']['Q'][0])
        #     node_attr.extend(node['f']['q'])
        #     node_attr.append(node['f']['r']) 

        # else:
        #     node_attr.extend([1,0])
        #     node_attr.append(np.array(node['g']['rhs']).shape[0])
        #     node_attr.extend([0,0,0,0,0,0,0,0,0,0,0,0])
        #     node_attr.extend([0,0])
        #     node_attr.extend([0,0,0])
        #     node_attr.extend([0,0,0,0,0,0,0,0,0,0,0,0])
        #     node_attr.extend([0,0,0])
        #     node_attr.append(0)
            #print("Is not bipartite itself")
            #print(node)   

        '''
        # The next feature is its domain size(measure) (optional).
        measure = 1.0
        for j in range(len(node['domian']['ub'])):
            measure *= node['domain']['ub'][j] - node['domain']['lb'][j]
        node_attr.append(measure)  

        # The next feature is its domain type (optional). 
        node_attr.append(node['domain']['type]) 
        '''

        # The next feature is the degree of the node.
        # node_attr.append(len(node['neighboring_edges']))

        # Random features to enhancement the expressive power of the model.
        # node_attr.extend(get_random_features(i))

        x.append(node_attr)
        # node_label.append(node['assigment'])


    for i in range(num_edges):
        key = 'TwoBlockEdge('+str(i+1)+')'
        edge = edges[key]
        edge_attr = []

        # # The first feature of the edge is its dimension.
        # edge_attr.append(1)

        # # The next feature of the edge is the minimum and maximum condition number of the two matrices.
        # cond_i = torch.linalg.cond(torch.tensor(edge['Q_i']))
        # cond_j = torch.linalg.cond(torch.tensor(edge['Q_j']))
        # edge_attr.append(min([cond_i, cond_j]))
        # edge_attr.append(max([cond_i, cond_j]))
        Q_i = []
        Q_j = []
        rhs = []
        
        constraint = edge['constraints']
        if len(edge['node_i'])==15:
            key_i = edge['node_i'][13]
        else:
            key_i = edge['node_i'][13:15]
        if len(edge['node_j'])==15:
            key_j = edge['node_j'][13]
        else:
            key_j = edge['node_j'][13:15]
        
        # Q_i.append(constraint[key_i][0][0])
        # Q_j.append(constraint[key_j][0][0]) 
        # rhs.append(edge['rhs'][0])
        # print(key_i,key_j)
        Q_i,Q_j = constraint[key_i],constraint[key_j]
        # print(constraint)
        # Filtered information about matrix Q and b append to features 
        # edge_attr.extend(Q_i)
        edge_attr.append(len(Q_i))
        edge_attr.extend(get_structure_features(np.array(Q_i)))
        edge_attr.extend(get_geometric_features(np.array(Q_i)))
        edge_attr.append(np.linalg.cond(np.array(Q_i)))
    

        # edge_attr.extend(Q_j)
        edge_attr.extend(get_structure_features(np.array(Q_j)))
        edge_attr.extend(get_geometric_features(np.array(Q_j)))
        edge_attr.append(np.linalg.cond(np.array(Q_j)))

        # edge_attr.extend(rhs)
        # edge_attr.extend(get_geometric_features(np.array(rhs)))


        if edge["type"] == "TWO_BLOCK_EDGE": 
            edge_attr.extend([1,0])
        else:
            edge_attr.extend([0,1])
            #print("Is not bipartite")
        # print(edge_attr)
        edges_attr.append(edge_attr)
        edges_attr.append(list(map(lambda x: -x, edge_attr)))
        # print(edge.keys())
        node_index_i = edge['node_i']
        node_index_j = edge['node_j']
        # print(node_index_i)
        edge_index.append([pos[node_index_i], pos[node_index_j]])
        edge_index.append([pos[node_index_j], pos[node_index_i]])
        
        # edge_split = []
        # #if 'split' in edge.keys():
        # #    edge_split.append(edge['split'])
        # #    if edge['split'] == 1:
        # #        edge_split.append(edge['assignment'])
        # #    else:
        # #        edge_split.append(0)
        # # if 'should_split' in edge.keys():
        # edge_split.append(edge['should_split'])
        # if edge['should_split'] == 1:
        #     edge_split.append(edge['assignment'])
        # else:
        #     edge_split.append(0)
        # edge_label.append(edge_split)
    # print(len(edge_index),len(edges_attr[3]))
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edges_attr = torch.tensor(edges_attr, dtype=torch.float)
    # node_label = torch.tensor(node_label, dtype=torch.float)
    # edge_label = torch.tensor(edge_label, dtype=torch.int)
    return x, edge_index, edges_attr #, node_label, edge_label, data['iter']


class Inference: 
    def __init__(self, 
                model, 
                input_dim,
                hidden_dim,           
                edge_dim, 
                param_path,
                device,
                model_mode = '1',
                activation = 'RELU',
                num_gnn_layers=3,          
                average_nodes=None,    # Needed for dense pooling methods
                pooling = None,          # Pooling method
                pool_ratio=0.1,
                ):
        
        self.model_mode = model_mode
        self.config = {
            'model': model,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'edge_dim': edge_dim,
            'num_layers_pre': num_gnn_layers,
            'num_layers_post': 1,
            'activation': activation,
            'average_nodes': average_nodes,
            'pooling': pooling,
            'pool_ratio': pool_ratio,
            'model_mode': model_mode,
        }
        self.device = device

        if model == 'GAT':
            self.model = GAT_Net(input_dim,hidden_dim,edge_dim,num_layers_pre=num_gnn_layers,num_layers_post=1,
                                 activation = activation,average_nodes=average_nodes,pooling = pooling,pool_ratio=pool_ratio,
                                 model_mode = self.model_mode
                                ).to(self.device)
        elif model == 'GCN':
            self.model = GCN_Net(input_dim,hidden_dim,edge_dim,num_layers_pre=num_gnn_layers,num_layers_post=1,
                                 activation = activation,average_nodes=average_nodes,pooling = pooling,pool_ratio=pool_ratio,
                                 model_mode = self.model_mode
                                ).to(self.device)
        elif model == 'GINE':
            self.model = GINE_Net(input_dim,hidden_dim,edge_dim,num_layers_pre=num_gnn_layers,num_layers_post=1,
                                 activation = activation,average_nodes=average_nodes,pooling = pooling,pool_ratio=pool_ratio,
                                 model_mode = self.model_mode
                                ).to(self.device)
        else:
            raise ValueError("Invalid model name, we only support GAT, GCN, GINE")
        load_device = self.device if torch.cuda.is_available() else torch.device('cpu')
        self.model.load_state_dict(
            torch.load(param_path, map_location=load_device, weights_only=True)
        )


    def inference(self,data):
        self.model.eval()

        with torch.no_grad():
            data = data.to(self.device)
            if self.model_mode == '1':
                output = self.model(data)
            elif self.model_mode == '2':
                out_clf, out_reg, aux_loss= self.model(data, inference=True)
                out_reg = out_reg.squeeze()
            pred = F.softmax(output,dim=1)
                # print(pred)
            pred = torch.argmax(pred,dim=1)
                        # print(loss_clf.item(),loss_reg.item())
        return pred


if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    dataset = []

    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.abspath(os.path.join(base_dir, "..", "graph_json", "mbp.json"))
    model_path = os.path.abspath(os.path.join(base_dir, "model_weights_10.pth"))
    output_path = os.path.abspath(os.path.join(base_dir, "..", "graph_json", "output.json"))

    x, edge_index, edges_attr = read_graph_from_json(graph_path)
    data = Data(x=x, edge_index=edge_index.T, edges_attr=edges_attr)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    input_dim_1 = data.x.shape[1]
    hidden_dim = 128
    edge_dim_1 = data.edges_attr.shape[1]

    output = Inference(
        'GINE', input_dim_1, hidden_dim, edge_dim_1,
        model_path,
        device,
        model_mode='1', activation='ELU', num_gnn_layers=40,
        average_nodes=10, pooling='mincut', pool_ratio=0.1,
    ).inference(data)
    output = output.cpu().numpy()
    with open(output_path, 'w') as f:
        json.dump(output.tolist(), f)