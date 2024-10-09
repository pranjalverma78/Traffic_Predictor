import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wix = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Wih = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wcx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Wch = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wfx = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Wfh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wox = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.Woh = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.sigmoid_f = nn.Sigmoid()
        self.sigmoid_i = nn.Sigmoid()
        self.sigmoid_o = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

    def forward(self, input):
        (x, h, c) = input
        it = self.sigmoid_i(self.Wix(x) + self.Wih(h))
        ft = self.sigmoid_f(self.Wfx(x) + self.Wfh(h))
        ot = self.sigmoid_o(self.Wox(x) + self.Woh(h))
        ct = ft * c + it * self.tanh_1(self.Wcx(x) + self.Wch(h))
        ht = ot * self.tanh_2(ct)
        return ct, ht

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, activation='relu'):
        super(GraphConvolution, self).__init__()
        self.bias = None
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def forward(self, x, adj):
        # print(adj.shape)
        # print(x.shape)
        out = torch.mm(adj, x)#(num_nodes, batch_size)
        # print(out.shape)
        out = torch.mm(out, self.weight)#(num_nodes, output_dim)
        # out = out.T
        if self.bias:
            out += self.bias#(num_nodes, output_dim)
        
        # print(out.shape)
        return self.act(out)#(num_nodes, output_dim)
    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, adj):
        super(GCN, self).__init__()
        self.adj = adj
        self.layer1 = GraphConvolution(input_dim, output_dim)
        self.layer2 = GraphConvolution(output_dim, output_dim, activation='sigmoid')

    def forward(self, x):
        out = self.layer1(x, self.adj)
        out = self.layer2(out, self.adj)
        # print("OK")
        return out

class MyLSTM_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, input_len, output_len, batch_size, adj):
        super(MyLSTM_GCN, self).__init__()
        self.input_size = [input_size] + hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.adj = adj

        self.all_layers = nn.ModuleList()
        self.gcns = nn.ModuleList()
        for i in range(num_layers):
            self.all_layers.append(LSTMCell(self.input_size[i], self.hidden_size[i]))
            # self.gcns.append(GCN(self.hidden_size[i], self.hidden_size[i], self.adj))
            self.gcns.append(GCN(1, 1, self.adj))
        
        self.hidden2out = nn.Linear(self.input_len, self.output_len)
        self.relu = nn.ReLU()

    def forward(self, input):
        batch_size = input.shape[0]
        input_len = input.shape[1]
        feature_num = input.shape[2]
        # print(batch_size)
        state = []
        output = torch.zeros((batch_size, input_len, feature_num)).float()
        h_n = []
        c_n = []

        for i in range(input_len):
            x = input[:, i, :]
            for j in range(self.num_layers):
                if i == 0:
                    h, c = Variable(torch.zeros((batch_size, self.hidden_size[j]))), Variable(torch.zeros((batch_size, self.hidden_size[j])))
                    state.append((c, h))
                else:
                    new_c = []
                    new_h = []
                    for z in range(batch_size):
                        (c, h) = state[j]
                        cy = c[z]
                        hy = h[z]
                        cy = cy.reshape(156,1)
                        hy = hy.reshape(156,1)
                        # print("YE")
                        # print(cy.shape)
                        # print("YES")
                        ci = self.gcns[j](cy)
                        hi = self.gcns[j](hy)
                        ci = ci.T
                        hi = hi.T
                        new_c.append(ci)
                        new_h.append(hi)
                    
                    
                    new_c = torch.cat(new_c, dim=0)
                    new_h = torch.cat(new_h, dim=0)
                    # new_c = torch.tensor(new_c, dtype=torch.float32)
                    # new_h = torch.tensor(new_h, dtype=torch.float32)
                    state[j] = (new_c, new_h)

                
                (c, h) = state[j]
                # c = c.permute(1, 0)  # new_Shape: (num_nodes, batch_size)
                # h = h.permute(1, 0)

                # new_c = self.gcns[j](c)
                # new_h = self.gcns[j](h)

                # c = new_c.permute(1, 0)  # new_Shape: (batch_size, num_nodes)
                # h = new_h.permute(1, 0)

                # print(c.shape)
                new_c, new_h = self.all_layers[j]((x, h, c))
                # print("SOKE")
                
                # Apply the corresponding GCN to h and c, not x
                state[j] = (new_c, new_h)
                
                if i == self.input_len - 1:
                    h_n.append(new_h)
                    c_n.append(new_c)
            
            output[:, i, :] = new_h

        output = output.view(batch_size, -1, self.input_len)
        output = self.hidden2out(output)
        output = output.view(batch_size, self.output_len, -1)
        return output, (h_n, c_n)
