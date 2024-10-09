import data_process
from data_process import MyDataSet
import numpy as np
from baseline import SVR_baseline,evaluate,HA_baseline
# from MyLSTM import  MyLSTM
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from GCN_LSTM import MyLSTM_GCN

adj, flow = data_process.load_data()
flow =np.asmatrix(flow,dtype=np.float32)


adj_norm = data_process.cal_adj_norm(adj)
# print(adj_norm.shape)

x_train, y_train, x_test, y_test = data_process.train_test_spilt(flow, 4, 20, 0.3)
y_pred = HA_baseline(x_test, y_test)
evaluate(y_test, y_pred)

# print(x_train.shape)
# dataset
BATCH_SIZE = 32
train_dataset = MyDataSet(x_train, y_train, type = 'train')
test_dataset = MyDataSet(x_test, y_test, type = 'test')
train_loader = Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE,shuffle=False)


# LSTMmodel
input_size = x_train.shape[2]
# print(x_train.shape)
hidden_size = [x_train.shape[2], x_train.shape[2], x_train.shape[2]]
# print(x_train.shape)
seq_len = 20
pre_len = 4
num_of_layers = 3
# model = MyLSTM(input_size,hidden_size,num_of_layers,seq_len,pre_len,BATCH_SIZE)
adj_norm = torch.tensor(adj_norm, dtype=torch.float32)
model  = MyLSTM_GCN(input_size, hidden_size, num_of_layers, seq_len, pre_len, BATCH_SIZE, adj_norm)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
num_epochs = 1000
train_loss = 0
# print(adj_norm)
for i in range(num_epochs):
    train_loader = tqdm(train_loader)
    train_loss = 0
    model.train() 
    # scheduler.step()
    #     # lr = scheduler.get_lr()
    for j, (X, Y) in enumerate(train_loader): 
        # X = X.view(-1,784)
        X = Variable(X)
        # X = Variable(X)
        X = X.float()
        Y = Variable(Y)
        Y = Y.float()
        # label = Variable(label)
        out,_ = model(X)
        # out = out[0][:,seq_len-pre_len:seq_len,:]
        lossvalue = loss(out, Y)
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step() 
        train_loss += float(lossvalue)
    print("train epoch:" + ' ' + str(i))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))

    model.eval()
    test_loss = 0
    y_pred = None
    for i,(X, Y) in enumerate(test_loader):
            X = Variable(X)
            X = X.float()
            Y = Variable(Y)
            Y = Y.float()
            with torch.no_grad():
                testout,_ = model(X)
            lossvalue = loss(testout,Y)
            if y_pred is None:
                y_pred = torch.cat([testout])
            else:
                y_pred = torch.cat([y_pred, testout])
            test_loss += float(lossvalue)
    y_pred = y_pred.numpy()
    evaluate(y_test,y_pred)
    print("lose:" + ' ' + str(test_loss / len(test_loader)))

# model.eval()
# test_loss = 0
# y_pred = None
# for i,(X, Y) in enumerate(test_loader):
#         X = Variable(X)
#         X = X.float()
#         Y = Variable(Y)
#         Y = Y.float()
#         with torch.no_grad():
#             testout,_ = model(X)
#         lossvalue = loss(testout,Y)
#         if y_pred is None:
#             y_pred = torch.cat([testout])
#         else:
#             y_pred = torch.cat([y_pred, testout])
#         test_loss += float(lossvalue)
# y_pred = y_pred.numpy()
# evaluate(y_test,y_pred)
# # r2 = r2_score(y_test,y_pred)
# print("lose:" + ' ' + str(test_loss / len(test_loader)))
# print(r2)

# y_true_flat = y_test.reshape(-1, y_test.shape[-1])
# y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

# r2 = r2_score(y_true_flat, y_pred_flat)
# print(f"R² Score: {r2:.4f}")
