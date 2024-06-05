import torch
import torch.nn as nn
import torch.optim as optim

class DCN(nn.Module):
    def __init__(self,num_features,cross_layers,dnn_layers):
        super(DCN,self).__init__()
        self.num_features = num_features
        self.cross_layers = cross_layers

        # 交叉层
        self.cross_network = nn.ModuleList(  # nn.ModuleList用于罗列相同类型的网络，每一层都是Linear
            [nn.Linear(num_features,num_features) for _ in range(cross_layers)]
        )
        # DNN层
        dnn_layers = [nn.Linear(num_features,dnn_layers[0])] + [nn.Linear(dnn_layers[i],dnn_layers[i+1]) for i in range(len(dnn_layers) - 1)]
        self.dnn = nn.Sequential(*dnn_layers,nn.ReLU())  # *dnn_layers 解包dnn_layers，多个层作为单独的参数传入Sequential函数，每一层后面接一个ReLU

        # 输出层
        self.output = nn.Linear(dnn_layers[-1].out_features + num_features,1)  # 交叉层和DNN层组合相加

    def forward(self,x):
        x0 = x

        # cross
        for layer in self.cross_network:
            x = layer(x0*x) + x  # 交叉
        cross_output  = x
        # dnn
        dnn_output = self.dnn(x0)
        # 组合
        combined = torch.cat([cross_output,dnn_output],dim=1)
        output = self.output(combined)
        return output

if __name__ == "__main__":
    num_features = 10
    cross_layers = 3
    dnn_layers = [64,32,16]

    model = DCN(num_features,cross_layers,dnn_layers)

    x = torch.randn(5,num_features)
    y = torch.randn(5,1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # train
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
