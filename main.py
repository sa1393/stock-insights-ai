import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime




class GRU(nn.Module) :
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) :
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.gru = nn.GRU(input_size=input_size,hidden_size=hidden_size,
                         num_layers=num_layers,batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x) :
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn) = self.gru(x, (h_0)) 
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out
    



    
def load_data(name, start_date, end_date):
    import yfinance as yf
    if start_date == '' and end_date == '':
        return yf.download(name) # 005930 : 삼성전자 주가
    else :
        return yf.download(name,
                     start=start_date,
                     end=end_date) # 005930 : 삼성전자 주가
    

    


def preprocess_data(df):
    df.head()
    fig = df['Close'].plot()

    X = df.drop('Close', axis=1) # X, y 분리
    y = df[['Close']]

    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    ms = MinMaxScaler() # 0 ~ 1
    ss = StandardScaler() # 평균 0, 분산 1

    X_ss = ss.fit_transform(X)
    y_ms = ms.fit_transform(y)

    X_train = X_ss[:79, :]
    X_test = X_ss[79:, :]

    y_train = y_ms[:79, :]
    y_test = y_ms[79:, :]

    print('Training Shape :', X_train.shape, y_train.shape)
    print('Testing Shape :', X_test.shape, y_test.shape)


    # 데이터셋 형태 및 크기 조정
    X_train_tensors = torch.Tensor(X_train)
    X_test_tensors = torch.Tensor(X_test)

    y_train_tensors = torch.Tensor(y_train)
    y_test_tensors = torch.Tensor(y_test)

    X_train_tensors_f = torch.reshape(X_train_tensors, 
                                    (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

    X_test_tensors_f = torch.reshape(X_test_tensors,
                                    (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    print('Training Shape :', X_train.shape, y_train.shape)
    print('Testing Shape :', X_test.shape, y_test.shape)

    return X_train_tensors_f, X_test_tensors_f, y_train_tensors, ss, ms, X, y




def train_model(X_train_tensors_f, y_train_tensors, ss, ms, X, y):
    num_epochs = 1000
    learning_rate = 0.0001

    input_size=5
    hidden_size=2
    num_layers=1

    num_classes=1
    model=GRU(num_classes,input_size,hidden_size,num_layers,X_train_tensors_f.shape[1])

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs) :
        outputs = model.forward(X_train_tensors_f)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensors)
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0 :
            print(f'Epoch : {epoch}, loss : {loss.item():1.5f}')

    df_x_ss = ss.transform(X)
    df_y_ms = ms.transform(y)

    df_x_ss = torch.Tensor(df_x_ss)
    df_y_ms = torch.Tensor(df_y_ms)
    df_x_ss = torch.reshape(df_x_ss, (df_x_ss.shape[0], 1, df_x_ss.shape[1]))

    return model, df_x_ss, df_y_ms, ms





def predict(model, df_x_ss, df_y_ms, ms):
    train_predict = model(df_x_ss)

    plt.style.use('seaborn-whitegrid')

    predicted = train_predict.data.numpy()

    label_y = df_y_ms.data.numpy()

    predicted = ms.inverse_transform(predicted)
    label_y = ms.inverse_transform(label_y)

    return predicted





def draw_graph(predicted, df):
    
    plt.figure(figsize=(10, 6))
    plt.axvline(x=datetime(2023,10,1), c='r', linestyle='--')

    df['pred'] = predicted
    plt.plot(df['Close'], label='Actual Data')
    plt.plot(df['pred'], label='Predicted Data')

    plt.title('Time-series Prediction')
    plt.legend()
    plt.show()





def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(125)

    if torch.cuda.is_available() :
        torch.cuda.manual_seed_all(125)
    df = load_data("005930.KS", "2023-10-01", "2023-10-31")
    # df = load_data("005930.KS", '', '')

    X_train_tensors_f, X_test_tensors_f, y_train_tensors, ss, ms, X, y = preprocess_data(df)
    model, df_x_ss, df_y_ms, ms = train_model(X_train_tensors_f, y_train_tensors, ss, ms, X, y)

    predicted = predict(model, df_x_ss, df_y_ms, ms)
    draw_graph(predicted, df)




if __name__ == '__main__':
    main()