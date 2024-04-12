import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import math
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch
print(torch.__version__)

 
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import whisper
import pickle
import torch.nn.functional as F
import argparse
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


 

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
 
  
def create_mask(seq_lengths, max_len):
    mask = torch.arange(max_len).expand(len(seq_lengths), max_len).to(seq_lengths.device)
    mask = mask >= seq_lengths.unsqueeze(-1)
    return mask

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        # Self attention for the query
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate)
        
        # Multi-head attention for query and key/value from encoder (x)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout_rate)
        
        # Dropouts for post-attention
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Another layer normalization for the feed-forward network
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query, x, query_mask=None):
        # Self attention
        # print(query_mask.shape,"skjdjkflskjlfmsd,mf,sdsssss
        self_attn_output, _ =  self.self_attn(query, query, query, key_padding_mask=query_mask)
        query = self.norm1(query + self.dropout1(self_attn_output))
        
        # Cross attention
        # print(query.shape,x.shape)
        attn_output, _ = self.cross_attn(query, x, x)
        query = self.norm2(query + self.dropout2(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(query)
        output = self.norm3(query + ffn_output)
        
        return output

class MultiLayerDecoder(nn.Module):
    def __init__(self, d_model, nhead, query_dim, num_layers, max_seq_length, dropout_rate=0.0, num_classes=8):
        super(MultiLayerDecoder, self).__init__()
        
        # Query transformation
        self.query_transform = nn.Linear(query_dim, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, dropout_rate) for _ in range(num_layers)])
        
        # Classification layer
        # self.classifier = nn.Linear(d_model,  8)

        # CLS token & position embedding
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length+1, d_model))
        self.register_buffer("positional_embedding", sinusoids(max_seq_length+1, d_model))
    def forward(self, query, x, query_lengths):
        batch_size = query.size(0)
        
        # Transform query
        query = self.query_transform(query)
        #query =torch.nn.functional.interpolate(query, size=1024, mode='nearest' )

        # Add CLS token
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        query = torch.cat([cls_tokens, query], dim=1)
        # print(query.size(1) ,"kjlkjlk")
        # Add position embedding
        query += self.positional_embedding#self.pos_embedding#[:, :query.size(1), :]
        query = query.transpose(0, 1)  # Transposed for MHA compatibility
        
        # Adjust query_lengths for added CLS token
        query_lengths += 1
        
        # Generate the mask based on query lengths
        query_mask = create_mask(query_lengths, query.shape[0])
        
        # Pass query through each decoder layer
        for layer in self.layers:
            query = layer(query, x,query_mask )
        
        # Using CLS token for classification
        cls_output = query[0]
        
        
        return cls_output  
        
import torch.nn.functional as F

feature =  1024
 
class T(nn.Module):
    def __init__(self  ):
        super(T, self).__init__()
       
        model = whisper.load_model("medium.en" )
       
        self.encoder=   model.encoder
        d_model =   feature
        nhead = 16
        query_dim = 35
        num_layers =   8
        self.transform = nn.Linear(d_model, d_model)
        self.decoder =   MultiLayerDecoder(d_model, nhead, query_dim, num_layers,110)
    


    def forward(self, x,au,v1):
    

        x=self.encoder(x)
        x1= (x).transpose(0, 1)
     
        
        logits = self.decoder(au,x1,v1)
     
      
        return logits 
 

@torch.no_grad()
def e(loader,x):

    we=[]
    for i in  loader:
    

        audio=i[0].to(f'cuda:{x.device_ids[0]}') 
        # print(len(audio),"wewewewe")
        au=i[1].to(f'cuda:{x.device_ids[0]}') 
        v1=i[2].to(f'cuda:{x.device_ids[0]}')

        label= (i[3]-1).to(f'cuda:{x.device_ids[0]}')
        # print(label.shape,"dskjhskjdk")
        
        tensor= x(audio,au,v1)
        we.append((tensor,label))

    # with open(name, 'wb') as file:
    #     pickle.dump(we, file)


    tt=[]
    ttt=[]
    for i in we:
        
        for k in i[0]:
            
            tt.append(k)
        for k in i[1]:
            
            ttt.append(k)
    
    

    w=torch.cat([t.unsqueeze(0) for t in tt], dim=0)
    w1=torch.cat([t.unsqueeze(0) for t in ttt], dim=0)


    we=[]
    we.append(w.cpu().numpy())
    we.append(w1.cpu().numpy())
  
    return we


 


def eval(x,loaded_data,episode):
   
    
    a,b,v1,c,de,ee,v2e,fe=loaded_data 

    dataset = TensorDataset( a,b,v1,c)
    test_dataset = TensorDataset(de,ee,v2e,fe)

 
 

    model_train_embedding=e(DataLoader(dataset, batch_size=100, shuffle=False),x)
    model_test_embedding=e(DataLoader(test_dataset, batch_size=100, shuffle=False),x)

     
      
    X_train, X_test, y_train, y_test =   model_train_embedding[0],  model_test_embedding[0],model_train_embedding[1],  model_test_embedding[1]#train_test_split(X, y,  

     
    X_train, y_train=model_train_embedding#loaded_list
    
 
    X_test, y_test = model_test_embedding#loaded_listt
 
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define the neural network architecture with non-linear layers
    import torch.nn as nn
    import torch.nn.functional as F
    dropout=0.2
    class NeuralNet(nn.Module):

        def __init__(self):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(feature, 768)
            self.batch_norm1 = nn.BatchNorm1d(768)
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(768, 512)
            self.batch_norm2 = nn.BatchNorm1d(512)
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(512, 256)
            self.batch_norm3 = nn.BatchNorm1d(256)
            self.dropout3 = nn.Dropout(dropout)
            self.fc4 = nn.Linear(256, 128)
            self.batch_norm4 = nn.BatchNorm1d(128)
            self.dropout4 = nn.Dropout(dropout)
            self.fc5 = nn.Linear(128, 64)
            self.batch_norm5 = nn.BatchNorm1d(64)
            self.fc6 = nn.Linear(64, 8)

        def forward(self, x):
            x = F.relu(self.batch_norm1(self.fc1(x)))
            x = self.dropout1(x)
            x = F.relu(self.batch_norm2(self.fc2(x)))
            x = self.dropout2(x)
            x = F.relu(self.batch_norm3(self.fc3(x)))
            x = self.dropout3(x)
            x = F.relu(self.batch_norm4(self.fc4(x)))
            x = self.dropout4(x)
            x = F.relu(self.batch_norm5(self.fc5(x)))
            x = self.fc6(x)
            return x
     

    
    
    


    # Initialize the model, loss function, and optimizer
    model =NeuralNet() 
    model=model.to("cuda:1")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=0.1)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # Forward pass
            inputs, labels=inputs.to("cuda:1"),labels.to("cuda:1")
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels=inputs.to("cuda:1"),labels.to("cuda:1")
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
 
    return accuracy

  
 

def torch_kron(a, b):
    a_shape = [a.size(0), a.size(1)]
    b_shape = [b.size(0), b.size(1)]
    return torch.reshape(torch.reshape(a, [a_shape[0], 1, a_shape[1], 1]) * torch.reshape(b, [1, b_shape[0], 1, b_shape[1]]), [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]])



from sklearn.model_selection import train_test_split

 
def featurate_mapping_and_classification():
    
    torch.cuda.empty_cache()



    with open("raw_data.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
    a,b,v1,c,actor=loaded_data


    a,aa,b,bb,v1,v11,c,cc =train_test_split(  a,b,v1,c ,test_size=0.3 )
    loaded_data=a,b,v1,c,aa,bb,v11,cc


    x=T()
    x=nn.DataParallel(x, device_ids = [0,1,2,3])
    x.to(f'cuda:{x.device_ids[0]}')




    loss_function = nn.CrossEntropyLoss()

    n=4
    h= feature
    labell = torch.arange(n, dtype=torch.int64).to(f'cuda:{x.device_ids[0]}')
    p = torch.tensor(np.ones((n, 1)), dtype=torch.float32).to(f'cuda:{x.device_ids[0]}')
    ones_tensor = torch.tensor(np.ones((1, n)), dtype=torch.float32).to(f'cuda:{x.device_ids[0]}')
    A=np.identity(n) 
    L=np.kron(A,np.ones([h,1])) 
    M = torch.tensor(L, dtype=torch.float32).to(f'cuda:{x.device_ids[0]}')

    lr=0.00001
    optimizer = torch.optim.Adam(params= x.parameters(), lr=lr   ) 


    total_steps =401





    for i in range(total_steps):



        ww=[]
        wwe=[]
        selected_indices = np.random.choice(8,n , replace=False)
        for ii in range(n):

            indices_where_equal_to_3 = torch.nonzero( (c-1) == selected_indices[ii], as_tuple=False)



            your_tensor =  indices_where_equal_to_3


            random_indices = np.random.choice( your_tensor.size(0), 2, replace=False)  #torch.randperm(your_tensor.size(0))[:2]

            selected_numbers = your_tensor[random_indices]




            ww.append(selected_numbers[0])
            wwe.append(selected_numbers[1])

        dd=ww+wwe
        d = np.stack([tensor.numpy() for tensor in  dd])
        d=d.T 


        optimizer.zero_grad()


        input1=a[d].to(f'cuda:{x.device_ids[0]}') 
        au=b[d].to(f'cuda:{x.device_ids[0]}')
        valid=v1[d].to(f'cuda:{x.device_ids[0]}')



        tensor2  = x(input1,au,valid)




        t1=tensor2[0:n]
        t2=tensor2[n:n*2]




        z = torch.matmul(p, t1.reshape(1, n*h)) - torch_kron(ones_tensor, t2)


        logits = -torch.matmul(torch.square(z), M)




        loss =     loss_function(logits,labell)
        loss.backward()



        optimizer.step()

        if i==400:

            acc=eval(x,loaded_data, i )
            print( acc)

            
if __name__ == "__main__":
    featurate_mapping_and_classification()
            

