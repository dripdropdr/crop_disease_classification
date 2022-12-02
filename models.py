import torch
from torch import nn
import timm

class ViTBase16(nn.Module):
    def __init__(self, n_classes):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.model.load_state_dict(torch.load("./pretrain/jx_vit_base_p16_224-80ecf9dd.pth"))
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, inputs):
        output = self.model(inputs)
        return output


class LSTM_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(LSTM_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 41) # keras의 dense layer와 같은 역할인 듯? (fully connected layer!!)
        self.final_layer = nn.Linear(41+41, class_n) # image out_dim (vit) + time series out_dim (lstm) : final layer에서 concat !!
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1) # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output


# ViT encoder + lstm decoder
class ViT2RNN(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, num_classes, rate):
        super(ViT2RNN, self).__init__()
        self.vit = ViTBase16(num_classes=num_classes) # image model
        self.lstm = LSTM_Decoder(max_len, embedding_dim, num_features, num_classes, rate) # time series model
        
    def forward(self, img, seq):
        vit_output = self.vit(img) 
        output = self.lstm(vit_output, seq)
        
        return output

