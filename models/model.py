import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.2):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        # src = src.transpose(0, 1) # B, T, C -> T, B, C
        # tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src

class TTMNet(nn.Module):
    def __init__(self, vidio_dim, audio_dim, output_dim):
        super(TTMNet, self).__init__()
        self.output_dim = output_dim
        feature_dim = 128
        self.vis_layer = nn.Linear(vidio_dim, feature_dim)
        self.aud_layer = nn.Linear(audio_dim, feature_dim)

        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = feature_dim, nhead = 8)
        self.crossV2A = attentionLayer(d_model = feature_dim, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 2*feature_dim, nhead = 8)

        self.predict_layer = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(2*feature_dim, output_dim, bias=True),
        )


    def forward(self, x_video, x_audio):
        if x_audio.size(0) < x_video.size(0)*3:
            padding = torch.zeros(x_video.size(0)*3, 128).to('cuda')
            padding[:x_audio.size(0)] = x_audio
            x_audio = padding
        x_audio = x_audio[:x_video.size(0)*3].view(x_video.size(0), -1)

        x_video = self.vis_layer(x_video) # N x 128
        x_audio = self.aud_layer(x_audio) # N x 128

        x1_c = self.crossA2V(src = x_video, tar = x_audio)
        x2_c = self.crossV2A(src = x_audio, tar = x_video)

        x = torch.cat((x1_c, x2_c), 1)
        x = self.selfAV(x, x)

        output = self.predict_layer(x).squeeze(1)
        if self.output_dim == 1:
            return output.mean()
        else:
            return output.mean(0)
        
class MLP(nn.Module):
    def __init__(self, vidio_dim, audio_dim, output_dim):
        super(MLP, self).__init__()
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(vidio_dim + audio_dim, 1024, bias=True),

            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, output_dim, bias=True),
        )

    def forward(self, x_video, x_audio):

        if x_audio.size(0) < x_video.size(0)*3:
            padding = torch.zeros(x_video.size(0)*3, 128).to('cuda')
            padding[:x_audio.size(0)] = x_audio
            x_audio = padding
        x_audio = x_audio[:x_video.size(0)*2].view(x_video.size(0), -1)
        output = self.layers(torch.cat((x_video, x_audio), 1)).squeeze(1)

        if self.output_dim == 1:
            return output.mean()
        else:
            return output.mean(0)

def load_model(vidio_dim, audio_dim, output_dim, ckpt = None):
    model = TTMNet(vidio_dim, audio_dim, output_dim)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    return model

if __name__ == '__main__':
    model = load_model(1024, 13680, 2)
    print(model)
