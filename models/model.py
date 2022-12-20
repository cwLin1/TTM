import torch 
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1, bias=True)
        )

    def forward(self, x_video, x_audio):
        x_video = nn.functional.normalize(x_video)
        x_audio = nn.functional.normalize(x_audio.contiguous().view(x_audio.size(0), -1))
        return self.layers(torch.cat((x_video, x_audio), 1)).squeeze(1)

def load_model(input_dim, ckpt = None):
    model = MLP(input_dim)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    return model

if __name__ == '__main__':
    model = load_model(1024+13680)
    print(model)