import torch 
import torch.nn as nn

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

        if x_audio.size(0) < x_video.size(0)*2:
            padding = torch.zeros(x_video.size(0)*2, 128).to('cuda')
            padding[:x_audio.size(0)] = x_audio
            x_audio = padding
        x_audio = x_audio[:x_video.size(0)*2].view(x_video.size(0), -1)
        output = self.layers(torch.cat((x_video, x_audio), 1)).squeeze(1)

        if self.output_dim == 1:
            return output.mean()
        else:
            return output.mean(0)

def load_model(vidio_dim, audio_dim, output_dim, ckpt = None):
    model = MLP(vidio_dim, audio_dim, output_dim)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
    return model

if __name__ == '__main__':
    model = load_model(1024, 13680, 2)
    print(model)