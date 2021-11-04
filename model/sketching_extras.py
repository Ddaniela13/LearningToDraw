import torch
import torch.nn as nn
import torchbearer as tb

from dsketch.experiments.characters.models import AdjustableSigmaMixin, metrics, Decoder
from dsketch.raster.disttrans import curve_edt2_polyline

ALPHAS = tb.state_key("alphas")


class _DecoderRNN(nn.Module):
    def __init__(self, hidden_size=512, output_size=8, stroke_enc_size=512, image_enc_size=64, steps=2,
                 attention_dim=512, attention=False, gru=True, dropout=0.5):
        super().__init__()

        self.hidden_size = hidden_size
        self.steps = steps
        self.output_size = output_size
        self.encode_stroke = nn.Linear(output_size, stroke_enc_size)
        self.init_h = nn.Linear(image_enc_size, hidden_size)
        self.attention = attention
        if gru:
            self.decode_step = nn.GRUCell(stroke_enc_size + image_enc_size, hidden_size)
        else:
            self.decode_step = nn.LSTMCell(stroke_enc_size + image_enc_size, hidden_size)
            self.init_c = nn.Linear(image_enc_size, hidden_size)
        self.decode_stroke = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)
        if attention:
            self.attn = Attention(image_enc_size, hidden_size, attention_dim)
            self.f_beta = nn.Linear(hidden_size, image_enc_size)  # linear layer to create a sigmoid-activated gate
        else:
            self.attn = lambda x, h: (x.mean(dim=1), torch.ones((x.shape[0], x.shape[1]), device=x.device))

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        if isinstance(self.decode_step, nn.GRUCell):
            return h

        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, encoder_out):
        device = encoder_out.device
        batch_size = encoder_out.shape[0]
        encoder_dim = encoder_out.shape[-1]

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.shape[1]

        # Embedding initial stroke
        decoder_input = torch.zeros((batch_size, self.output_size), device=device)
        decoder_input = self.encode_stroke(decoder_input)

        # init hidden state
        hc = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        h = hc[0] if isinstance(hc, tuple) else hc

        # Create tensors to hold strokes
        predictions = torch.zeros(batch_size, self.steps, self.output_size).to(device)
        alphas = torch.zeros(batch_size, self.steps, num_pixels).to(device)

        for i in range(self.steps):
            attention_weighted_encoding, alpha = self.attn(encoder_out, h)
            if self.attention:
                gate = torch.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding

            hc = self.decode_step(torch.cat([decoder_input, attention_weighted_encoding], dim=1),
                                  hc)  # (batch_size, decoder_dim)

            h = hc[0] if isinstance(hc, tuple) else hc
            preds = self.decode_stroke(self.dropout(h))  # (batch_size, output_size)

            predictions[:, i, :] = preds
            alphas[:, i, :] = alpha

        return predictions, alphas


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class RNNAttnBezierDecoder(AdjustableSigmaMixin, Decoder):
    def __init__(self, args, latent_size, steps, sz, sigma2, attention):
        super().__init__(args)
        self.sigma2 = sigma2
        self.rnn = _DecoderRNN(hidden_size=latent_size, steps=steps, attention=attention)
        r = torch.linspace(-1, 1, sz)
        c = torch.linspace(-1, 1, sz)
        grid = torch.meshgrid(r, c)
        grid = torch.stack(grid, dim=2)
        self.register_buffer("grid", grid)

    @staticmethod
    def _add_args(p):
        p.add_argument("--steps", help="number of steps for recurrent models", type=int, default=5, required=False)
        p.add_argument("--attention", help="Enable attention", default=False, required=False, action='store_true')
        AdjustableSigmaMixin._add_args(p)

    def create_edt2(self, params):
        return curve_edt2_polyline(params, self.grid, 10)

    @staticmethod
    def create(args):
        return RNNAttnBezierDecoder(args, args.latent_size, args.steps, args.size, args.sigma2,
                                    attention=args.attention)

    def decode_to_params(self, inp):
        params, alphas = self.rnn(inp)
        params = params.view(params.shape[0], -1, 4, 2)
        return params, alphas

    def forward(self, inp, state=None):
        params, alphas = self.decode_to_params(inp)
        sigma2 = self.get_sigma2(params)
        edt2 = self.create_edt2(params)
        images = self.raster_soft(edt2, sigma2)

        if state is not None:
            state[metrics.HARDRASTER] = self.raster_hard(edt2)
            state[metrics.SQ_DISTANCE_TRANSFORM] = edt2
            state[ALPHAS] = alphas

        return images * self.args.contrast

    def get_callbacks(self, args):
        cbs = super().get_callbacks(args)

        return cbs
