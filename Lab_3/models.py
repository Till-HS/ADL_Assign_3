import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageEncoder(nn.Module):
    def __init__(self, encoding_pixel_len):
        super().__init__()

        # load resnet without the two last layers
        resnet = resnet18(ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

        # resize encodings to encoding_size
        self.resize = nn.AdaptiveAvgPool2d((encoding_pixel_len, encoding_pixel_len))

    def forward(self, img):
        # img: input images (batch_size, 3, image_pixel_len, image_pixel_len)

        encoding = self.resnet(img) # batch_size, 512, resnet_output_pixel_len, resnet_output_pixel_len
        encoding = self.resize(encoding) # batch_size, 512, encoding_pixel_len, encoding_pixel_len
        encoding = encoding.permute(0,2,3,1) # batch_size, encoding_pixel_len, encoding_pixel_len, 512
        return encoding


class Attention(nn.Module):
    def __init__(self, encoding_dim, rnn_dim, attention_dim):
        super().__init__()

        # linear layers
        self.l_encoder = nn.Linear(encoding_dim, attention_dim)
        self.l_rnn = nn.Linear(rnn_dim, attention_dim)
        self.l_attention = nn.Linear(attention_dim, 1)

        # activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, rnn_hidden):
        # encoder_out: encoding returned by ImageEncoder (batch_size, encoding_pixel_size, encoding_dim)
        # rnn_hidden: hidden layer of the RNN (batch_size, decoding_dim)

        # put both input on attention_dim
        enc = self.l_encoder(encoder_out) # batch_size, encoding_pixel_size, attention_dim
        rnn = self.l_rnn(rnn_hidden) # batch_size, attention_dim

        # combine inputs and compute attention
        att = self.relu(enc + rnn.unsqueeze(1)) # batch_size, encoding_pixel_size, attention_dim
        att = self.l_attention(att).squeeze(2) # batch_size, encoding_pixel_size

        # normalize
        att = self.softmax(att) # batch_size, encoding_pixel_size

        # apply the attention on the encoding and keep one value per dimension
        out = encoder_out * att.unsqueeze(2) # batch_size, encoding_pixel_size, encoding_dim
        out = out.sum(dim=1) # batch_size, encoding_dim
        return out, att


class RNN(nn.Module):
    def __init__(self, embed_dim, rnn_dim, attention_dim, vocab_size, encoder_dim=512, dropout=0.5):
        super().__init__()

        # save all parameters
        self.encoder_dim = encoder_dim # number of dimensions that output the encoder
        self.rnn_dim = rnn_dim # number of dimensions that output the rnn
        self.attention_dim = attention_dim # number of dimensions used for attention
        self.vocab_size = vocab_size # size of the vocabulary
        self.embed_dim = embed_dim # number of dimensions that output the embedding

        # layers for initialization
        self.embedding = nn.Embedding(vocab_size, embed_dim) # create embeddings
        self.init_h = nn.Linear(encoder_dim, rnn_dim) # init hidden state of lstm
        self.init_c = nn.Linear(encoder_dim, rnn_dim) # init cell state of lstm

        # layers for forward loop
        self.lstmCell = nn.LSTMCell(embed_dim + encoder_dim, rnn_dim, bias=True) # lstm cell
        self.attention = Attention(encoder_dim, rnn_dim, attention_dim) # attention model
        self.l_gate = nn.Linear(rnn_dim, encoder_dim) # sigmoid gate
        self.l_out = nn.Linear(rnn_dim, vocab_size) # predict score for each word

        # activation function and dropout
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def init_lstm(self, encoder_out):
        # encoder_out: encoding returned by ImageEncoder (batch_size, encoding_pixel_size, encoding_dim)

        # mean over the pixels to change the dimension of tensor
        mean_encoder_out = encoder_out.mean(dim=1) # batch_size, encoding_dim

        # initialize hidden state and cell state
        hidden = self.init_h(mean_encoder_out) # batch_size, rnn_dim
        cell = self.init_c(mean_encoder_out) # batch_size, rnn_dim
        return hidden, cell

    def forward(self, encoder_out, captions, caption_lengths):
        # encoder_out: encoding returned by ImageEncoder (batch_size, encoding_pixel_len, encoding_pixel_len, encoding_dim)
        # captions: captions ready to be used (batch_size, max_caption_length)
        # caption_lengths: length of caption from <start> to <end> (batch_size, 1)

        # get useful data
        batch_size = encoder_out.size(0)

        # flatten encodings
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)  # batch_size, encoding_pixel_size, encoder_dim
        encoding_pixel_size = encoder_out.size(1)

        # Sort input data by decreasing lengths in order to not compute the padding in the lstm loop
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        captions = captions[sort_ind]

        # Initialization
        embeddings = self.embedding(captions)  # batch_size, max_caption_length, embed_dim
        hidden, cell = self.init_lstm(encoder_out)  # batch_size, rnn_dim

        # shorten the caption_lengths since we don't generate when we see the <end> token
        caption_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and attentions
        predictions = torch.zeros(batch_size, max(caption_lengths), self.vocab_size).to(device)
        attentions = torch.zeros(batch_size, max(caption_lengths), encoding_pixel_size).to(device)

        #LSTM loop
        for t in range(max(caption_lengths)):
            # look up which captions are finished to continue the loop only on the others
            batch_size_t = sum([l > t for l in caption_lengths])

            # apply attention
            attention_out, att = self.attention(encoder_out[:batch_size_t], hidden[:batch_size_t]) # batch_size, encoder_dim

            # apply sigmoid gate
            gate = self.sigmoid(self.l_gate(hidden[:batch_size_t])) # batch_size_t, encoder_dim
            attention_out = gate * attention_out # batch_size_t, encoder_dim

            # calculation of the new hidden and cell state
            hidden, cell = self.lstmCell(
                torch.cat([embeddings[:batch_size_t, t, :], attention_out], dim=1),
                (hidden[:batch_size_t], cell[:batch_size_t]))  # batch_size_t, rnn_dim

            # predicting words
            preds = self.l_out(self.dropout(hidden))  # batch_size_t, vocab_size

            # keeping results
            predictions[:batch_size_t, t, :] = preds
            attentions[:batch_size_t, t, :] = att

        return predictions, captions, caption_lengths, sort_ind

    def test(self, encoder_out, first_word, end_token, max_len=50):
        # encoder_out: encoding returned by ImageEncoder (1, encoding_pixel_len, encoding_pixel_len, encoding_dim)
        # first_word: <start> token (1)
        # end_token: <end> token (1)

        # flatten encodings
        encoder_out = encoder_out.view(1, -1, self.encoder_dim)  # 1, encoding_pixel_size, encoder_dim
        encoding_pixel_size = encoder_out.size(1)

        # Initialization
        prev_word = first_word
        hidden, cell = self.init_lstm(encoder_out)  # 1, rnn_dim

        # Create tensors to hold word prediction scores
        predictions = torch.zeros(max_len).to(device)
        predictions[0] = prev_word

        # LSTM loop
        for t in range(max_len):
            # compute embedding of last word
            emb = self.embedding(prev_word.view(1,1)) # 1, 1, embed_dim

            # apply attention
            attention_out, att = self.attention(encoder_out, hidden) # 1, encoder_dim

            # apply sigmoid gate
            gate = self.sigmoid(self.l_gate(hidden))  # 1, encoder_dim
            attention_out = gate * attention_out # 1, encoder_dim

            # calculation of the new hidden and cell state
            hidden, cell = self.lstmCell(
                torch.cat([emb[:, 0, :], attention_out], dim=1), (hidden, cell))  # 1, rnn_dim

            # predicting words
            preds = self.l_out(hidden)  # 1, vocab_size
            prev_word = torch.max(preds, 1).indices

            # keeping results
            predictions[t+1] = prev_word

            # end condition
            if prev_word == end_token:
                break

        return predictions.cpu().numpy()
