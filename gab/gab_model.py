#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CNN_LSTM(nn.Moddule):
    """
        Tensor Product Generation Network
        img -CNN-> feature vector -S-> seq. of words (recurrent)
    """

    def __init__(self, pretrained_embeddings, hidden_size, dropout_rate = 0.5):
        """
        Init TPGN model

        @pretrained_embeddings : word_idx -> word_vec, Tensor(|Vocab|, embed_size)
        """
        super(TPGN,self).__init__()

        # values of dimensions
        """
        @hidden_size : hiddens state of decoder LSTM
        @embed_size : pretrained word embedding
        @vocab_size : word vocabulary
        """
        self.hidden_size = hidden_size
        self.embed_size = pretrained_embeddings.size(1)
        self.vocab_size = pretrained_embeddings.size(0)


        # layers
        """
        @pretrained_embedding : (batch, vocab_size)->(batch, embed_size)
        @conv1 :
        @conv2 :
        @linear_sentence : (self+softmax) = dense layer:flattend_feature->encoded_feature
        @lstm_cell:
        @linear_words : (self+softmax) = dense layer:hidden_state -> word_probability_distribution
        """
        self.pretrained_embedding = nn.Embedding(pretrained_embeddings.size(0), pretrained_embeddings.size(1))
        self.pretrained_embedding.weight = nn.Parameter(torch.tensor(pretrained_embeddings))
            ## Depend on the size of training dataset, it can be omitted from the parameter. (small training dataset -> no parameter i.e. no finetuning i.e. constant) 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5) 
            # input : (1024,1024,3) -> ( , , 6)
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)
            # (,,6)-> (,,6)
        self.conv2 = nn.Conv2d(in_channels =6, out_channels = 15, kernel_size = 5)
            # (,,6)-> (,,15)
        self.dropout = nn.Dropout2d(dropout_rate)
        # Flatten layer
        self.linear_sentence = nn.Linear(in_features = (앞에서 flatten한 결과) , out_features = self.embed_size) ## encoder의 결과를, t=1에서 input으로 넣는다 가정했을때. h_0로 넣는다면 self.hidden_size로 바꿔야함 
            # (in_features)->(out_features), linear layer
        self.lstm_cell = nn.LSTMCell(input_size = self.embed_size, hidden_size = self.hidden_size)
        self.linear_words = nn.Linear(in_features=self.hidden_size , out_features=self.vocab_size)

"""
    init
    forward
        ㄴ encode_img
        ㄴ generate_caption_w_target
            ㄴ step
        ㄴ generate_caption_wo_target
            ㄴ step
"""

    def forward(self,x):
        """
        x(img) -> words(caption)
        """
        h_0 = encode_img(x)
        words = generate_caption(h_0)
        return words

    def encode_img(self, x): ## paper에선 ResNet 사용했음
        """
        x(img) -> 1-d embedding vector
        """
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x)
        h_0 = self.linear_sentence(x)
        return h_0

    def generate_caption_w_target(self, h_0, target_caption):
        """
        h_0(sentence embedding), target caption -LSTM-> list of (word probability distribution) 
        
        use for training. Forced teacher method

        @h_0 : (batch, hidden_size)
        @target_caption : (batch, num_of_words, len_of_Vocab) (batch*(seq of one-hot vectors))
            * starting from <START>??
        """
        batch_size = h_0.size(0) 
        num_of_words = target_caption.size(1)

        h=torch.zeros((batch_size, self.hidden_size)) # hidden state
        c=torch.zeors((batch_size, self.hidden_size)) # cell state
        hat_y_s = torch.empty((batch_size,num_of_words,self.vocab_size)) # tensor of output states(i.e. prob distributions)

        target_caption_embedding = self.pretrained_embedding(target_caption)
        # (batch, len(target_caption), embedding_size)

        for t in range(num_of_words):
            if t==0:
                h, c = self.lstm_cell(h_0, (h,c)) ## starting from <START> 면 여기서도 else와 같이 해야하는 것 아닌가?
                                                ## 2가지 architecture가 있는듯 - feature vector을  1) h_0에 2)x_1에(h_0는 zero vector) 
            else:
                h, c= self.lstm_cell(target_caption_embedding[:,t,:], (h,c))

            word_pb_distribution = nn.functional.log_softmax(self.linear_words(h))    
            hat_y_s[:,t,:] = word_pb_distribution

        return hat_y_s
        
    def generate_caption_wo_target(self,h_0):
        """
        h_0(sentence embedding)-LSTM-> list of (word probability distribution) 

        use for testing . Greedy algorithm (argmax)
        """

        batch_size = h_0.size(0) 
        num_of_words = target_caption.size(1)

        h=torch.zeros((batch_size, self.hidden_size)) # hidden state
        c=torch.zeors((batch_size, self.hidden_size)) # cell state
        hat_y_s = torch.empty((batch_size,num_of_words,self.vocab_size)) # tensor of output states(i.e. prob distributions)

        nextword_embedding = torch.zeros((batch_size,self.embed_size))

        for t in range(num_of_words):   ## <END> token 나왔을때 멈춰야 하는 것 아닌가?
            if t==0:
                h, c = self.lstm_cell(h_0, (h,c))
            else:
                h, c= self.lstm_cell(nextword_embedding, (h,c))

            word_pb_distribution = nn.functional.log_softmax(self.linear_words(h))
                # (batch, vocab_size)    
            hat_y_s[:,t,:] = word_pb_distribution

            max_pb_idx = torch.argmax(word_pb_distribution, dim=1)
                # (batch)
            nextword_embedding = self.pretrained_embedding(max_pb_idx)
                # (batch, embed_size)

        return hat_y_s
    

   # def step():
   # """
   #     one step in lstm cell
   # """
    # not used currently, just using self.lstm_cell( , (h,c))


