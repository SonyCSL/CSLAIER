# -*- coding: utf-8 -*-
# HINT:text
import chainer
import chainer.functions as F
import chainer.links as L

##############################
## DO NOT CHANGE CLASS NAME ##
##############################

class Network(chainer.Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(CharRNN,self).__init__(
        
            embed = L.EmbedID(n_vocab, n_units),
            l1 = L.LSTM(n_units,n_units),
            l2 = L.LSTM(n_units,n_units),
            l3 = L.LSTM(n_units,n_units),
            l4 = L.LSTM(n_units,n_units),
            l5 = L.LSTM(n_units,n_units),
            l6 = L.Linear(n_units,n_vocab),
            
        )
        
        self.train = train
        
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.l5.reset_state()

    def __call__(self,x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        h3 = self.l3(F.dropout(h2, train=self.train))
        h4 = self.l4(F.dropout(h3, train=self.train))
        h5 = self.l5(F.dropout(h4, train=self.train))
        y = self.l6(F.dropout(h5, train=self.train))
        
        return y
    
    def predict(self,x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        h3 = self.l3(F.dropout(h2, train=self.train))
        h4 = self.l4(F.dropout(h3, train=self.train))
        h5 = self.l5(F.dropout(h4, train=self.train))
        y = self.l6(h5)
        
        return F.softmax(y)