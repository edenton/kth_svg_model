import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm2(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size, normalize=False):
        super(lstm2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm0 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()
        self.normalize = normalize

    def init_hidden(self):
        return  [(Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()), 
                 Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())),
                 (Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()), 
                 Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()))]

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        self.hidden[0] = self.lstm0(embedded, self.hidden[0])
        self.hidden[1] = self.lstm1(self.hidden[0][0], self.hidden[1])
        output = self.output(self.hidden[1][0])
        if self.normalize:
            return nn.functional.normalize(output, p=2)
        else:
            return output


class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, normalize=False):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()
        self.normalize = normalize

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        output = self.output(h_in)
        if self.normalize:
            return nn.functional.normalize(output, p=2)
        else:
            return output


class lstm_prior(nn.Module):
    def __init__(self, input_size, output_size, prior_dim, hidden_size, n_layers, batch_size, normalize=False):
        super(lstm_prior, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())

        self.mu_net = nn.Linear(hidden_size, prior_dim)
        self.logvar_net = nn.Linear(hidden_size, prior_dim)
        self.hidden = self.init_hidden()
        self.normalize = normalize

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        output = self.output(h_in)
        if self.normalize:
            output = nn.functional.normalize(output, p=2)
        return output, mu, logvar

class oracle_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size):
        super(oracle_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return  (Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()), 
                 Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()))



    def reparameterize(self, mu, logvar):
        if self.training:
            logvar = logvar.mul(0.5).exp_()
            eps = Variable(logvar.data.new(logvar.size()).normal_())
            return eps.mul(logvar).add_(mu)
        else:
            return mu

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        self.hidden = self.lstm(embedded, self.hidden)
        mu = self.mu_net(self.hidden[0])
        logvar = self.logvar_net(self.hidden[0])
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
            
