"""
Code adapted from 
https://github.com/patrick-kidger/NeuralCDE/blob/7e529f58441d719d2ce85f56bdee3208a90d5132/experiments/models/other.py

- removed the use of natural cubic spline for imputation
- adapted to our dataset format (X, M, t, s, y)
  especially, the missingness indicator is 1 if non-missing else 0. (i.e. delta instead of cumulative.)
"""

# import pathlib
# import sys
import torch
import torchdiffeq
from torch import nn
import torch.nn.functional as F

# here = pathlib.Path(__file__).resolve().parent
# sys.path.append(str(here / '..'))

# import controldiffeq

"""
their X looks like this:
    (delta t, M, X)
"""

class _GRU(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, rnn_type):
        super(_GRU, self).__init__()

        # assert (input_channels % 2) == 1, "Input channels must be odd: 1 for time, plus 1 for each actual input, " \
        #                                   "plus 1 for whether an observation was made for the actual input."
        self.rnn_type = rnn_type
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        gru_channels = input_channels * 2 + 1 ## GRU on (X, M, t) stacked.
        if rnn_type == 'GRU':
            self.gru_cell = torch.nn.GRUCell(input_size=gru_channels, hidden_size=hidden_channels)
        elif rnn_type == 'LSTM':
            self.gru_cell = torch.nn.LSTMCell(input_size=gru_channels, hidden_size=hidden_channels)
        elif rnn_type == 'RNN':
            self.gru_cell = torch.nn.RNNCell(input_size=gru_channels, hidden_size=hidden_channels)
        else:
            raise ValueError("Unknown RNN type {}".format(rnn_type))
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    # def extra_repr(self):
    #     return "input_channels={}, hidden_channels={}, output_channels={}, use_intensity={}" \
    #            "".format(self.input_channels, self.hidden_channels, self.output_channels, self.use_intensity)

    def evolve(self, h, time_diff):
        raise NotImplementedError

    # def _step(self, Xi, Mi, ti, h, dt):
    #     """
    #     Note: ti is delta t, instead of t.
    #     """
    #     observation = Mi.max(dim=1).values > 0.5 # whether there is an observation for this time step. (padded will be skipped)
    #     if observation.any():
    #         new_h = self.gru_cell(torch.cat([Xi, Mi, ti + dt], dim=-1), h)
    #         h = torch.where(observation.unsqueeze(1), new_h, h) # update only where there is an observation. 
    #         dt += torch.where(observation, torch.tensor(0., dtype=Xi.dtype, device=Xi.device), ti) # if there is observation, add 0. else add ti.
    #     return h, dt

    def _step(self, Xi, Mi, ti, h):
        """
        Did not understand their logic of dt.
        We rewrite it.
        By our way of padding, if there is no observation for all the features of one patient, deltati=0.
        ti is the cumulated t, not delta t.
        """
        observation = Mi.max(dim=1).values > 0.5 # whether there is an observation for this time step. (padded will be skipped)
        if self.rnn_type == 'LSTM':
            h, c = h  # Unpack h into h and c
            if observation.any():
                new_h, new_c = self.gru_cell(torch.cat([Xi, Mi, ti], dim=-1), (h, c))
                h = torch.where(observation.unsqueeze(1), new_h, h)
                c = torch.where(observation.unsqueeze(1), new_c, c)
            h = (h, c)
        else:
            if observation.any():
                new_h = self.gru_cell(torch.cat([Xi, Mi, ti], dim=-1), h)
                h = torch.where(observation.unsqueeze(1), new_h, h) # update only where there is an observation. 
                # dt += torch.where(observation, torch.tensor(0., dtype=Xi.dtype, device=Xi.device), ti) # if there is observation, add 0. else add ti.
        return h


    # def forward(self, times, coeffs, final_index, z0=None):
    def forward(self, X, M, t, z0=None):

        # interp = controldiffeq.NaturalCubicSpline(times, coeffs)
        # X = torch.stack([interp.evaluate(t) for t in times], dim=-2)
        # half_num_channels = (self.input_channels - 1) // 2

        # change cumulative intensity into intensity i.e. was an observation made or not, which is what is typically
        # used here
        # X[:, 1:, 1:1 + half_num_channels] -= X[:, :-1, 1:1 + half_num_channels]

        # change times into delta-times
        # X[:, 0, 0] -= times[0]
        # X[:, 1:, 0] -= times[:-1]
        
        # Set the initial time to 0.
        tc = t.clone()
        tc -= t[0, :]

        # Compute the differences dt[i, :] = t[i, :] - t[i-1, :] for i > 0
        dt_diff = tc[1:, :] - tc[:-1, :]

        # Concatenate t[0, :] with the differences
        deltat = torch.cat((torch.zeros_like(tc[0:1, :]), dt_diff), dim=0) # the first deltat is set to 0.

        batch_dims = X.shape[:-2]

        if z0 is None:
            if self.rnn_type == 'LSTM':
                z0 = (torch.zeros(*batch_dims, self.hidden_channels, dtype=X.dtype, device=X.device),
                      torch.zeros(*batch_dims, self.hidden_channels, dtype=X.dtype, device=X.device))
            else:
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=X.dtype, device=X.device)

        X_unbound = X.unbind(dim=1)
        M_unbound = M.unbind(dim=1)
        t_unbound = tc.unbind(dim=1)
        deltat_unbound = deltat.unbind(dim=1)
        # set the first hidden state to z0.
        h = self._step(X_unbound[0], M_unbound[0], t_unbound[0], z0)
        # hs = [h]
        # time_diffs = times[1:] - times[:-1]
        for Xi, Mi, ti, deltati in zip(X_unbound[1:], M_unbound[1:], t_unbound[1:], deltat_unbound[1:]):
            h = self.evolve(h, deltati)
            h = self._step(Xi, Mi, ti, h)
            # hs.append(h)
        # out = torch.stack(hs, dim=1)

        """
        We don't need final index either because our data is padded at the front.
        """
        if self.rnn_type == 'LSTM':
            h = h[0]
        # final_index_indices = final_index.unsqueeze(-1).expand(out.size(0), out.size(2)).unsqueeze(1)
        # final_out = out.gather(dim=1, index=final_index_indices).squeeze(1)
        # return self.linear(final_out)

        return self.linear(h)


class GRU_dt(_GRU):
    def evolve(self, h, time_diff):
        return h


class GRU_D(_GRU):
    def __init__(self, input_channels, hidden_channels, output_channels, rnn_type):
        super(GRU_D, self).__init__(input_channels=input_channels,
                                    hidden_channels=hidden_channels,
                                    output_channels=output_channels, rnn_type=rnn_type)
        self.decay = torch.nn.Linear(1, hidden_channels)

    def evolve(self, h, time_diff):
        return h * torch.exp(-self.decay(time_diff.unsqueeze(0)).squeeze(0).relu())


class _ODERNNFunc(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(_ODERNNFunc, self).__init__()

        layers = [torch.nn.Linear(hidden_channels, hidden_hidden_channels)]
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_channels))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, t, x):
        return self.sequential(x)


class ODERNN(_GRU):
    def __init__(self, input_channels, hidden_channels, output_channels, hidden_hidden_channels, num_hidden_layers, rnn_type):
        super(ODERNN, self).__init__(input_channels=input_channels,
                                     hidden_channels=hidden_channels,
                                     output_channels=output_channels, rnn_type=rnn_type)
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.func = _ODERNNFunc(hidden_channels, hidden_hidden_channels, num_hidden_layers)

    # def extra_repr(self):
    #     return "hidden_hidden_channels={}, num_hidden_layers={}".format(self.hidden_hidden_channels,
    #                                                                     self.num_hidden_layers)

    def evolve(self, h, time_diff):
        # t = torch.tensor([0, time_diff.item()], dtype=time_diff.dtype, device=time_diff.device)
        ## t must in 1-dimension, so can't use batched time; also, cuz the nn takes time as a channel, we can set this 
        ## fake "time" to s\in[0, 1], and assume the nn can learn g(t, x) = f(x)dt/ds(s).
        ## because this is ode rnn, we only care about the last time step. 
        t = torch.tensor([0, 1], dtype=time_diff.dtype, device=time_diff.device)
        out = torchdiffeq.odeint_adjoint(func=self.func, y0=h, t=t, method='rk4')
        return out[1]

class RNNModel(nn.Module):
    def __init__(self, s_dim, X_feat_dim, hidden_dim, output_dim, model_type, rnn_type, mean_dyn=0., std_dyn=1., mean_stat=0., std_stat=1.):
        super(RNNModel, self).__init__()
        self.mean_dyn = mean_dyn
        self.std_dyn = std_dyn
        self.mean_stat = mean_stat
        self.std_stat = std_stat
        
        # Submodel A: MLP for static features (s)
        self.mlp_A = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Submodel B: RNN model for dynamic features (X)
        self.rnn = make_rnn_model(model_type, X_feat_dim, hidden_dim, hidden_dim, rnn_type)
        
        # MLP to process concatenated outputs of A and the last time step of B
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def set_normalization(self, mean_dyn, std_dyn, mean_stat, std_stat):
        self.mean_dyn = mean_dyn
        self.std_dyn = std_dyn
        self.mean_stat = mean_stat
        self.std_stat = std_stat
    
    def normalize(self, X, M, t, s):
        X = (X - self.mean_dyn) / self.std_dyn
        s = (s - self.mean_stat) / self.std_stat
        return X, M, t, s

    def forward(self, X, M, t, s):
        X, M, t, s = self.normalize(X, M, t, s)
        # Forward pass for submodel A
        out_A = self.mlp_A(s)
        
        # Forward pass for submodel B
        # Embed dynamic data features to hidden_dim using MLP
        out_B = self.rnn(X, M, t) # TODO z0.
        
        # Extracting the last time step from the output of B
        last_time_step_B = out_B # for our _GRU it returns the last hidden state.
        
        # Concatenating the outputs of A and the last time step of B
        concatenated_out = torch.cat((out_A, last_time_step_B), dim=1)
        
        # Final MLP to produce the output y
        y_pred = self.final_mlp(concatenated_out)
        
        return y_pred
    
    def get_latent_repn(self, X, M, t, s):
        out_A = self.mlp_A(s)
        
        # Forward pass for submodel B
        # Embed dynamic data features to hidden_dim using MLP
        out_B = self.rnn(X, M, t) # TODO z0.
        
        # Extracting the last time step from the output of B
        last_time_step_B = out_B # for our _GRU it returns the last hidden state.
        
        # Concatenating the outputs of A and the last time step of B
        concatenated_out = torch.cat((out_A, last_time_step_B), dim=1)
        
        # Final MLP to produce the output y
        latent_repn = self.final_mlp[0](concatenated_out)
        latent_repn = self.final_mlp[1](latent_repn)
        return latent_repn

def make_rnn_model(model_type, input_dim, hidden_dim, output_dim, rnn_type):
    if model_type == 'GRU-dt':
        return GRU_dt(input_dim, hidden_dim, output_dim, rnn_type)
    elif model_type == 'GRU-D':
        return GRU_D(input_dim, hidden_dim, output_dim, rnn_type)
    elif model_type == 'ODERNN':
        return ODERNN(input_dim, hidden_dim, output_dim, hidden_dim, 1, rnn_type) # TODO hard-coded hyperparameters here.
    else:
        raise ValueError("Unknown model type {}".format(model_type))