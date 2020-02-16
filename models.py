import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist


class VAE(nn.Module):

    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=torch.cuda.is_available()):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x,z)=p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", self.z_prior(x.shape[0]).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    # define a helper function to sample from p(z) and output p(x|z)
    def sample_img(self, num_samples, return_z=False):
        # sample from p(z)
        z = self.z_prior(num_samples).sample()
        loc_img = self.decoder.forward(z)
        if return_z:
            return loc_img, z
        else:
            return loc_img

    def z_prior(self, num_samples):
        # sample from p(z)
        z_loc = torch.zeros(num_samples, self.z_dim)
        z_scale = torch.ones(num_samples, self.z_dim)
        z = dist.Normal(z_loc, z_scale)
        return z


class GMMVAE(VAE):

    def __init__(self, n_cats, loc_sep=3.0, z_dim=50, hidden_dim=400, use_cuda=torch.cuda.is_available()):
        super().__init__(z_dim=z_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
        self.n_cats = n_cats
        self.component_locs = torch.zeros(self.n_cats, self.z_dim)
        for i in range(n_cats):
            self.component_locs[i, i] = 1 * loc_sep

    def z_prior(self, num_samples):
        cats = dist.Categorical(torch.ones(num_samples, self.n_cats) * 1 / self.n_cats).sample()
        z_loc = self.component_locs[cats]
        z_scale = torch.ones(num_samples, self.z_dim)
        z = dist.Normal(z_loc, z_scale)
        return z


class Encoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img
