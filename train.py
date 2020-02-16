import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import visdom
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from models import VAE, GMMVAE
from vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples

model_type = 'gmmvae'
z_dim = 50
h_dim = 400
sep = 10
cats = None
n_epochs = 100
test_freq = 5

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)


# for loading and batching MNIST dataset
def setup_data_loaders(cats=None, batch_size=128, use_cuda=torch.cuda.is_available()):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    if cats is not None:
        train_set.data = train_set.data[np.isin(train_set.targets, cats)]
        train_set.targets = train_set.targets[np.isin(train_set.targets, cats)]
        test_set.data = test_set.data[np.isin(test_set.targets, cats)]
        test_set.targets = test_set.targets[np.isin(test_set.targets, cats)]

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    if not os.path.exists('{}_results'.format(model_type)):
        os.mkdir('{}_results'.format(model_type))

    if model_type == 'vae':
        vae = VAE(z_dim=z_dim, hidden_dim=h_dim)
    else:
        n_cats = 10 if cats is None else len(cats)
        vae = GMMVAE(n_cats=n_cats, z_dim=z_dim, hidden_dim=h_dim)

    optimizer = Adam({"lr": 1.0e-3})
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_loader, test_loader = setup_data_loaders(cats=cats, batch_size=256)

    vis = visdom.Visdom()

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(n_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if torch.cuda.is_available():
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % test_freq == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if torch.cuda.is_available():
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    plot_vae_samples(vae, vis)
                    reco_indices = np.random.randint(0, x.shape[0], 3)
                    for index in reco_indices:
                        test_img = x[index, :]
                        reco_img = vae.reconstruct_img(test_img)
                        vis.image(test_img.reshape(28, 28).detach().cpu().numpy(),
                                  opts={'caption': 'test image'})
                        vis.image(reco_img.reshape(28, 28).detach().cpu().numpy(),
                                  opts={'caption': 'reconstructed image'})

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

    mnist_test_tsne(vae, test_loader, model_type)
    plot_llk(np.array(test_elbo), model_type)

    vae.eval()
    torch.save(vae.state_dict(), '{}_results/model.pth'.format(model_type))