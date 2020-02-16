import torch


def plot_llk(test_elbo, model_type):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy as sp
    import seaborn as sns
    plt.figure(figsize=(30, 10))
    sns.set_style("whitegrid")
    data = np.concatenate([np.arange(len(test_elbo))[:, sp.newaxis], -test_elbo[:, sp.newaxis]], axis=1)
    df = pd.DataFrame(data=data, columns=['Training Epoch', 'Test ELBO'])
    g = sns.FacetGrid(df, size=10, aspect=1.5)
    g.map(plt.scatter, "Training Epoch", "Test ELBO")
    g.map(plt.plot, "Training Epoch", "Test ELBO")
    plt.savefig('./{}_results/test_elbo.png'.format(model_type))
    plt.close('all')


def plot_vae_samples(vae, visdom_session):
    vis = visdom_session
    x = torch.zeros([1, 784])
    for i in range(10):
        images = []
        for rr in range(100):
            # get loc from the model
            sample_loc_i = vae.model(x)
            img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
            images.append(img)
        vis.images(images, 10, 2)


def mnist_test_tsne(vae, test_loader, model_type):
    """
    This is used to generate a t-sne embedding of the vae
    """
    data = test_loader.dataset.data.float()
    mnist_labels = test_loader.dataset.targets
    z_loc, z_scale = vae.encoder(data)
    plot_tsne(z_loc, mnist_labels, model_type)


def plot_tsne(z_loc, classes, model_type):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    if z_states.shape[-1] > 2:
        z_embed = model_tsne.fit_transform(z_states)
    else:
        z_embed = z_states
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        ind_class = classes == ic
        color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=1, color=color)
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig('./{}_results/embedding_'.format(model_type)+str(ic)+'.png')
    fig.savefig('./{}_results/embedding.png'.format(model_type))
