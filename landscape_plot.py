import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, rcParams

def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize=16
    else:
        fontsize=int(fontsize)

    rc('font',**{'family':'serif'})
    rc('text', usetex=True)

    #plt.rcParams.update({'figure.facecolor':'w'})
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'

    return

set_rc_params(fontsize=20)


#loss_landscape = np.load("loss_landscape_small_25.npy")
#loss_landscape = np.load("loss_landscape_small_100.npy")
loss_landscape = np.load("loss_landscape_100.npy")

x = np.linspace(-1, 1, loss_landscape.shape[0])
y = np.linspace(-1, 1, loss_landscape.shape[1])
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, loss_landscape, cmap='viridis')

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Loss')
ax.set_title('3D Loss Landscape of the Model')
ax.set_xlabel(r'$\delta \Theta $', labelpad=15)
ax.set_ylabel(r'$\delta \Theta _\perp$', labelpad=15)

#ax.set_xticks(np.arange(-1, 1.5, 0.5))
#ax.set_yticks(np.arange(-1, 1.5, 0.5))

plt.savefig("loss_landscape.png")
