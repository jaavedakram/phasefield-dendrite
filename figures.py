"""The README figures: one dendrite, and several grains competing for solute.

    python figures.py            # writes docs/dendrite.png and docs/multigrain.png

Each panel shows phi (solid in white) beside the solute c/c_l0, where the rejected
solute piles up ahead of the tips and between the arms.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phasefield import PhaseField
from multigrain import MultiGrain

OUT = Path(__file__).resolve().parent/'docs'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def concentration(m):
    k = m.k
    return ((1 + (1 - k)*m.U)*(1 + k - (1 - k)*m.phi)/2).cpu().numpy()


def save(m, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    ax[0].imshow(m.phi.cpu().numpy(), cmap='bone', vmin=-1, vmax=1, origin='lower')
    ax[0].set_title('phase field  $\\phi$')
    c = concentration(m)
    im = ax[1].imshow(c, cmap='inferno', origin='lower',
                      vmin=np.percentile(c, 0.5), vmax=np.percentile(c, 99.5))
    ax[1].set_title('solute  $c/c_l^0$')
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


def single(n=600, omega=0.45, theta0=15.0, rstop=0.8):
    m = PhaseField(n=n, omega=omega, theta0=theta0, device=DEV)
    while m.tip() < rstop*n*m.dx/2:
        m.step(1000)
    save(m, f'single grain, $\\Omega$ = {omega}, $\\theta_0$ = {theta0:.0f}$^\\circ$, '
            f't = {m.t:.0f} $\\tau_0$', OUT/'dendrite.png')


def multi(n=600, omega=0.45, tend=1200.0):
    seeds = [(150, 150, 0.0), (160, 440, 30.0), (440, 140, 15.0),
             (450, 450, 40.0), (300, 300, 22.0)]
    m = MultiGrain(n=n, omega=omega, seeds=seeds, device=DEV)
    while m.t < tend and not m.touching():
        m.step(1000)
    save(m, f'{len(seeds)} grains, $\\Omega$ = {omega}, t = {m.t:.0f} $\\tau_0$',
         OUT/'multigrain.png')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    single()
    multi()
