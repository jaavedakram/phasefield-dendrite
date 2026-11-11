"""Many grains in one phase field, orientation carried as a per-cell field.

The usual multi-grain phase field gives every grain its own order parameter, which
costs one field per grain. Here there is a single phi, and each cell carries its
crystal orientation, inherited from its solid neighbours when it starts to solidify.
Cost per cell is O(1) whatever the grain count.

Orientation travels as (cos 4*theta0, sin 4*theta0) rather than theta0 itself: those
are the only combinations the four-fold anisotropy needs, and they are linear, so a
cell can inherit a weighted average of its neighbours without angle wraparound.

Inheritance is frozen, not diffused. A cell takes its orientation once, at the moment
it starts to solidify, and never changes it after; smoothing every step would bleed
orientation across grain boundaries and slowly merge the polycrystal into one crystal.

KNOWN LIMIT: with a single phi there is no grain-boundary energy, so two grains that
touch merge into one solid region. That is fine for soft, solutal impingement --
grains stall a few cells apart because the liquid between them enriches from both
sides -- and wrong for hard impingement. `touching()` reports whether any pair reached
contact, so the assumption is checked rather than assumed.
"""
import argparse, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from phasefield import PhaseField, pad

NEIGHBOURS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


class MultiGrain(PhaseField):
    def __init__(self, n=400, dx=0.8, k=0.113, eps4=0.05, Dtilde=2.0, omega=0.45,
                 seeds=None, ngrain=4, seed_r=6.0, dt=None, device='cpu',
                 iso=True, rng=0, assign_phi=-0.90):
        # the parent seeds one grain at the centre; we overwrite phi entirely
        super().__init__(n=n, dx=dx, k=k, eps4=eps4, Dtilde=Dtilde, omega=omega,
                         theta0=0.0, seed_r=seed_r, dt=dt, device=device, iso=iso)
        self.assign_phi = assign_phi
        g = np.random.default_rng(rng)
        if seeds is None:
            lo, hi = int(0.18*n), int(0.82*n)
            pos = g.integers(lo, hi, (ngrain, 2))
            ang = g.uniform(0.0, 90.0, ngrain)
            seeds = [(int(a), int(b), float(c)) for (a, b), c in zip(pos, ang)]
        self.seeds = seeds

        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        phi = np.full((n, n), -1.0)
        c4 = np.zeros((n, n))
        s4 = np.zeros((n, n))
        got = np.zeros((n, n), dtype=bool)
        for (ci, cj, a) in seeds:
            r = np.hypot((yy-ci)*dx, (xx-cj)*dx)
            p = np.tanh((seed_r - r)/np.sqrt(2.0))
            phi = np.maximum(phi, p)
            m = r < seed_r + 2.0*dx
            t = math.radians(a)
            c4[m], s4[m] = math.cos(4*t), math.sin(4*t)
            got |= m
        self.phi = torch.tensor(phi, dtype=torch.float64, device=device)
        self.c4 = torch.tensor(c4, dtype=torch.float64, device=device)
        self.s4 = torch.tensor(s4, dtype=torch.float64, device=device)
        self.got = torch.tensor(got, device=device)
        self.angles = np.array([a for (_, _, a) in seeds])

    def aniso(self, gx, gy):
        """a_s = 1 + eps4*cos(4(theta - theta0)) with theta0 a per-cell field.

        Expanded through the angle-difference identity so only cos/sin of 4*theta0
        are needed, which is what the field actually carries."""
        th = torch.atan2(gy, gx)
        c, s = torch.cos(4.0*th), torch.sin(4.0*th)
        a = 1.0 + self.eps4*(c*self.c4 + s*self.s4)
        ap = -4.0*self.eps4*(s*self.c4 - c*self.s4)
        return a, ap

    def inherit(self):
        """Give newly-solidifying unassigned cells their neighbours' orientation.

        Weighted by (phi+1)/2 so the solid side dominates, restricted to already-
        assigned neighbours, and applied once per cell."""
        nd = (~self.got) & (self.phi > self.assign_phi)
        if not bool(nd.any()):
            return
        n = self.n
        w = ((self.phi + 1.0)/2.0)*self.got.double()
        wp, cp, sp = pad(w), pad(self.c4*w), pad(self.s4*w)
        wsum = torch.zeros_like(w); csum = torch.zeros_like(w); ssum = torch.zeros_like(w)
        for di, dj in NEIGHBOURS:
            wsum = wsum + wp[1+di:1+di+n, 1+dj:1+dj+n]
            csum = csum + cp[1+di:1+di+n, 1+dj:1+dj+n]
            ssum = ssum + sp[1+di:1+di+n, 1+dj:1+dj+n]
        ok = nd & (wsum > 1e-12)
        self.c4 = torch.where(ok, csum/torch.clamp(wsum, min=1e-30), self.c4)
        self.s4 = torch.where(ok, ssum/torch.clamp(wsum, min=1e-30), self.s4)
        # renormalise: the average of unit vectors is not a unit vector, and a
        # short one would silently weaken the anisotropy rather than rotate it
        mag = torch.sqrt(self.c4**2 + self.s4**2)
        good = ok & (mag > 1e-9)
        self.c4 = torch.where(good, self.c4/mag, self.c4)
        self.s4 = torch.where(good, self.s4/mag, self.s4)
        self.got = self.got | ok

    def step(self, nsub=1):
        for _ in range(nsub):
            super().step(1)
            self.inherit()

    # ---- diagnostics ----------------------------------------------------
    def grain_id(self):
        """Nearest seed by orientation, for per-grain accounting."""
        t = torch.atan2(self.s4, self.c4)/4.0
        ids = torch.full(t.shape, -1, dtype=torch.long, device=t.device)
        best = torch.full(t.shape, 1e9, dtype=torch.float64, device=t.device)
        for i, a in enumerate(self.angles):
            ta = math.radians(a)
            d = torch.abs(torch.atan2(torch.sin(4*(t-ta)), torch.cos(4*(t-ta))))
            sel = (d < best) & self.got
            best = torch.where(sel, d, best)
            ids = torch.where(sel, i, ids)
        return ids

    def per_grain_area(self):
        ids = self.grain_id()
        sol = self.phi > 0
        return [float(((ids == i) & sol).sum())*self.dx**2
                for i in range(len(self.angles))]

    def touching(self):
        """Solid cells whose 8-neighbourhood holds a different grain.

        Non-zero means two grains have met, and with a single phi there is no
        grain boundary there -- the single-field assumption has run out."""
        ids = self.grain_id()
        sol = (self.phi > 0) & self.got
        n = self.n
        ip = F.pad(ids[None, None].double(), (1, 1, 1, 1), mode='replicate')[0, 0]
        sp = F.pad(sol[None, None].double(), (1, 1, 1, 1), mode='replicate')[0, 0]
        idd = ids.double()
        diff = torch.zeros_like(idd)
        for di, dj in NEIGHBOURS:
            nid = ip[1+di:1+di+n, 1+dj:1+dj+n]
            nsl = sp[1+di:1+di+n, 1+dj:1+dj+n]
            diff = diff + ((nid != idd) & (nsl > 0.5)).double()
        return int(((diff > 0) & sol).sum())


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--n', type=int, default=400)
    P.add_argument('--grains', type=int, default=4)
    P.add_argument('--omega', type=float, default=0.45)
    P.add_argument('--eps4', type=float, default=0.05)
    P.add_argument('--steps', type=int, default=20000)
    P.add_argument('--every', type=int, default=2000)
    P.add_argument('--rng', type=int, default=0)
    P.add_argument('--seeds', default=None,
                   help='explicit placement "i,j,angle;i,j,angle;..." (cells, deg). '
                        'Random placement can bunch grains together.')
    P.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    A = P.parse_args()
    sd = None
    if A.seeds:
        sd = [(int(i), int(j), float(a))
              for i, j, a in (p.split(',') for p in A.seeds.split(';'))]
    m = MultiGrain(n=A.n, ngrain=A.grains, omega=A.omega, eps4=A.eps4,
                   device=A.device, rng=A.rng, seeds=sd)
    print(f"{len(m.angles)} grains on {A.n}x{A.n}, omega={A.omega}, eps4={A.eps4}, "
          f"dt={m.dt}, device={A.device}")
    print("orientations " + ", ".join(f"{a:.1f}" for a in m.angles) + " deg\n")
    c0 = m.solute()
    print(f"{'step':>8} {'t':>8} {'solid%':>8} {'touching':>9} {'drift':>11}")
    for i in range(0, A.steps+1, A.every):
        if i:
            m.step(A.every)
        sol = float((m.phi > 0).sum())/(A.n*A.n)*100
        print(f"{m.steps:8d} {m.t:8.1f} {sol:7.2f}% {m.touching():9d} "
              f"{(m.solute()-c0)/abs(c0):+11.2e}", flush=True)
    print("\nper-grain solid area (W0^2): "
          + ", ".join(f"{a:.0f}" for a in m.per_grain_area()))
