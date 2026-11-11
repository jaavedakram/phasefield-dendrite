"""Quantitative phase-field model of dendritic solidification of a dilute binary alloy, 2D.

Echebarria, Folch, Karma & Plapp, Phys. Rev. E 70, 061604 (2004): isothermal free
growth into a supersaturated melt, with the anti-trapping current that makes the model
quantitative at a realistic interface width. The same model as PRISMS-PF's
alloy_solidification application.

Fields:
  phi   order parameter, +1 solid, -1 liquid, tanh profile across the interface
  U     dimensionless supersaturation; U = -omega in the far-field melt

Units are the model's own: W0 = tau0 = 1, so lengths are in interface widths and
times in tau0. lambda is tied to the diffusivity by the thin-interface relation
lambda = D/a2, which removes interface kinetics; do not set them independently.

Anisotropy is analytic, a_s = 1 + eps4*cos(4(theta - theta0)), so crystal orientation
is a free parameter rather than something the grid decides. Laplacians use a 9-point
isotropic stencil: the 5-point one adds a grid-locked four-fold term of the same size
as the physical anisotropy (13.7% at dx = 0.8, eps4 = 0; 2.8% with 9 points).

Time stepping is explicit Euler, in float64, on any torch device.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

A2 = 0.6267                      # thin-interface constant, Echebarria Eq (130)
A_T = 1.0/(2.0*np.sqrt(2.0))     # anti-trapping coefficient


def pad(f):
    """Replicate padding = zero-flux walls."""
    return F.pad(f[None, None], (1, 1, 1, 1), mode='replicate')[0, 0]


class PhaseField:
    def __init__(self, n=400, dx=0.8, k=0.113, eps4=0.05, Dtilde=2.0,
                 omega=0.45, theta0=0.0, seed_r=8.0, dt=None, device='cpu',
                 iso=True):
        self.iso = iso
        self.n, self.dx, self.k = n, dx, k
        self.eps4, self.D = eps4, Dtilde
        self.lam = Dtilde/A2                      # thin-interface coupling
        self.theta0 = np.deg2rad(theta0)
        self.omega = omega
        self.dev = device
        # explicit stability: the phi equation carries W0^2/tau0 = 1, the U
        # equation carries D. The 0.8 factor is headroom, not a derived bound.
        self.dt = dt if dt else 0.8*dx*dx/(4.0*max(Dtilde, 1.0))

        y, x = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        c = (n-1)/2.0
        r = np.hypot((y-c)*dx, (x-c)*dx)
        phi = np.tanh((seed_r - r)/np.sqrt(2.0))
        self.phi = torch.tensor(phi, dtype=torch.float64, device=device)
        self.U = torch.full((n, n), -omega, dtype=torch.float64, device=device)
        self.t = 0.0
        self.steps = 0

    # ---- geometry -------------------------------------------------------
    def grads(self, f):
        fp = pad(f)
        gx = (fp[1:-1, 2:] - fp[1:-1, :-2])/(2*self.dx)
        gy = (fp[2:, 1:-1] - fp[:-2, 1:-1])/(2*self.dx)
        return gx, gy

    def aniso(self, gx, gy):
        th = torch.atan2(gy, gx) - self.theta0
        a = 1.0 + self.eps4*torch.cos(4.0*th)
        ap = -4.0*self.eps4*torch.sin(4.0*th)
        return a, ap

    # link weights: 5-point, and the 9-point isotropic form (2/3 axial, 1/6
    # diagonal) whose leading error has no four-fold component
    LINKS_5 = ((0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0))
    LINKS_9 = ((0, 1, 2/3), (0, -1, 2/3), (1, 0, 2/3), (-1, 0, 2/3),
               (1, 1, 1/6), (1, -1, 1/6), (-1, 1, 1/6), (-1, -1, 1/6))

    def div_face(self, coef, f):
        """Conservative divergence of coef*grad(f) over the link stencil.

        Each link carries the face-averaged coefficient, so the operator stays
        symmetric in the link and conserves exactly."""
        n = self.n
        cp, fp = pad(coef), pad(f)
        out = torch.zeros_like(f)
        for di, dj, w in (self.LINKS_9 if self.iso else self.LINKS_5):
            cs = cp[1+di:1+di+n, 1+dj:1+dj+n]
            fs = fp[1+di:1+di+n, 1+dj:1+dj+n]
            out = out + w*0.5*(coef + cs)*(fs - f)
        return out/(self.dx**2)

    def div_vec(self, vx, vy):
        vxp, vyp = pad(vx), pad(vy)
        return ((vxp[1:-1, 2:] - vxp[1:-1, :-2])
                + (vyp[2:, 1:-1] - vyp[:-2, 1:-1]))/(2*self.dx)

    # ---- the model ------------------------------------------------------
    def rhs(self):
        phi, U, k = self.phi, self.U, self.k
        gx, gy = self.grads(phi)
        a, ap = self.aniso(gx, gy)

        # grad-energy term plus the two anisotropy cross terms, which come from
        # d(theta)/d(phi_x) = -phi_y/|grad phi|^2 and +phi_x/|grad phi|^2
        lap = self.div_face(a*a, phi)
        B = a*ap
        cross = -self.div_vec(B*gy, torch.zeros_like(gy)) \
            + self.div_vec(torch.zeros_like(gx), B*gx)
        bulk = (phi - phi**3) - self.lam*(1.0 - phi**2)**2 * U
        dphi = (lap + cross + bulk)/(a*a)          # tau(n) = tau0 * a_s^2

        # solute: one-sided diffusion in the liquid, plus the anti-trapping flux
        q = (1.0 - phi)/2.0
        diff = self.D*self.div_face(q, U)
        g = torch.sqrt(gx*gx + gy*gy) + 1e-12
        amp = A_T*(1.0 + (1.0 - k)*U)*dphi
        jat = self.div_vec(amp*gx/g, amp*gy/g)
        src = (1.0 + (1.0 - k)*U)/2.0 * dphi
        denom = (1.0 + k)/2.0 - (1.0 - k)*phi/2.0
        dU = (diff + jat + src)/denom
        return dphi, dU

    def step(self, nsub=1):
        for _ in range(nsub):
            dphi, dU = self.rhs()
            self.phi = self.phi + self.dt*dphi
            self.U = self.U + self.dt*dU
            self.t += self.dt
            self.steps += 1

    # ---- diagnostics ----------------------------------------------------
    def solute(self):
        """The conserved quantity: c/c_l0 = (1+(1-k)U)(1+k-(1-k)phi)/2."""
        k = self.k
        c = (1.0 + (1.0 - k)*self.U)*(1.0 + k - (1.0 - k)*self.phi)/2.0
        return float(c.sum())*self.dx**2

    def tip(self):
        """Furthest solid cell from the seed, in units of W0."""
        n = self.n; c = (n-1)/2.0
        soli = (self.phi > 0).nonzero()
        if len(soli) == 0:
            return 0.0
        yy = soli[:, 0].double() - c
        xx = soli[:, 1].double() - c
        return float(torch.sqrt(xx*xx + yy*yy).max())*self.dx


def run(steps=20000, every=2000, verbose=True, **kw):
    m = PhaseField(**kw)
    c0 = m.solute()
    hist = []
    for i in range(0, steps+1, every):
        if i:
            m.step(every)
        R = m.tip()
        drift = (m.solute()-c0)/abs(c0)
        hist.append((m.steps, m.t, R, drift))
        if verbose:
            print(f"  step {m.steps:7d}  t {m.t:9.2f}  Rtip {R:8.2f} W0  "
                  f"solute drift {drift:+.3e}", flush=True)
        if not np.isfinite(R) or R > 0.42*m.n*m.dx:
            print("  stopping: diverged or reached the wall", flush=True)
            break
    return m, hist


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('--n', type=int, default=400)
    P.add_argument('--dx', type=float, default=0.8)
    P.add_argument('--k', type=float, default=0.113)
    P.add_argument('--eps4', type=float, default=0.05)
    P.add_argument('--D', type=float, default=2.0)
    P.add_argument('--omega', type=float, default=0.45)
    P.add_argument('--theta0', type=float, default=0.0)
    P.add_argument('--steps', type=int, default=20000)
    P.add_argument('--every', type=int, default=2000)
    P.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    A = P.parse_args()
    print(f"Echebarria 2004, n={A.n}, dx={A.dx} W0, k={A.k}, eps4={A.eps4}, "
          f"D={A.D}, omega={A.omega}, theta0={A.theta0} deg, device={A.device}")
    m, h = run(steps=A.steps, every=A.every, n=A.n, dx=A.dx, k=A.k, eps4=A.eps4,
               Dtilde=A.D, omega=A.omega, theta0=A.theta0, device=A.device)
    print(f"dt = {m.dt:.4f} tau0, lambda = {m.lam:.3f}")
