#!/usr/bin/env python3
"""Can a fresh net learn a QUADRATIC and extrapolate it beyond the training range? (torch/GPU)

Continuous analogue of the parity test. Target y = x^2. Train on x in [-R,R], then test:
  interp : held-out x inside [-R,R]        (fits the curve in-range?)
  extrap : |x| in [R, Rx]  -- NEVER SEEN   (derives the quadratic rule forward?)

Arms (the representation / inductive-bias variable):
  raw     : input [x] -> ReLU MLP.  ReLU nets are piecewise-LINEAR -> they extrapolate with a straight
            line, so a quadratic's curvature is lost outside the training range. Fits in, fails out.
  raw_big : same, much wider -> shows CAPACITY does not buy extrapolation (still piecewise linear).
  poly    : input [x, x^2] -> the target is LINEAR in these features -> extrapolates ~exactly.

Lesson (same as parity): learning a function well enough to DERIVE it beyond the data is
representation/inductive-bias dependent, not a matter of fit quality or capacity in-range.
"""
import argparse
import torch
import torch.nn.functional as F


def target(x):
    return x ** 2


def encode(x, mode, R):
    if mode == "poly":
        return torch.stack([x / R, (x / R) ** 2], 1)     # normalized [x, x^2]
    return (x / R).unsqueeze(1)                           # raw scalar (normalized)


def mlp(din, h, seed, device, depth=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = {"W1": (torch.randn(din, h, generator=g) / din ** 0.5).to(device).requires_grad_(),
         "b1": torch.zeros(h, device=device, requires_grad=True)}
    P["W2"] = (torch.randn(h, h, generator=g) / h ** 0.5).to(device).requires_grad_()
    P["b2"] = torch.zeros(h, device=device, requires_grad=True)
    P["W3"] = (torch.randn(h, 1, generator=g) / h ** 0.5).to(device).requires_grad_()
    P["b3"] = torch.zeros(1, device=device, requires_grad=True)
    return P


def fwd(P, x):
    h = torch.relu(x @ P["W1"] + P["b1"])
    h = torch.relu(h @ P["W2"] + P["b2"])
    return (h @ P["W3"] + P["b3"]).squeeze(1)


def run(seed, mode, args, device):
    R, Rx, ynorm = args.R, args.Rx, args.R ** 2
    torch.manual_seed(seed)
    # train/interp inside [-R,R]; extrap in [-Rx,-R] U [R,Rx]
    xtr = (torch.rand(args.ntrain, device=device) * 2 - 1) * R
    xip = (torch.rand(1000, device=device) * 2 - 1) * R
    xex = torch.cat([torch.rand(500, device=device) * (Rx - R) + R,
                     -(torch.rand(500, device=device) * (Rx - R) + R)])
    h = args.h_big if mode == "raw_big" else args.h
    din = 2 if mode == "poly" else 1
    P = mlp(din, h, seed, device)
    opt = torch.optim.Adam([P[k] for k in P], lr=args.lr)
    ytr = target(xtr) / ynorm
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        F.mse_loss(fwd(P, encode(xtr, mode, R)), ytr).backward()
        opt.step()

    @torch.no_grad()
    def stats(x):
        pred = fwd(P, encode(x, mode, R)) * ynorm
        true = target(x)
        rmse = ((pred - true) ** 2).mean().sqrt().item() / ynorm     # normalized RMSE
        within = ((pred - true).abs() <= 0.15 * true.abs().clamp(min=1e-6)).float().mean().item()
        return rmse, within
    return stats(xtr), stats(xip), stats(xex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=4.0)      # train range [-R,R]
    ap.add_argument("--Rx", type=float, default=10.0)    # extrapolation out to |x|=Rx
    ap.add_argument("--ntrain", type=int, default=4000)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--h_big", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} QUADRATIC y=x^2 train=[-{args.R},{args.R}] "
          f"extrap|x|=[{args.R},{args.Rx}] h={args.h}/big{args.h_big} seeds={args.seeds}")
    print(f"{'arm':>8} | {'train_rmse':>10} {'interp_rmse':>11} {'extrap_rmse':>11} | "
          f"{'extrap_within15%':>15}   verdict")
    for mode in ["raw", "raw_big", "poly"]:
        acc = {"tr": [0, 0], "ip": [0, 0], "ex": [0, 0]}
        for s in range(args.seeds):
            (tr_r, tr_w), (ip_r, ip_w), (ex_r, ex_w) = run(s, mode, args, device)
            acc["tr"][0] += tr_r; acc["ip"][0] += ip_r; acc["ex"][0] += ex_r
            acc["ex"][1] += ex_w
        n = args.seeds
        trr, ipr, exr, exw = acc["tr"][0]/n, acc["ip"][0]/n, acc["ex"][0]/n, acc["ex"][1]/n
        v = ("DERIVES the quadratic (extrapolates)" if exw > 0.9 else
             "fits in-range, FAILS to extrapolate (linear outside)")
        print(f"{mode:>8} | {trr:>10.4f} {ipr:>11.4f} {exr:>11.4f} | {exw:>15.3f}   {v}")
    print("\nReLU MLP on raw x fits the curve in-range but extrapolates with a straight line -> the "
          "quadratic's curvature is lost (extrap error large, within-tol ~0). More width (raw_big) "
          "does not help. Give it x^2 as a feature (poly) and it extrapolates ~exactly. Same lesson "
          "as parity: DERIVING a function beyond the data needs the representation to expose it.")


if __name__ == "__main__":
    main()
