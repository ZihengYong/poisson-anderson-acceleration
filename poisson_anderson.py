"""
Matrix-Free 2D Poisson Solver & Anderson Acceleration Benchmark
===============================================================
Solves the 2D Poisson equation:
    -Delta u(x, y) = 2 * (x(1 - x) + y(1 - y))  on (0, 1)^2
    u = 0 on the boundary
with exact solution u*(x, y) = x(1 - x)y(1 - y).

Compares standard Richardson iteration against Anderson Acceleration
(history window m in {2, 5, 10}) on a 250 x 400 interior grid (N = 100,000).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt


def make_problem(nx: int, ny: int):
    """Construct grid parameters, RHS vector b, and exact solution u_exact."""
    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)
    x = np.linspace(hx, 1.0 - hx, nx)
    y = np.linspace(hy, 1.0 - hy, ny)
    X, Y = np.meshgrid(x, y)  # shape: (ny, nx)
    u_exact = (X * (1.0 - X) * Y * (1.0 - Y)).ravel()
    b = (2.0 * (X * (1.0 - X) + Y * (1.0 - Y))).ravel()
    tau = 1.0 / (2.0 / hx**2 + 2.0 / hy**2)
    return hx, hy, tau, b, u_exact


def apply_A(v: np.ndarray, nx: int, ny: int, hx: float, hy: float) -> np.ndarray:
    """Matrix-free 5-point finite-difference Laplacian operator Av."""
    V = v.reshape(ny, nx)
    inv_hx2 = 1.0 / hx**2
    inv_hy2 = 1.0 / hy**2
    out = (2.0 * inv_hx2 + 2.0 * inv_hy2) * V
    out[:, 1:]  -= inv_hx2 * V[:, :-1]
    out[:, :-1] -= inv_hx2 * V[:, 1:]
    out[1:, :]  -= inv_hy2 * V[:-1, :]
    out[:-1, :] -= inv_hy2 * V[1:, :]
    return out.ravel()


def run_richardson(nx, ny, hx, hy, tau, b, u_exact, tol=1e-8, max_evals=1000):
    """Standard Richardson fixed-point iteration."""
    v = np.zeros(nx * ny)
    norm_b = np.linalg.norm(b)
    norm_u = np.linalg.norm(u_exact)
    res_hist, err_hist = [], []

    t0 = time.perf_counter()
    for eval_idx in range(1, max_evals + 1):
        res_vec = b - apply_A(v, nx, ny, hx, hy)
        R_n = np.linalg.norm(res_vec) / norm_b
        E_n = np.linalg.norm(v - u_exact) / norm_u
        res_hist.append(R_n)
        err_hist.append(E_n)

        if not (np.isfinite(R_n) and np.isfinite(E_n)):
            return res_hist, err_hist, eval_idx, time.perf_counter() - t0, "nonfinite"
        if R_n < tol or eval_idx == max_evals:
            status = "converged" if R_n < tol else "max_evals"
            return res_hist, err_hist, eval_idx, time.perf_counter() - t0, status

        v = v + tau * res_vec


def run_anderson(nx, ny, hx, hy, tau, b, u_exact, m=5, tol=1e-8, max_evals=1000):
    """Anderson Acceleration with rolling history length m."""
    v = np.zeros(nx * ny)
    norm_b = np.linalg.norm(b)
    norm_u = np.linalg.norm(u_exact)
    res_hist, err_hist = [], []
    g_list, r_list = [], []

    t0 = time.perf_counter()
    for n in range(max_evals):
        eval_idx = n + 1
        res_vec = b - apply_A(v, nx, ny, hx, hy)
        R_n = np.linalg.norm(res_vec) / norm_b
        E_n = np.linalg.norm(v - u_exact) / norm_u
        res_hist.append(R_n)
        err_hist.append(E_n)

        if not (np.isfinite(R_n) and np.isfinite(E_n)):
            return res_hist, err_hist, eval_idx, time.perf_counter() - t0, "nonfinite"
        if R_n < tol or eval_idx == max_evals:
            status = "converged" if R_n < tol else "max_evals"
            return res_hist, err_hist, eval_idx, time.perf_counter() - t0, status

        r_n = tau * res_vec
        g_n = v + r_n
        g_list.append(g_n)
        r_list.append(r_n)
        if len(g_list) > m + 1:
            g_list.pop(0)
            r_list.pop(0)

        q = min(m, n)
        if q == 0:
            v = g_n
        else:
            dR = np.column_stack([r_list[i + 1] - r_list[i] for i in range(q)])
            dG = np.column_stack([g_list[i + 1] - g_list[i] for i in range(q)])
            gamma, _, _, _ = np.linalg.lstsq(dR, r_n, rcond=None)
            v = g_n - dG @ gamma


def main():
    # 1. Verification on a small 12 x 18 grid
    hx_s, hy_s, tau_s, b_s, u_s = make_problem(12, 18)
    verif_err = np.linalg.norm(apply_A(u_s, 12, 18, hx_s, hy_s) - b_s) / np.linalg.norm(b_s)
    r_rich, _, _, _, _ = run_richardson(12, 18, hx_s, hy_s, tau_s, b_s, u_s, max_evals=50)
    r_aa0, _, _, _, _ = run_anderson(12, 18, hx_s, hy_s, tau_s, b_s, u_s, m=0, max_evals=50)
    aa0_diff = np.max(np.abs(np.array(r_rich) - np.array(r_aa0)))
    print(f"[Verification 12x18] ||Au* - b||/||b|| = {verif_err:.3e}, max|Rich - AA(m=0)| = {aa0_diff:.3e}")

    # 2. Full 250 x 400 grid benchmark (N = 100,000)
    nx, ny = 250, 400
    hx, hy, tau, b, u_exact = make_problem(nx, ny)
    results = {"Richardson": run_richardson(nx, ny, hx, hy, tau, b, u_exact)}
    for m in [2, 5, 10]:
        results[f"Anderson (m={m})"] = run_anderson(nx, ny, hx, hy, tau, b, u_exact, m=m)

    print("\n=== Full 250x400 Grid Results (N = 100,000) ===")
    print(f"{'Method':<18} | {'Final R_n':<11} | {'Final E_n':<11} | {'Evals':<6} | {'Time (s)':<8} | {'Status'}")
    print("-" * 75)
    for name, (rh, eh, ev, tm, st) in results.items():
        print(f"{name:<18} | {rh[-1]:<11.3e} | {eh[-1]:<11.3e} | {ev:<6d} | {tm:<8.3f} | {st}")

    # 3. Generate convergence plots
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    colors = {
        "Richardson": "#7f8c8d",
        "Anderson (m=2)": "#2980b9",
        "Anderson (m=5)": "#27ae60",
        "Anderson (m=10)": "#c0392b",
    }
    for name, (rh, eh, ev, tm, st) in results.items():
        x_evals = np.arange(1, len(rh) + 1)
        axes[0].semilogy(x_evals, rh, label=f"{name} ({tm:.2f}s)", color=colors[name], linewidth=1.8)
        axes[1].semilogy(x_evals, eh, label=name, color=colors[name], linewidth=1.8)

    axes[0].set_title(r"Relative Residual $R_n = \|b - Av_n\|_2 / \|b\|_2$", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(r"Cumulative evaluations of $G$")
    axes[0].set_ylabel("Relative Residual (log scale)")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].set_title(r"Relative Error $E_n = \|v_n - u_*\|_2 / \|u_*\|_2$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(r"Cumulative evaluations of $G$")
    axes[1].set_ylabel("Relative Error (log scale)")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].legend(frameon=True, fontsize=9)

    plt.suptitle(
        r"2D Poisson Equation ($250 	imes 400$ Grid, $N = 10^5$): Richardson vs. Anderson Acceleration",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("results/anderson_convergence.png", bbox_inches="tight")
    print("\nSaved plot to results/anderson_convergence.png")


if __name__ == "__main__":
    main()
