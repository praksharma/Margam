"""Run from the repository root: python -m examples.baseline_poisson."""
import argparse
import torch
import margam as dp


def residual(x, u):
    # Strong-form u_xx + 2 = 0 on [0, 1]. Exact solution: u = x(1-x).
    return dp.utils.laplacian(u, x) + 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    problem = dp.PINNProblem(
        residual=residual,
        collocation_points=lambda: torch.linspace(0, 1, 34)[1:-1, None],
        constraints=[dp.PointConstraint("boundary", torch.tensor([[0.0], [1.0]]))],
    )
    network = dp.nn.FullyConnected(
        [1, 16, 16, 1], "tanh", "Xavier normal", input_bounds=([0.0], [1.0]))
    pinn = dp.PINN(problem, network).compile(lr=0.01, device=args.device)
    print(f"Training device: {pinn.device}")
    if pinn.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(pinn.device)}")
    pinn.train(1000, display_every=250)
    x = torch.linspace(0, 1, 101, device=pinn.device)[:, None]
    rmse = torch.mean((pinn.predict(x) - x*(1-x))**2).sqrt().item()
    print(f"Solution RMSE: {rmse:.6g}")


if __name__ == '__main__':
    main()
