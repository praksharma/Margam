"""GPU benchmark: python -m examples.benchmark_poisson --device cuda."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import uuid

import torch
import DeepINN as dp
from DeepINN.benchmarking import BenchmarkCase, Recipe, run_benchmark
from .baseline_poisson import residual


def build_problem():
    return dp.PINNProblem(
        residual, lambda: torch.linspace(0, 1, 34)[1:-1, None],
        [dp.PointConstraint("boundary", torch.tensor([[0.0], [1.0]]))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("benchmark-results") / f"poisson-{stamp}-{uuid.uuid4().hex[:8]}"
    case = BenchmarkCase(
        "poisson_1d", build_problem, torch.linspace(0, 1, 101)[:, None],
        lambda x: x*(1-x),
        config={"equation": "u_xx + 2 = 0", "domain": [0, 1], "boundary_value": 0,
                "collocation_points": 32, "evaluation_points": 101})

    def baseline(problem):
        network = dp.nn.FullyConnected(
            [1, 16, 16, 1], "tanh", "Xavier normal", input_bounds=([0.0], [1.0]))
        return dp.PINN(problem, network).compile(lr=0.01, device=args.device)

    recipe = Recipe("baseline_pinn", baseline, config={
        "network": [1, 16, 16, 1], "activation": "tanh", "initialiser": "Xavier normal",
        "input_bounds": [[0.0], [1.0]], "optimiser": "adam", "lr": 0.01,
        "weights": {"pde": 1, "boundary": 1}, "device": args.device})
    results = run_benchmark(case, [recipe], seeds=args.seeds,
                            iterations=args.iterations, output_dir=output)
    for result in results:
        print(f"seed={result['seed']} device={result['device']} "
              f"RMSE={result['rmse']:.6g} relative L2={result['relative_l2']:.6g} "
              f"training={result['training_seconds']:.3f}s")
    print(f"Saved results: {output.resolve()}")


if __name__ == "__main__":
    main()
