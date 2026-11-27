"""Reproducible multi-seed FBPINN vs Baseline benchmarks at equal data and update budget.

Run with:
    .venv/bin/python -m examples.fbpinn_benchmark --device cpu --output assets/benchmarks/fbpinn-multiscale-cpu.json
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
import torch

from margam.templates.examples import example_config
from margam.templates.poisson import build_pinn


def _git_info():
    try:
        rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
        return dict(revision=rev, modified=dirty)
    except Exception:
        return dict(revision='unknown', modified=None)


def _metrics(pred, exact):
    diff = pred - exact
    rmse = float(diff.square().mean().sqrt().item())
    rel_l2 = float((diff.norm() / exact.norm()).item())
    max_err = float(diff.abs().max().item())
    return dict(rmse=rmse, relative_l2=rel_l2, max_error=max_err)


def _run_case(name, configs, probe, exact_fn, seeds, device):
    exact = exact_fn(probe).to('cpu')
    methods_out = {}
    for label, builder in configs.items():
        runs = []
        for seed in seeds:
            cfg = builder(seed, device)
            model = build_pinn(cfg)
            params = sum(p.numel() for p in model.network.parameters())
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.train(cfg.iterations, display_every=0)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            pred = model.predict(probe).detach().cpu()
            err = _metrics(pred, exact)
            last = model.loss_history[-1]
            runs.append(dict(
                seed=seed,
                parameters=params,
                wall_seconds=round(elapsed, 4),
                final_pde_loss=float(last['pde']),
                final_total_loss=float(last['total']),
                **err,
            ))
        rel_values = [r['relative_l2'] for r in runs]
        rmse_values = [r['rmse'] for r in runs]
        time_values = [r['wall_seconds'] for r in runs]
        methods_out[label] = dict(
            parameters=runs[0]['parameters'],
            median_relative_l2=statistics.median(rel_values),
            mean_relative_l2=statistics.fmean(rel_values),
            median_rmse=statistics.median(rmse_values),
            median_wall_seconds=round(statistics.median(time_values), 4),
            runs=runs,
        )
    return dict(case=name, evaluation_points=len(probe), methods=methods_out)


def _multiscale_1d_configs():
    def base(seed, device, layers):
        c = example_config('fbpinn_multiscale_1d')
        c.seed = seed
        c.network_seed = seed
        c.device = device
        c.method = 'baseline'
        c.layers = layers
        return c

    def fb(seed, device, multilevel):
        c = example_config('fbpinn_multiscale_1d')
        c.seed = seed
        c.network_seed = seed
        c.device = device
        c.fbpinn_settings['multilevel'] = multilevel
        return c

    return {
        'baseline_16x16': lambda s, d: base(s, d, (16, 16)),
        'baseline_64x64_param_matched': lambda s, d: base(s, d, (64, 64)),
        'fbpinn_1level_12sub': lambda s, d: fb(s, d, False),
        'fbpinn_2level_12sub': lambda s, d: fb(s, d, True),
    }


def _multiscale_2d_configs():
    def base(seed, device, layers):
        c = example_config('fbpinn_multiscale_2d')
        c.seed = seed
        c.network_seed = seed
        c.device = device
        c.method = 'baseline'
        c.layers = layers
        return c

    def fb(seed, device, multilevel):
        c = example_config('fbpinn_multiscale_2d')
        c.seed = seed
        c.network_seed = seed
        c.device = device
        c.fbpinn_settings['multilevel'] = multilevel
        return c

    return {
        'baseline_16x16': lambda s, d: base(s, d, (16, 16)),
        'baseline_64x64_param_matched': lambda s, d: base(s, d, (64, 64)),
        'fbpinn_2level_4x4': lambda s, d: fb(s, d, True),
    }


def _wave_1d_configs():
    def base(seed, device):
        c = example_config('fbpinn_wave_progressive')
        c.seed = seed
        c.network_seed = seed
        c.device = device
        c.method = 'baseline'
        return c

    def fb(seed, device, schedule):
        c = example_config('fbpinn_wave_progressive')
        c.seed = seed
        c.network_seed = seed
        c.device = device
        c.fbpinn_settings['schedule'] = schedule
        return c

    return {
        'baseline_16x16': base,
        'fbpinn_simultaneous_2x4': lambda s, d: fb(s, d, 'simultaneous'),
        'fbpinn_progressive_2x4': lambda s, d: fb(s, d, 'progressive'),
    }


def main():
    parser = argparse.ArgumentParser(description='Run equal-budget FBPINN vs Baseline benchmarks.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19])
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable.')

    torch.set_num_threads(1)
    x1 = torch.linspace(0.0, 1.0, 501).unsqueeze(1)
    g2 = torch.linspace(0.0, 1.0, 41)
    xx, yy = torch.meshgrid(g2, g2, indexing='ij')
    x2 = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    results = [
        _run_case(
            'multiscale_poisson_1d',
            _multiscale_1d_configs(),
            x1,
            lambda x: torch.sin(2.0 * math.pi * x) + 0.5 * torch.sin(16.0 * math.pi * x),
            args.seeds,
            args.device,
        ),
        _run_case(
            'multiscale_poisson_2d',
            _multiscale_2d_configs(),
            x2,
            lambda x: (
                torch.sin(2.0 * math.pi * x[:, 0:1]) * torch.sin(2.0 * math.pi * x[:, 1:2])
                + 0.5 * torch.sin(8.0 * math.pi * x[:, 0:1]) * torch.sin(8.0 * math.pi * x[:, 1:2])
            ),
            args.seeds,
            args.device,
        ),
        _run_case(
            'standing_wave_1d_time',
            _wave_1d_configs(),
            x2,
            lambda x: (
                torch.sin(math.pi * x[:, 0:1]) * torch.cos(2.0 * math.pi * x[:, 1:2])
                + 0.5 * torch.sin(4.0 * math.pi * x[:, 0:1]) * torch.cos(8.0 * math.pi * x[:, 1:2])
            ),
            args.seeds,
            args.device,
        ),
    ]

    record = dict(
        study='fbpinn_equal_budget_comparison',
        git=_git_info(),
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            device=args.device,
            gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            dtype=str(torch.get_default_dtype()),
            cpu_threads=torch.get_num_threads(),
        ),
        protocol=dict(
            seeds=list(args.seeds),
            losses='Pre-update mean MSE recorded at every Adam step; predictions evaluated post-update on uniform grids.',
            notes='Equal collocation points, boundary sets, initial seeds, learning rate, loss weights, and update budgets across methods within each case.',
        ),
        cases=results,
    )
    text = json.dumps(record, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
        print(f'Wrote benchmark record to {args.output}')
    for case in results:
        print(f"\n=== {case['case']} ({args.device}) ===")
        for label, summary in case['methods'].items():
            print(
                f"  {label:30s} params={summary['parameters']:6d}  "
                f"med_relL2={summary['median_relative_l2']:.5f}  "
                f"med_rmse={summary['median_rmse']:.5f}  "
                f"med_s={summary['median_wall_seconds']:.2f}s"
            )


if __name__ == '__main__':
    main()
