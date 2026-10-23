"""Benchmark data isolation, artefacts, metrics and real CUDA execution."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import DeepINN as dp
from DeepINN.benchmarking import BenchmarkCase, Recipe, run_benchmark, solution_errors


class ExactRecipe:
    device = torch.device("cpu")

    def train(self, iterations, display_every=0):
        self.loss_history = [{"pde": 0.0, "total": 0.0} for _ in range(iterations)]

    def predict(self, x):
        return x[:, 0].square()


def case():
    return BenchmarkCase(
        "quadratic", lambda: dp.PINNProblem(lambda x, u: u, lambda: torch.rand(4, 1)),
        torch.linspace(0, 1, 7)[:, None], lambda x: x.square())


class BenchmarkTests(unittest.TestCase):
    def test_metrics_and_zero_reference(self):
        errors = solution_errors(torch.tensor([2., 4.]), torch.tensor([[1.], [2.]]))
        self.assertAlmostEqual(errors["rmse"], (2.5)**0.5)
        self.assertEqual(errors["mae"], 1.5)
        self.assertEqual(errors["max_abs_error"], 2)
        self.assertEqual(errors["relative_l2"], 1)
        self.assertIsNone(solution_errors(torch.ones(2, 1), torch.zeros(2, 1))["relative_l2"])
        for prediction in (torch.zeros(2, 2), torch.tensor([[float("nan")], [0.]])):
            with self.assertRaises(ValueError):
                solution_errors(prediction, torch.zeros(2, 1))

    def test_shared_samples_isolation_and_repeatability(self):
        builds, observed = [], []
        benchmark = case()
        def make_problem():
            builds.append(True)
            return dp.PINNProblem(lambda x, u: u, lambda: torch.rand(4, 1), [
                dp.PointConstraint("boundary", torch.zeros(2, 1), lambda x: x[:, 0])])
        benchmark.build_problem = make_problem
        def factory(problem):
            observed.append(problem.collocation_points.clone())
            problem.collocation_points.fill_(999)  # Must not affect the next recipe.
            return ExactRecipe()
        recipes = [Recipe("first", factory), Recipe("second", factory)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = run_benchmark(benchmark, recipes, seeds=[7, 8], iterations=2,
                                    output_dir=root / "run")
            self.assertEqual(len(builds), 2)
            self.assertTrue(torch.equal(observed[0], observed[1]))
            self.assertTrue(torch.equal(observed[2], observed[3]))
            self.assertFalse(torch.equal(observed[0], observed[2]))
            self.assertEqual(len(results), 4)
            aggregate = json.loads((root / "run/aggregate.json").read_text())[0]
            self.assertEqual(aggregate["runs_completed"], 2)
            self.assertEqual(aggregate["rmse_mean"], 0)
            self.assertEqual(aggregate["rmse_std"], 0)
            self.assertTrue(all(r["status"] == "completed" and r["rmse"] == 0 for r in results))
            self.assertEqual(results[0]["samples_sha256"], results[1]["samples_sha256"])
            with np.load(root / "run/seed-7/samples.npz") as samples:
                np.testing.assert_array_equal(samples["collocation_points"], observed[0].numpy())
                self.assertEqual(samples["constraint_0_target"].shape, (2,))
            with np.load(root / "run/seed-7/first/predictions.npz") as saved:
                self.assertEqual(saved["prediction"].shape, saved["reference"].shape)
            with (root / "run/seed-7/first/loss_history.csv").open() as stream:
                self.assertEqual([r["iteration"] for r in csv.DictReader(stream)], ["1", "2"])
            rerun = run_benchmark(benchmark, recipes, seeds=[7], iterations=2,
                                  output_dir=root / "rerun")
            self.assertEqual(results[0]["samples_sha256"], rerun[0]["samples_sha256"])
            self.assertEqual(json.loads((root / "run/manifest.json").read_text())["seeds"], [7, 8])
            self.assertEqual(len(json.loads((root / "run/summary.json").read_text())), 4)
            with self.assertRaises(FileExistsError):
                run_benchmark(benchmark, recipes, seeds=[7], iterations=1, output_dir=root / "run")
            self.assertEqual(len(json.loads((root / "run/summary.json").read_text())), 4)

    def test_failures_are_saved_and_raised(self):
        class Broken(ExactRecipe):
            def train(self, iterations, display_every=0):
                raise RuntimeError("training failed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "failure"
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                run_benchmark(case(), [Recipe("broken", lambda p: Broken())], seeds=[7],
                              iterations=2, output_dir=root)
            result = json.loads((root / "seed-7/broken/result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("rmse", result)
            self.assertEqual(json.loads((root / "summary.json").read_text())[0]["status"], "failed")

    def test_sampling_failure_and_zero_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = case()
            benchmark.reference = torch.zeros(3, 2)  # Wrong row count.
            with self.assertRaises(ValueError):
                run_benchmark(benchmark, [Recipe("exact", lambda p: ExactRecipe())],
                              seeds=[7], iterations=1, output_dir=root / "bad")
            failure = json.loads((root / "bad/failure.json").read_text())
            self.assertEqual(failure["stage"], "sampling")
            results = run_benchmark(case(), [Recipe("exact", lambda p: ExactRecipe())],
                                    seeds=[7], iterations=0, output_dir=root / "zero")
            self.assertIsNone(results[0]["final_training_loss"])
            self.assertEqual((root / "zero/seed-7/exact/loss_history.csv").read_text().strip(),
                             "iteration,total")

    def test_configuration_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "invalid"
            for seeds in ([], [1, 1], [-1], [True]):
                with self.assertRaises(ValueError):
                    run_benchmark(case(), [Recipe("exact", lambda p: ExactRecipe())],
                                  seeds=seeds, iterations=1, output_dir=root)
            with self.assertRaises(ValueError):
                run_benchmark(case(), [Recipe("../escape", lambda p: ExactRecipe())],
                              seeds=[1], iterations=1, output_dir=root)
            self.assertFalse(root.exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
    def test_real_gpu_recipe_artifacts(self):
        from examples.benchmark_poisson import build_problem
        benchmark = BenchmarkCase("poisson", build_problem,
                                  torch.linspace(0, 1, 9)[:, None], lambda x: x*(1-x))
        def build(problem):
            net = dp.nn.FullyConnected([1, 8, 1], "tanh", "Xavier normal")
            return dp.PINN(problem, net).compile(device="cuda")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "gpu"
            result = run_benchmark(benchmark, [Recipe("baseline", build)], seeds=[7],
                                   iterations=3, output_dir=root)[0]
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["device"].startswith("cuda"))
            self.assertGreater(result["training_seconds"], 0)
            with np.load(root / "seed-7/baseline/predictions.npz") as predictions:
                errors = solution_errors(predictions["prediction"], predictions["reference"])
            self.assertEqual(errors["rmse"], result["rmse"])


if __name__ == "__main__":
    unittest.main()
