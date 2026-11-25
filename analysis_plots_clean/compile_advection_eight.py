"""Compile the official eight-beta advection comparison from stored metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from burgers_common import load_split, relative_l2
from run_advection_beta_queue import fixed_transport_prediction


CROSS = ROOT / "experiments" / "outputs_cross_pde"
PARAMS = CROSS / "advection_parameters"
OUTPUT = Path(__file__).resolve().parent / "advection_eight_parameter_summary.json"

CONFIGS = {
    0.1: {
        "prefix": "advection_beta01",
        "data": "pdebench_advection_beta01_official",
        "fno": [
            PARAMS / "advection_beta01" / f"fno_m25_w68_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
        "deeponet": [
            PARAMS / "advection_beta01" / f"deeponet_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
    0.2: {
        "prefix": "advection_beta02",
        "data": "pdebench_advection_beta02_official",
        "fno": [
            PARAMS / "advection_beta02" / f"fno_m25_w68_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
        "deeponet": [
            PARAMS / "advection_beta02" / f"deeponet_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
    0.4: {
        "prefix": "advection_beta04",
        "data": "pdebench_advection_beta04_official",
        "fno": [
            PARAMS / "advection_beta04" / f"fno_m25_w68_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
        "deeponet": [
            PARAMS / "advection_beta04" / f"deeponet_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
    0.7: {
        "prefix": "advection_beta07",
        "data": "pdebench_advection_beta07_official",
        "fno": [
            PARAMS / "beta07" / "fno_m25_w68_s7" / "metrics.json",
            PARAMS / "advection_beta07" / "fno_m25_w68_s11" / "metrics.json",
            PARAMS / "advection_beta07" / "fno_m25_w68_s23" / "metrics.json",
        ],
        "deeponet": [
            PARAMS / "beta07" / "deeponet_s7" / "metrics.json",
            PARAMS / "advection_beta07" / "deeponet_s11" / "metrics.json",
            PARAMS / "advection_beta07" / "deeponet_s23" / "metrics.json",
        ],
    },
    1.0: {
        "prefix": "advection_beta1",
        "data": "pdebench_advection_beta1_official",
        "fno": [
            CROSS / "fno_pm503" / "advection_beta1" / "s7" / "metrics.json",
            CROSS / "fno_pm503" / "advection_beta1" / "s11" / "metrics.json",
            CROSS / "fno_pm503" / "advection_beta1" / "s23" / "metrics.json",
        ],
        "deeponet": [
            CROSS / "full" / "advection_beta1" / "deeponet" / "metrics.json",
            CROSS / "seeds" / "advection_beta1" / "deeponet_s11" / "metrics.json",
            CROSS / "seeds" / "advection_beta1" / "deeponet_s23" / "metrics.json",
        ],
    },
    2.0: {
        "prefix": "advection_beta20",
        "data": "pdebench_advection_beta20_official",
        "fno": [
            PARAMS / "beta20" / "fno_m25_w68_s7" / "metrics.json",
            PARAMS / "advection_beta20" / "fno_m25_w68_s11" / "metrics.json",
            PARAMS / "advection_beta20" / "fno_m25_w68_s23" / "metrics.json",
        ],
        "deeponet": [
            PARAMS / "beta20" / "deeponet_s7" / "metrics.json",
            PARAMS / "advection_beta20" / "deeponet_s11" / "metrics.json",
            PARAMS / "advection_beta20" / "deeponet_s23" / "metrics.json",
        ],
    },
    4.0: {
        "prefix": "advection_beta40",
        "data": "pdebench_advection_beta40_official",
        "fno": [
            PARAMS / "advection_beta40" / f"fno_m25_w68_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
        "deeponet": [
            PARAMS / "advection_beta40" / f"deeponet_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
    7.0: {
        "prefix": "advection_beta70",
        "data": "pdebench_advection_beta70_official",
        "fno": [
            PARAMS / "advection_beta70" / f"fno_m25_w68_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
        "deeponet": [
            PARAMS / "advection_beta70" / f"deeponet_s{seed}" / "metrics.json"
            for seed in (7, 11, 23)
        ],
    },
}


def read_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(paths: list[Path]) -> dict:
    metrics = [read_metrics(path) for path in paths]
    values = [float(item["test_relative_l2"]) for item in metrics]
    parameters = {int(item["parameter_count"]) for item in metrics}
    if len(parameters) != 1:
        raise ValueError("inconsistent parameter counts")
    return {
        "seeds": [int(item["config"]["seed"]) for item in metrics],
        "test_values": values,
        "test_mean": fmean(values),
        "test_std": stdev(values),
        "parameters": parameters.pop(),
    }


def main() -> None:
    summary = {}
    for beta, config in CONFIGS.items():
        target = load_split(
            ROOT
            / "datasets"
            / config["data"]
            / f"{config['prefix']}_test.parquet",
            1000,
        )[..., 0]
        prediction = fixed_transport_prediction(target[:, :, 0], beta)
        entry = {
            "fixed_transport": relative_l2(prediction, target),
            "fixed_branch_parameters": 0,
            "parameter_budget": 501435,
            "fno": summarize(config["fno"]),
            "deeponet": summarize(config["deeponet"]),
        }
        summary[f"beta{beta:g}"] = entry
        print(
            f"beta={beta:g} fixed={entry['fixed_transport']:.8f} "
            f"fno={entry['fno']['test_mean']:.8f} "
            f"deeponet={entry['deeponet']['test_mean']:.8f}",
            flush=True,
        )
    OUTPUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
