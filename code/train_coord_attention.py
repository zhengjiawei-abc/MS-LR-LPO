"""训练坐标查询交叉注意力算子（code/coord_attention_operator.py）。

损失只有数据 MSE：每步随机抽 batch_size 条轨迹，每条随机抽 points 个网格点。
AdamW + warmup + 余弦退火 + 梯度裁剪；每 eval_every 步在全部 dev 样本上整场评估，
按 dev 相对 L2 保留最佳权重；最后在全部 test 样本上整场评估。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from burgers_common import (NT, NX, configure_backend, count_parameters, grid_coordinates,
                            load_split, relative_l2, set_seed)
from coord_attention_operator import CoordAttentionOperator, OperatorConfig

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "pdebench_burgers"


@torch.no_grad()
def predict_full(model, fields: torch.Tensor, device, batch: int = 8) -> torch.Tensor:
    """fields [N, nx, nt, 1] -> 整场预测 [N, nx, nt]。"""
    model.eval()
    x, t = grid_coordinates(device)
    out = []
    for s in range(0, fields.size(0), batch):
        u0 = fields[s:s + batch, :, 0, 0]
        kv = model.context(u0)
        b = u0.size(0)
        out.append(model.decode(kv, x.expand(b, -1, -1), t.expand(b, -1, -1)).view(b, NX, NT))
    model.train()
    return torch.cat(out)


def evaluate(model, fields, device) -> float:
    return relative_l2(predict_full(model, fields, device), fields[..., 0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-samples", type=int, default=512)
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--points", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--quick", action="store_true")
    for name, field in OperatorConfig.__dataclass_fields__.items():
        p.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = p.parse_args()
    if args.quick:
        args.train_samples, args.steps, args.eval_every = 32, 60, 30

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_backend(device)
    train = load_split(DATA_DIR / "burgers_nu0.01_train_000.parquet", args.train_samples).to(device)
    dev = load_split(DATA_DIR / "burgers_nu0.01_dev.parquet", 10**6).to(device)
    test = load_split(DATA_DIR / "burgers_nu0.01_test.parquet", 10**6).to(device)
    if args.quick:
        dev, test = dev[:16], test[:16]
    print(f"train {tuple(train.shape)} dev {tuple(dev.shape)} test {tuple(test.shape)}", flush=True)

    cfg = OperatorConfig(**{k: getattr(args, k) for k in OperatorConfig.__dataclass_fields__})
    model = CoordAttentionOperator(cfg).to(device)
    print(f"params {count_parameters(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    x_axis, t_axis = grid_coordinates(device)
    x_axis, t_axis = x_axis.view(NX, NT), t_axis.view(NX, NT)
    best, best_step, best_state, history = float("inf"), -1, None, []
    started = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, train.size(0), (args.batch_size,), device=device)
        xi = torch.randint(0, NX, (args.batch_size, args.points), device=device)
        ti = torch.randint(0, NT, (args.batch_size, args.points), device=device)
        fields = train[idx, :, :, 0]
        target = fields[torch.arange(args.batch_size, device=device)[:, None], xi, ti]
        pred = model(fields[:, :, 0], x_axis[xi, ti].unsqueeze(-1), t_axis[xi, ti].unsqueeze(-1))
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            dev_l2 = evaluate(model, dev, device)
            history.append({"step": step + 1, "loss": loss.item(), "dev_l2": dev_l2})
            if dev_l2 < best:
                best, best_step, best_state = dev_l2, step + 1, copy.deepcopy(model.state_dict())
            print(f"step {step + 1:6d} loss {loss.item():.3e} dev_l2 {dev_l2:.4f} best {best:.4f}@{best_step}", flush=True)

    model.load_state_dict(best_state)
    res = {"train_l2": evaluate(model, train[:256], device), "dev_l2": best,
           "test_l2": evaluate(model, test, device)}
    print(f"best@{best_step}: train(256) {res['train_l2']:.4f} dev {res['dev_l2']:.4f} "
          f"test(all {test.size(0)}) {res['test_l2']:.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "model": "coord_attention", "config": run_cfg, "operator": asdict(cfg),
        "parameter_count": count_parameters(model), "best_step": best_step,
        "train_relative_l2": res["train_l2"], "dev_relative_l2": res["dev_l2"],
        "test_relative_l2": res["test_l2"], "test_samples": test.size(0),
        "training_seconds": time.time() - started, "history": history}, indent=2), encoding="utf-8")
    if args.save_model:
        torch.save({"state_dict": best_state, "operator": asdict(cfg)}, args.output_dir / "model.pt")


if __name__ == "__main__":
    main()
