import searchGPU
import os
from loguru import logger

gpusindx = searchGPU.searchGPU()
logger.info(f'Using GPU: {gpusindx}.')

os.chdir(os.path.dirname(__file__))

import argparse
import csv
import math
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.stats import spearmanr, ttest_rel, rankdata
from transformers import AutoConfig, AutoModel, AutoTokenizer

DEFAULT_DATA = "/H7/mypj/mrcse/SynCSE/SentEval/data/downstream/STS/STSBenchmark/sts-test.csv"
DEFAULT_STS_ROOT = "/H7/mypj/mrcse/SynCSE/SentEval/data/downstream/STS"
TASK_DIR_MAP: Dict[str, str] = {
    "STS12": "STS12-en-test",
    "STS13": "STS13-en-test",
    "STS14": "STS14-en-test",
    "STS15": "STS15-en-test",
    "STS16": "STS16-en-test",
}

def read_stsbenchmark(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    sent1, sent2, labels = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 7:
                continue
            score = float(row[4])
            s1 = row[5].strip()
            s2 = row[6].strip()
            if not s1 or not s2:
                continue
            sent1.append(s1)
            sent2.append(s2)
            labels.append(score)
    if not labels:
        raise ValueError(f"未从数据集中读到有效样本: {path}")
    return sent1, sent2, np.array(labels, dtype=np.float64)


def read_sts_year(sts_root: str, task_name: str) -> Tuple[List[str], List[str], np.ndarray]:
    if task_name not in TASK_DIR_MAP:
        raise ValueError(f"不支持的任务: {task_name}")

    year_dir = Path(sts_root) / TASK_DIR_MAP[task_name]
    if not year_dir.exists():
        raise FileNotFoundError(f"任务目录不存在: {year_dir}")

    sent1_all: List[str] = []
    sent2_all: List[str] = []
    labels_all: List[float] = []

    for gs_path in sorted(year_dir.glob("STS.gs.*.txt")):
        sub = gs_path.name.replace("STS.gs.", "").replace(".txt", "")
        input_path = year_dir / f"STS.input.{sub}.txt"
        if not input_path.exists():
            continue

        gs_lines = gs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        in_lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(gs_lines) != len(in_lines):
            limit = min(len(gs_lines), len(in_lines))
            gs_lines = gs_lines[:limit]
            in_lines = in_lines[:limit]

        for gold_line, input_line in zip(gs_lines, in_lines):
            gold_text = gold_line.strip()
            if gold_text == "":
                continue

            parts = input_line.split("\t", 1)
            if len(parts) < 2:
                continue

            s1 = parts[0].strip()
            s2 = parts[1].strip()
            if not s1 or not s2:
                continue

            try:
                score = float(gold_text)
            except ValueError:
                continue

            sent1_all.append(s1)
            sent2_all.append(s2)
            labels_all.append(score)

    if not labels_all:
        raise ValueError(f"未从任务 {task_name} 读到有效样本: {year_dir}")

    return sent1_all, sent2_all, np.array(labels_all, dtype=np.float64)


def read_task_data(task_name: str, sts_root: str, benchmark_path: str) -> Tuple[List[str], List[str], np.ndarray]:
    if task_name == "STSBenchmark":
        return read_stsbenchmark(benchmark_path)
    return read_sts_year(sts_root, task_name)

def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.norm(x, p=2, dim=1, keepdim=True), min=1e-12)

def encode_sentences(
    model,
    tokenizer,
    sentences: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooler: str,
    use_cl_forward: bool,
) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_text = sentences[i : i + batch_size]
            batch = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_cl_forward:
                result = model(
                    **batch,
                    output_hidden_states=pooler in ["avg_top2", "avg_first_last"],
                    return_dict=True,
                    sent_emb=True,
                )
            else:
                result = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = result.last_hidden_state

            if use_cl_forward:
                pooled = result.pooler_output
            elif pooler == "cls":
                pooled = result.pooler_output
                if pooled is None:
                    pooled = last_hidden[:, 0]
            elif pooler == "cls_before_pooler":
                pooled = last_hidden[:, 0]
            elif pooler == "avg":
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(1) / mask.sum(1)
            elif pooler == "avg_first_last":
                first_hidden = result.hidden_states[1]
                final_hidden = result.hidden_states[-1]
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = ((first_hidden + final_hidden) / 2.0 * mask).sum(1) / mask.sum(1)
            elif pooler == "avg_top2":
                second_last = result.hidden_states[-2]
                final_hidden = result.hidden_states[-1]
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = ((second_last + final_hidden) / 2.0 * mask).sum(1) / mask.sum(1)
            else:
                raise ValueError(f"不支持的pooler: {pooler}")

            outputs.append(pooled.detach().cpu())

    return torch.cat(outputs, dim=0)

def predict_scores(
    model_path: str,
    sent1: List[str],
    sent2: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooler: str,
) -> np.ndarray:
    model_path_obj = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    use_cl_forward = False

    pt_path = model_path_obj / "pytorch_model.bin"
    has_cl_mlp = False
    if pt_path.exists():
        try:
            state_dict = torch.load(str(pt_path), map_location="cpu")
            has_cl_mlp = ("mlp.dense.weight" in state_dict) and ("mlp.dense.bias" in state_dict)
        except Exception:
            has_cl_mlp = False

    if has_cl_mlp:
        try:
            from simcse.models import RobertaForCL, BertForCL

            config = AutoConfig.from_pretrained(model_path)
            model_args = SimpleNamespace(
                pooler_type=pooler,
                temp=0.05,
                do_mlm=False,
                mlp_only_train=False,
            )
            if config.model_type == "roberta":
                model = RobertaForCL.from_pretrained(model_path, config=config, model_args=model_args)
            elif config.model_type == "bert":
                model = BertForCL.from_pretrained(model_path, config=config, model_args=model_args)
            else:
                model = AutoModel.from_pretrained(model_path)
            use_cl_forward = config.model_type in ["roberta", "bert"]
            print(f"[info] 使用 CL 模型推理路径: {model.__class__.__name__} (has mlp head)")
        except Exception as exc:
            print(f"[warn] 加载 CL 模型失败，回退 AutoModel。原因: {exc}")
            model = AutoModel.from_pretrained(model_path)
            use_cl_forward = False
    else:
        model = AutoModel.from_pretrained(model_path)
        use_cl_forward = False

    model = model.to(device)

    emb1 = encode_sentences(model, tokenizer, sent1, device, batch_size, max_length, pooler, use_cl_forward)
    emb2 = encode_sentences(model, tokenizer, sent2, device, batch_size, max_length, pooler, use_cl_forward)

    emb1 = l2_normalize(emb1)
    emb2 = l2_normalize(emb2)
    cosine = (emb1 * emb2).sum(dim=1).numpy()  # [-1, 1]

    pred = (cosine + 1) /2
    pred = 5 * (pred ** 2) 
    pred = np.clip(pred, 0.0, 5.0)
    return pred.astype(np.float64)

def bootstrap_spearman_test(
    gold: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    alternative: str = "greater"  # "greater" (pred1>pred2), "less", "two-sided"
) -> dict:
    """
    基于Bootstrap的斯皮尔曼系数显著性检验
    :param gold: 真实标签
    :param pred1: 模型1预测值（ours）
    :param pred2: 模型2预测值（simcse）
    :param n_bootstrap: 抽样次数（越多越准，建议≥1000）
    :param alpha: 显著性水平
    :param alternative: 检验方向
    :return: 检验结果字典
    """
    n = len(gold)
    delta_dist = []  # 存储每次bootstrap的spearman差值 (rho1 - rho2)
    
    # 重复抽样
    for _ in range(n_bootstrap):
        # 有放回抽样
        idx = np.random.choice(n, size=n, replace=True)
        gold_boot = gold[idx]
        pred1_boot = pred1[idx]
        pred2_boot = pred2[idx]
        
        # 计算本次抽样的斯皮尔曼系数
        rho1 = spearmanr(gold_boot, pred1_boot).correlation
        rho2 = spearmanr(gold_boot, pred2_boot).correlation
        
        # 存储差值
        delta_dist.append(rho1 - rho2)
    
    delta_dist = np.array(delta_dist)
    delta_mean = np.mean(delta_dist)  # 平均差值（ours - simcse）
    
    # 计算p值
    if alternative == "greater":
        # H1: delta > 0 (ours的spearman更大)
        p_value = np.sum(delta_dist <= 0) / n_bootstrap
    elif alternative == "less":
        # H1: delta < 0 (simcse的spearman更大)
        p_value = np.sum(delta_dist >= 0) / n_bootstrap
    else:
        # 双侧检验
        p_value = 2 * min(np.sum(delta_dist <= 0) / n_bootstrap, np.sum(delta_dist >= 0) / n_bootstrap)
    
    # 计算95%置信区间
    ci_lower = np.percentile(delta_dist, 100 * alpha / 2)
    ci_upper = np.percentile(delta_dist, 100 * (1 - alpha / 2))
    
    return {
        "delta_mean": delta_mean,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_dist": delta_dist
    }


def bootstrap_macro_spearman_test(
    task_cache: List[Dict[str, np.ndarray]],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    alternative: str = "greater",  # H1: ours_macro > sim_macro
) -> dict:
    """
    对“跨任务平均 Spearman（macro average）”做 bootstrap 显著性检验。
    每次bootstrap在每个任务内部有放回抽样，计算该任务spearman后再做跨任务平均。
    """
    if len(task_cache) == 0:
        raise ValueError("task_cache 为空，无法做平均显著性检验")

    delta_dist = []
    ours_macro_dist = []
    sim_macro_dist = []

    for _ in range(n_bootstrap):
        ours_task_sps = []
        sim_task_sps = []

        for item in task_cache:
            gold = item["gold"]
            ours_pred = item["ours_pred"]
            sim_pred = item["sim_pred"]

            n = len(gold)
            idx = np.random.choice(n, size=n, replace=True)

            rho_ours = spearmanr(gold[idx], ours_pred[idx]).correlation
            rho_sim = spearmanr(gold[idx], sim_pred[idx]).correlation

            ours_task_sps.append(rho_ours)
            sim_task_sps.append(rho_sim)

        ours_macro = float(np.mean(ours_task_sps))
        sim_macro = float(np.mean(sim_task_sps))
        delta = ours_macro - sim_macro

        ours_macro_dist.append(ours_macro)
        sim_macro_dist.append(sim_macro)
        delta_dist.append(delta)

    delta_dist = np.array(delta_dist)
    ours_macro_dist = np.array(ours_macro_dist)
    sim_macro_dist = np.array(sim_macro_dist)

    if alternative == "greater":
        p_value = np.sum(delta_dist <= 0) / n_bootstrap
    elif alternative == "less":
        p_value = np.sum(delta_dist >= 0) / n_bootstrap
    else:
        p_value = 2 * min(np.sum(delta_dist <= 0) / n_bootstrap, np.sum(delta_dist >= 0) / n_bootstrap)

    ci_lower = np.percentile(delta_dist, 100 * alpha / 2)
    ci_upper = np.percentile(delta_dist, 100 * (1 - alpha / 2))

    return {
        "ours_macro_mean": float(np.mean(ours_macro_dist)),
        "sim_macro_mean": float(np.mean(sim_macro_dist)),
        "delta_mean": float(np.mean(delta_dist)),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_dist": delta_dist,
    }

def rank_ttest(gold: np.ndarray, ours_pred: np.ndarray, sim_pred: np.ndarray, alternative: str = "greater"):
    """
    对模型预测值的秩次做配对t检验（反映排序能力的差异）
    :param gold: 真实标签
    :param ours_pred: 我方预测值
    :param sim_pred: SimCSE预测值
    :param alternative: 检验方向
    :return: t值和p值
    """
    # 计算预测值的秩次（rank越大，分数越高）
    ours_rank = rankdata(ours_pred)
    sim_rank = rankdata(sim_pred)
    gold_rank = rankdata(gold)
    
    # 计算秩次与真实秩次的差值（越小说明排序越准）
    ours_rank_diff = np.abs(ours_rank - gold_rank)
    sim_rank_diff = np.abs(sim_rank - gold_rank)
    
    # 配对t检验：检验 ours_rank_diff < sim_rank_diff (ours排序更准)
    t_stat, p_val = ttest_rel(sim_rank_diff, ours_rank_diff, alternative=alternative)
    return t_stat, p_val, ours_rank_diff, sim_rank_diff

def print_debug_rows(
    sent1: List[str],
    sent2: List[str],
    gold: np.ndarray,
    ours_pred: np.ndarray,
    sim_pred: np.ndarray,
    ours_err: np.ndarray,
    sim_err: np.ndarray,
    diff: np.ndarray,
    debug_rows: int,
):
    n = len(diff)
    if debug_rows <= 0 or n == 0:
        return

    pos = int(np.sum(diff > 0))
    neg = int(np.sum(diff < 0))
    zero = n - pos - neg

    print("\n===== 调试信息（逐样本）=====")
    print(f"diff = sim_err - ours_err")
    print(f"diff > 0 (ours更好): {pos}/{n} ({pos / n:.2%})")
    print(f"diff < 0 (simcse更好): {neg}/{n} ({neg / n:.2%})")
    print(f"diff = 0: {zero}/{n} ({zero / n:.2%})")
    print(
        "diff分位数: "
        f"min={np.min(diff):.4f}, "
        f"p25={np.percentile(diff, 25):.4f}, "
        f"p50={np.percentile(diff, 50):.4f}, "
        f"p75={np.percentile(diff, 75):.4f}, "
        f"max={np.max(diff):.4f}"
    )

    def _print_one(idx: int, title: str):
        print(f"\n[{title}] idx={idx}")
        print(f"gold={gold[idx]:.4f}, ours_pred={ours_pred[idx]:.4f}, sim_pred={sim_pred[idx]:.4f}")
        print(f"ours_err={ours_err[idx]:.4f}, sim_err={sim_err[idx]:.4f}, diff={diff[idx]:.4f}")
        print(f"s1={sent1[idx]}")
        print(f"s2={sent2[idx]}")

    print(f"\n--- 前 {min(debug_rows, n)} 条样本 ---")
    for idx in range(min(debug_rows, n)):
        _print_one(idx, "head")

    k = min(debug_rows, n)
    best_for_ours_idx = np.argsort(diff)[-k:][::-1]
    worst_for_ours_idx = np.argsort(diff)[:k]

    print(f"\n--- 对 ours 最有利的 {k} 条（diff 最大）---")
    for idx in best_for_ours_idx:
        _print_one(int(idx), "best_for_ours")

    print(f"\n--- 对 ours 最不利的 {k} 条（diff 最小）---")
    for idx in worst_for_ours_idx:
        _print_one(int(idx), "worst_for_ours")

def main():
    parser = argparse.ArgumentParser(
        description="在 STSBenchmark/STS12-STS16 上跑两个模型，并对斯皮尔曼系数做显著性检验"
    )
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA, help="sts-test.csv 路径")
    parser.add_argument("--sts-root", type=str, default=DEFAULT_STS_ROOT, help="STS目录路径（包含STS12-16子目录）")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["STSBenchmark", "STS12", "STS13", "STS14", "STS15", "STS16"],
        choices=["STSBenchmark", "STS12", "STS13", "STS14", "STS15", "STS16"],
        help="要评估的任务列表",
    )
    parser.add_argument("--ours-model", type=str, default="/H7/mypj/mrcse/SynCSE/result/rankingcse_1.6_roberta-base_SumCSE_512batch_3epoch", help="我方模型路径")
    parser.add_argument("--simcse-model", type=str, default="/H7/mypj/mrcse/SynCSE/result/Sumcse_roberta-base", help="SimCSE模型路径")
    parser.add_argument(
        "--pooler",
        type=str,
        default="cls",
        choices=["cls", "cls_before_pooler", "avg", "avg_first_last", "avg_top2"],
        help="句向量池化方式",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--metric",
        type=str,
        default="abs",
        choices=["abs", "sq"],
        help="逐样本误差类型：abs=绝对误差, sq=平方误差",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )
    parser.add_argument(
        "--debug-rows",
        type=int,
        default=0,
        help="打印逐样本调试信息（建议 3~10），默认0不打印",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap抽样次数（斯皮尔曼检验用）",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if "STSBenchmark" in args.tasks and not data_path.exists():
        raise FileNotFoundError(f"数据集不存在: {data_path}")

    sts_root = Path(args.sts_root)
    if any(task != "STSBenchmark" for task in args.tasks) and not sts_root.exists():
        raise FileNotFoundError(f"STS目录不存在: {sts_root}")

    device = torch.device(args.device)
    print(f"设备: {device}")
    print(f"pooler: {args.pooler}")
    print(f"任务: {', '.join(args.tasks)}")

    summary_rows = []
    task_eval_cache: List[Dict[str, np.ndarray]] = []

    for task_name in args.tasks:
        sent1, sent2, gold = read_task_data(task_name, str(sts_root), str(data_path))

        print(f"\n================ {task_name} ================")
        print(f"样本数: {len(gold)}")
        print("[1/2] 正在运行 ours 模型...")
        ours_pred = predict_scores(
            args.ours_model,
            sent1,
            sent2,
            device,
            args.batch_size,
            args.max_length,
            args.pooler,
        )

        print("[2/2] 正在运行 SimCSE 模型...")
        sim_pred = predict_scores(
            args.simcse_model,
            sent1,
            sent2,
            device,
            args.batch_size,
            args.max_length,
            args.pooler,
        )

        ours_sp = spearmanr(ours_pred, gold).correlation
        sim_sp = spearmanr(sim_pred, gold).correlation

        if args.metric == "abs":
            ours_err = np.abs(ours_pred - gold)
            sim_err = np.abs(sim_pred - gold)
        else:
            ours_err = (ours_pred - gold) ** 2
            sim_err = (sim_pred - gold) ** 2

        diff = sim_err - ours_err
        t_two, p_two = ttest_rel(sim_err, ours_err)
        t_greater, p_greater = ttest_rel(sim_err, ours_err, alternative="greater")
        t_less, p_less = ttest_rel(sim_err, ours_err, alternative="less")

        bootstrap_result = bootstrap_spearman_test(
            gold=gold,
            pred1=ours_pred,
            pred2=sim_pred,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
            alternative="greater",
        )

        rank_t, rank_p, _, _ = rank_ttest(
            gold=gold,
            ours_pred=ours_pred,
            sim_pred=sim_pred,
            alternative="greater",
        )

        print("\n===== 核心指标（排序能力）=====")
        print(f"ours Spearman:   {ours_sp * 100:.4f}")
        print(f"simcse Spearman: {sim_sp * 100:.4f}")
        print(f"Spearman差值 (ours - simcse): {(ours_sp - sim_sp) * 100:.4f}")

    # print("\n===== 辅助检验：逐样本误差的配对t检验 =====")
    # print("H0: mean(sim_err - ours_err) <= 0")
    # print("H1: mean(sim_err - ours_err) > 0  (ours 误差更小)")
    # print(f"error metric: {args.metric}")
    # print(f"mean(sim_err - ours_err): {np.mean(diff):.8f}")
    # print(f"t-statistic: {t_two:.8f}")
    # print(f"p-value(two-sided): {p_two:.3e}")
    # print(f"p-value(one-tailed, ours更优): {p_greater:.3e}")
    # print(f"p-value(one-tailed, simcse更优): {p_less:.3e}")

        print("\n===== 核心检验：斯皮尔曼系数的Bootstrap检验 =====")
        print("H0: ours Spearman <= simcse Spearman")
        print("H1: ours Spearman > simcse Spearman (排序能力更强)")
        print(f"Bootstrap抽样次数: {args.n_bootstrap}")
        print(f"Spearman差值均值 (ours - simcse): {bootstrap_result['delta_mean']:.6f}")
        print(f"p-value (one-tailed): {bootstrap_result['p_value']:.3e}")
        print(f"{(1-args.alpha)*100}% 置信区间: [{bootstrap_result['ci_lower']:.6f}, {bootstrap_result['ci_upper']:.6f}]")
        if bootstrap_result['p_value'] < args.alpha:
            print(f"结论: 通过显著性检验 (p < {args.alpha})，可认为 ours 的排序能力显著优于 SimCSE。")
        else:
            print(f"结论: 未通过排序能力的显著性检验 (p >= {args.alpha})。")

        print("\n===== 补充检验：秩次差的配对t检验 =====")
        print("H0: ours秩次误差 >= simcse秩次误差")
        print("H1: ours秩次误差 < simcse秩次误差 (排序更准)")
        print(f"t-statistic: {rank_t:.8f}")
        print(f"p-value (one-tailed): {rank_p:.3e}")
        if rank_p < args.alpha:
            print(f"结论: 通过显著性检验 (p < {args.alpha})，可认为 ours 的秩次排序显著更准。")
        else:
            print(f"结论: 未通过秩次排序的显著性检验 (p >= {args.alpha})。")

        summary_rows.append(
            {
                "task": task_name,
                "n": len(gold),
                "ours_sp": ours_sp,
                "sim_sp": sim_sp,
                "delta_sp": ours_sp - sim_sp,
                "bootstrap_p": bootstrap_result["p_value"],
                "rank_p": rank_p,
                "err_t_p_greater": p_greater,
                "err_t_p_less": p_less,
                "err_diff_mean": float(np.mean(diff)),
            }
        )

        task_eval_cache.append(
            {
                "task": task_name,
                "gold": gold,
                "ours_pred": ours_pred,
                "sim_pred": sim_pred,
            }
        )

        if args.debug_rows > 0 and task_name == args.tasks[0]:
            print_debug_rows(
                sent1,
                sent2,
                gold,
                ours_pred,
                sim_pred,
                ours_err,
                sim_err,
                diff,
                args.debug_rows,
            )

    print("\n================ 斯皮尔曼显著性一览 ================")
    print(f"判定规则: PASS 当且仅当 (ΔSpearman>0 且 bootstrap_p<{args.alpha})")
    print(
        f"{'Task':<14} {'N':>6} {'Ours':>9} {'SimCSE':>9} {'Delta':>9} {'p(one)':>12} {'Result':>10}"
    )
    print("-" * 76)

    pass_count = 0
    for row in summary_rows:
        delta = row['delta_sp']
        pval = row['bootstrap_p']
        is_pass = (delta > 0) and (pval < args.alpha)
        if is_pass:
            result = "PASS"
            pass_count += 1
        elif delta <= 0:
            result = "FAIL(方向反)"
        else:
            result = "FAIL(不显著)"

        print(
            f"{row['task']:<14} {row['n']:>6d} "
            f"{row['ours_sp']*100:>8.3f}% {row['sim_sp']*100:>8.3f}% "
            f"{row['delta_sp']*100:>+8.3f}% {row['bootstrap_p']:>12.3e} {result:>10}"
        )

    print("-" * 76)
    print(f"PASS {pass_count}/{len(summary_rows)} 个任务")

    macro_ours = float(np.mean([row["ours_sp"] for row in summary_rows]))
    macro_sim = float(np.mean([row["sim_sp"] for row in summary_rows]))
    macro_delta = macro_ours - macro_sim

    macro_bootstrap = bootstrap_macro_spearman_test(
        task_cache=task_eval_cache,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        alternative="greater",
    )

    macro_pass = (macro_delta > 0) and (macro_bootstrap["p_value"] < args.alpha)

    print("\n================ 平均Spearman显著性（跨任务） ================")
    print("定义: 对每个任务先算Spearman，再做任务级平均（macro average）")
    print(f"ours macro Spearman:   {macro_ours*100:.4f}%")
    print(f"simcse macro Spearman: {macro_sim*100:.4f}%")
    print(f"macro Delta (ours-sim): {macro_delta*100:+.4f}%")
    print(f"bootstrap p(one-tailed): {macro_bootstrap['p_value']:.3e}")
    print(f"{(1-args.alpha)*100:.1f}% CI of Delta: [{macro_bootstrap['ci_lower']*100:.4f}%, {macro_bootstrap['ci_upper']*100:.4f}%]")
    print(f"Result: {'PASS' if macro_pass else 'FAIL'}")


if __name__ == "__main__":
    main()