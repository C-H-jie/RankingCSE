import argparse
import json
import re
from typing import Dict, Set

import numpy as np
import pandas as pd


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> Set[str]:
	if pd.isna(text):
		return set()
	return set(TOKEN_PATTERN.findall(str(text).lower()))


def jaccard(a: str, b: str) -> float:
	tokens_a = tokenize(a)
	tokens_b = tokenize(b)
	union = tokens_a | tokens_b
	if not union:
		return 1.0
	return len(tokens_a & tokens_b) / len(union)


def summarize(series: pd.Series) -> Dict[str, float]:
	q = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
	return {
		"count": int(series.count()),
		"mean": float(series.mean()),
		"std": float(series.std()),
		"min": float(series.min()),
		"p25": float(q.loc[0.25]),
		"median": float(q.loc[0.5]),
		"p75": float(q.loc[0.75]),
		"p90": float(q.loc[0.9]),
		"p95": float(q.loc[0.95]),
		"p99": float(q.loc[0.99]),
		"max": float(series.max()),
	}


def sample_easy_scores(
	df: pd.DataFrame,
	sample_size: int,
	seed: int,
) -> pd.Series:
	rng = np.random.default_rng(seed)
	columns = ["sent0", "sent1", "hard_neg"]

	sentence_pool = []
	for row_idx, row in df[columns].iterrows():
		for col in columns:
			sentence_pool.append((row_idx, row[col]))

	easy_scores = []
	n_rows = len(df)
	for _ in range(sample_size):
		i = int(rng.integers(0, n_rows))
		s0 = df.iloc[i]["sent0"]
		while True:
			cand_row, cand_text = sentence_pool[int(rng.integers(0, len(sentence_pool)))]
			if cand_row != i:
				easy_scores.append(jaccard(s0, cand_text))
				break

	return pd.Series(easy_scores, dtype=float)


def compare_independent_distributions(
	a: pd.Series,
	b: pd.Series,
	num_pairs: int,
	seed: int,
) -> Dict[str, float]:
	rng = np.random.default_rng(seed)
	a_idx = rng.integers(0, len(a), size=num_pairs)
	b_idx = rng.integers(0, len(b), size=num_pairs)
	a_vals = a.to_numpy()[a_idx]
	b_vals = b.to_numpy()[b_idx]
	return {
		"gt_ratio": float((a_vals > b_vals).mean()),
		"eq_ratio": float((a_vals == b_vals).mean()),
		"lt_ratio": float((a_vals < b_vals).mean()),
	}


def compute(csv_path: str, easy_sample_size: int, seed: int) -> Dict[str, object]:
	df = pd.read_csv(csv_path, usecols=["sent0", "sent1", "hard_neg"])

	pos_scores = df.apply(lambda row: jaccard(row["sent0"], row["sent1"]), axis=1)
	hard_scores = df.apply(lambda row: jaccard(row["sent0"], row["hard_neg"]), axis=1)
	easy_scores = sample_easy_scores(df, sample_size=easy_sample_size, seed=seed)
	hard_easy_cmp = compare_independent_distributions(
		a=hard_scores,
		b=easy_scores,
		num_pairs=easy_sample_size,
		seed=seed + 1,
	)

	result = {
		"rows": int(len(df)),
		"pos": summarize(pos_scores),
		"hard": summarize(hard_scores),
		"easy_random_others": summarize(easy_scores),
		"easy_sample_size": int(easy_sample_size),
		"seed": int(seed),
		"pos_gt_hard_ratio": float((pos_scores > hard_scores).mean()),
		"pos_eq_hard_ratio": float((pos_scores == hard_scores).mean()),
		"pos_lt_hard_ratio": float((pos_scores < hard_scores).mean()),
		"hard_gt_easy_ratio": hard_easy_cmp["gt_ratio"],
		"hard_eq_easy_ratio": hard_easy_cmp["eq_ratio"],
		"hard_lt_easy_ratio": hard_easy_cmp["lt_ratio"],
		"easy_mean_minus_pos_mean": float(easy_scores.mean() - pos_scores.mean()),
		"easy_mean_minus_hard_mean": float(easy_scores.mean() - hard_scores.mean()),
		"easy_scores": easy_scores,
		"pos_scores": pos_scores,
		"hard_scores": hard_scores,
		"base_df": df,
	}
	return result


def print_result(result: Dict[str, object]) -> None:
	pos = result["pos"]
	hard = result["hard"]
	easy = result["easy_random_others"]
	print("=" * 68)
	print("Jaccard 系数统计（pos: sent0-sent1, hard: sent0-hard_neg）")
	print("=" * 68)
	print(f"样本数: {result['rows']}")
	print(f"easy随机采样数: {result['easy_sample_size']} (seed={result['seed']})")
	print()
	print("[均值 ± 标准差]")
	print(f"pos : {pos['mean']:.6f} ± {pos['std']:.6f}")
	print(f"hard: {hard['mean']:.6f} ± {hard['std']:.6f}")
	print(f"easy: {easy['mean']:.6f} ± {easy['std']:.6f}")
	print()
	print("[分位数]")
	print(f"{'指标':<10}{'pos':>14}{'hard':>14}{'easy':>14}")
	for key, name in [
		("min", "min"),
		("p25", "p25"),
		("median", "median"),
		("p75", "p75"),
		("p90", "p90"),
		("p95", "p95"),
		("p99", "p99"),
		("max", "max"),
	]:
		print(f"{name:<10}{pos[key]:>14.6f}{hard[key]:>14.6f}{easy[key]:>14.6f}")
	print()
	print("[逐条比较（pos vs hard）]")
	print(f"pos > hard : {result['pos_gt_hard_ratio'] * 100:.2f}%")
	print(f"pos = hard : {result['pos_eq_hard_ratio'] * 100:.2f}%")
	print(f"pos < hard : {result['pos_lt_hard_ratio'] * 100:.2f}%")
	print()
	print("[随机配对比较（hard vs easy）]")
	print(f"hard > easy : {result['hard_gt_easy_ratio'] * 100:.2f}%")
	print(f"hard = easy : {result['hard_eq_easy_ratio'] * 100:.2f}%")
	print(f"hard < easy : {result['hard_lt_easy_ratio'] * 100:.2f}%")
	print()
	print("[easy均值差]")
	print(f"easy_mean - pos_mean  = {result['easy_mean_minus_pos_mean']:.6f}")
	print(f"easy_mean - hard_mean = {result['easy_mean_minus_hard_mean']:.6f}")


def export_detail(result: Dict[str, object], output_path: str, keep_text: bool) -> None:
	df = result["base_df"].copy()
	df["pos_jaccard"] = result["pos_scores"]
	df["hard_jaccard"] = result["hard_scores"]
	if not keep_text:
		df = df[["pos_jaccard", "hard_jaccard"]]
	df.to_csv(output_path, index=False)
	print(f"\n已导出逐条 pos/hard Jaccard 系数到: {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Compute Jaccard stats for SumCSE triples.")
	parser.add_argument(
		"--csv",
		default="/H7/mypj/mrcse/SynCSE/data/SynCSE-partial-NLI.csv",
		help="Path to CSV file containing sent0, sent1, hard_neg columns.",
	)
	parser.add_argument(
		"--json",
		action="store_true",
		help="Print output as JSON.",
	)
	parser.add_argument(
		"--save-detail",
		default="",
		help="Optional CSV output path for per-row jaccard scores.",
	)
	parser.add_argument(
		"--keep-text",
		action="store_true",
		help="When using --save-detail, keep original sent0/sent1/hard_neg text columns.",
	)
	parser.add_argument(
		"--easy-sample-size",
		type=int,
		default=512,
		help="How many random easy pairs to sample from other rows.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for easy pair sampling.",
	)
	args = parser.parse_args()

	result = compute(args.csv, easy_sample_size=args.easy_sample_size, seed=args.seed)
	json_ready = {
		k: v
		for k, v in result.items()
		if k not in {"easy_scores", "pos_scores", "hard_scores", "base_df"}
	}
	if args.json:
		print(json.dumps(json_ready, ensure_ascii=False, indent=2))
	else:
		print_result(json_ready)

	if args.save_detail:
		export_detail(result, args.save_detail, args.keep_text)


if __name__ == "__main__":
	main()
