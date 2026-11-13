"""Audit and document a read-only copy of the 20261027v2 main analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
PROJECT = Path(r"E:\CodexPorjects\PUMC-rebuttal")
SOURCE = {
    "results/cacox_main": PROJECT / "results/cox_es_rcs_newdata_10x5/20261104_v1",
    "results/rcs_ridge_main": PROJECT / "results/rcs_ridge_newdata_10x5/20261104_v1",
}
DOCS = ARCHIVE / "docs"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def files_under(root: Path) -> dict[str, int]:
    out = {}
    for dirpath, _, names in os.walk(root):
        base = Path(dirpath)
        for name in names:
            p = base / name
            out[p.relative_to(root).as_posix()] = p.stat().st_size
    return out


def audit_copy() -> dict:
    result = {}
    for key, original in SOURCE.items():
        src, dest = files_under(original), files_under(ARCHIVE / key)
        if src != dest:
            missing = list(src.keys() - dest.keys())[:10]
            extra = list(dest.keys() - src.keys())[:10]
            different = [p for p in src.keys() & dest.keys() if src[p] != dest[p]][:10]
            raise RuntimeError(f"Copy mismatch {key}: missing={missing}, extra={extra}, size={different}")
        result[key] = {"files": len(src), "bytes": sum(src.values())}
    input_copy = ARCHIVE / "results/cacox_main/input/20261027v2.xlsx"
    expected = json.loads((ARCHIVE / "results/cacox_main/input/INPUT_PROVENANCE.json").read_text(encoding="utf-8"))["source_SHA256"]
    actual = digest(input_copy)
    if actual != expected:
        raise RuntimeError(f"Input workbook hash mismatch: {actual} != {expected}")
    result["workbook_sha256"] = actual
    return result


def fold_checks() -> dict:
    c = ARCHIVE / "results/cacox_main"
    r = ARCHIVE / "results/rcs_ridge_main"
    checks = Counter()
    for rep in range(1, 11):
        for fold in range(1, 6):
            key = f"repeat{rep:02d}_fold{fold:02d}"
            base = c / key
            required = [
                base / "prepared.rds",
                c / "bundles" / key / "main/COMPLETE.json",
                c / "i20" / f"repeat{rep:02d}" / "COMPLETE.json",
                c / "e_selected_sources" / f"repeat{rep:02d}" / "outer" / f"fold{fold:02d}" / "shared/COMPLETE.json",
                r / "folds" / key / "COMPLETE.json",
            ]
            for p in required:
                if not p.is_file():
                    raise RuntimeError(f"Missing fold artifact: {p}")
            for mi in range(1, 21):
                mi2 = f"{mi:02d}"
                required_mi = [
                    base / f"imputation_{mi2}/combined_selection.csv",
                    c / "bundles" / key / "main/outer" / f"development_{mi2}.csv",
                    c / "bundles" / key / "main/outer" / f"assessment_{mi2}.csv",
                    c / "i20" / f"repeat{rep:02d}" / "outer" / f"fold{fold:02d}" / f"mi{mi2}/COMPLETE.json",
                    r / "folds" / key / f"model_{mi2}.rds",
                    r / "folds" / key / f"prediction_{mi2}.csv",
                ]
                for p in required_mi:
                    if not p.is_file():
                        raise RuntimeError(f"Missing MI artifact: {p}")
                checks["imputation_units"] += 1
            for inner in range(1, 6):
                p = base / f"inner_{inner:02d}/imputation.rds"
                if not p.is_file():
                    raise RuntimeError(f"Missing inner MICE: {p}")
                checks["inner_mice_partitions"] += 1
            if not (base / "outer/imputation.rds").is_file():
                raise RuntimeError(f"Missing outer MICE: {base}")
            checks["outer_mice_partitions"] += 1
            checks["folds"] += 1
    for p in [c / "DATA_COMPLETE.json", c / "I20_COMPLETE.json", c / "E_SELECTED_COMPLETE.json", c / "comparison/COMPLETE.json", r / "FIT_COMPLETE.json"]:
        if not p.is_file():
            raise RuntimeError(f"Missing terminal marker: {p}")
    return dict(checks)


def metadata(rel: str) -> tuple[str, str, str, str]:
    parts = rel.split("/")
    method = "CACox" if rel.startswith("results/cacox_main/") else ("RCS + ridge-Cox" if rel.startswith("results/rcs_ridge_main/") else "code/docs")
    rep = fold = mi = ""
    for i, part in enumerate(parts):
        m = re.fullmatch(r"repeat(\d{2})_fold(\d{2})", part)
        if m:
            rep, fold = m.groups()
        elif re.fullmatch(r"repeat\d{2}", part):
            rep = part[6:]
        elif re.fullmatch(r"fold\d{2}", part):
            fold = part[4:]
        m = re.fullmatch(r"imputation_(\d{2})|mi(\d{2})", part)
        if m:
            mi = next(v for v in m.groups() if v)
    name = parts[-1]
    m = re.search(r"(?:model_|prediction_|assessment_|development_|imp)(\d{2})(?:\.|$)", name)
    if m and not mi:
        mi = m.group(1)
    if name == "imputation.rds":
        mi = "01-20 bundled"
    if not mi and rep and fold:
        mi = "shared/all"
    return method, rep, fold, mi


def write_fold_map() -> None:
    p = DOCS / "FOLD_IMPUTATION_MAP.md"
    with p.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 10 次重复 × 5 折 × 20 份插补的路径对照\n\n")
        f.write("每行是一份外层插补。`repeat01` 为第 1 次重复，`fold01` 为该次第 1 个外层折，`01` 为该折第 1 份插补。全部 1000 行均已检查文件存在。\n\n")
        f.write("| 重复 | 折 | 插补 | 筛选 | 开发集数据 | 验证集数据 | I20 单元 | CACox 折级权重/预测 | RCS 权重 | RCS 预测 |\n")
        f.write("|---:|---:|---:|---|---|---|---|---|---|---|\n")
        for rep in range(1, 11):
            for fold in range(1, 6):
                key = f"repeat{rep:02d}_fold{fold:02d}"
                for mi in range(1, 21):
                    n = f"{mi:02d}"
                    paths = [
                        f"results/cacox_main/{key}/imputation_{n}/combined_selection.csv",
                        f"results/cacox_main/bundles/{key}/main/outer/development_{n}.csv",
                        f"results/cacox_main/bundles/{key}/main/outer/assessment_{n}.csv",
                        f"results/cacox_main/i20/repeat{rep:02d}/outer/fold{fold:02d}/mi{n}/",
                        f"results/cacox_main/e_selected_sources/repeat{rep:02d}/outer/fold{fold:02d}/shared/",
                        f"results/rcs_ridge_main/folds/{key}/model_{n}.rds",
                        f"results/rcs_ridge_main/folds/{key}/prediction_{n}.csv",
                    ]
                    f.write(f"| {rep} | {fold} | {mi} | " + " | ".join(f"`{x}`" for x in paths) + " |\n")


def write_readme(audit: dict, checks: dict) -> None:
    readme = f"""# 20261027v2 主分析工程归档

本目录是原始已完成主分析的**只读副本**，没有重训、重插补或改写原有结果。模型显示名统一写作 **CACox**；源代码和原报告中历史名称 `CACox (ES)` 指同一方法。另一方法为 RCS + ridge-Cox。本归档不包含另行运行的五分支敏感性分析。

## 输入与完成范围

- 原始工作簿：`results/cacox_main/input/20261027v2.xlsx`；SHA-256：`{audit['workbook_sha256']}`。
- 输入溯源：`results/cacox_main/input/INPUT_PROVENANCE.json`。1140 名患者、29 个候选变量；其中 `Antibody1` 映射为建模字段 `Antibody`。
- 10 次重复 × 5 个外层折 = **{checks['folds']} 折**；每折 20 份插补，合计 **{checks['imputation_units']} 个外层插补编号**。每折还有 1 个外层和 5 个内部 MICE 分区，共 **{checks['outer_mice_partitions'] + checks['inner_mice_partitions']} 个分区**；每个 `imputation.rds` 保存同一分区的 20 链插补对象。
- CACox 与 RCS + ridge-Cox 各有 50 折、1000 个外层拟合/预测。完成依据：`results/cacox_main/DATA_COMPLETE.json`、`I20_COMPLETE.json`、`E_SELECTED_COMPLETE.json`、`comparison/COMPLETE.json` 和 `results/rcs_ridge_main/FIT_COMPLETE.json`。

## 目录

| 路径 | 内容 |
|---|---|
| `results/cacox_main/input/` | 工作簿、类型化输入、固定外层划分和配置 |
| `results/cacox_main/repeatXX_foldYY/` | 每折 `prepared.rds`、6 个 MICE 分区、20 份筛选和日志 |
| `results/cacox_main/bundles/repeatXX_foldYY/` | 开发集、验证集及每份插补的导出数据 |
| `results/cacox_main/i20/repeatXX/` | 内部选型、外层单元、权重、预测及审计 |
| `results/cacox_main/e_selected_sources/repeatXX/` | CACox 候选与最终折级权重、预测及审计；最终 `outer/foldYY/shared/` |
| `results/rcs_ridge_main/folds/repeatXX_foldYY/` | 每折调参、20 个 `model_NN.rds`、`prediction_NN.csv` 及完成标记 |
| `results/cacox_main/comparison/` | 两方法配对比较、汇总指标、图和原报告 |
| `code/experiments/` | 本次控制器及其所调用的 CACox/RCS 科学模块 |
| `code/upstream/` | 本次 R 拟合和数据处理调用的冻结上游源码 |
| `docs/FILE_INDEX.csv` | **逐文件**相对路径、大小、SHA-256、方法、重复、折、插补编号 |
| `docs/FOLD_IMPUTATION_MAP.md` | **1000 行**重复—折—插补—数据—权重—预测定位表 |
| `docs/COPY_AUDIT.json` | 源/副本相对路径及大小核对、完整性计数 |

## 如何定位

`repeat03_fold04` 表示第 3 次重复的第 4 个外层折；`imputation_07`、`mi07`、`model_07.rds`、`prediction_07.csv` 以及数据表的 `_07.csv` 均表示该折第 7 份插补。每个折的 `outer/imputation.rds` 与 `inner_01` 至 `inner_05/imputation.rds` 是各含 20 份的 R MICE 对象，不能把一个 `.rds` 当成一份插补。`shared/` 是 20 份汇合后的折级 CACox 权重/预测，不代表第 1 份插补。无折次编号的汇总文件覆盖全部折，详见逐文件索引。

## 可复核结果与边界

原配对比较报告位于 `results/cacox_main/comparison/FINAL_REPORT.md`。CACox 合并 C-index 为 0.74654（95% CI 0.71192–0.78117），RCS + ridge-Cox 为 0.73357（0.69825–0.76889），配对差为 0.01298（0.00495–0.02101）。这些是同一队列重复交叉验证，**不是外部独立验证**。归档保留原报告和所有原始文件；本页仅用于定位和审计，不替代统计说明。

原代码中有绝对工作目录及本机 R/Python/CUDA 依赖；归档保存完整调用源码与结果，但在另一台机器上重新执行仍需配置环境及路径。历史恢复日志、状态文件也按原样保留，以区分实际完成标记与中途重启记录。
"""
    (ARCHIVE / "README.md").write_text(readme, encoding="utf-8")


def write_index() -> tuple[int, int]:
    rows = 0
    nbytes = 0
    paths = [p for p in ARCHIVE.rglob("*") if p.is_file() and p.name != "FILE_INDEX.csv"]
    paths.sort(key=lambda x: x.relative_to(ARCHIVE).as_posix())
    with (DOCS / "FILE_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "size_bytes", "sha256", "method", "repeat", "fold", "imputation"])
        for p in paths:
            rel = p.relative_to(ARCHIVE).as_posix()
            size = p.stat().st_size
            method, rep, fold, mi = metadata(rel)
            w.writerow([rel, size, digest(p), method, rep, fold, mi])
            rows += 1
            nbytes += size
            if rows % 20000 == 0:
                print(f"hashed {rows}/{len(paths)} files", flush=True)
    return rows, nbytes


def main() -> None:
    start = time.time()
    audit = audit_copy()
    checks = fold_checks()
    write_fold_map()
    write_readme(audit, checks)
    (DOCS / "COPY_AUDIT.json").write_text(json.dumps({"source_comparison": audit, "completion_checks": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    n, size = write_index()
    info = {"archive_path": str(ARCHIVE), "files_in_index": n, "indexed_bytes": size, "elapsed_seconds": round(time.time() - start, 2), "workbook_sha256": audit["workbook_sha256"], "completion_checks": checks}
    (DOCS / "PACKAGE_STATUS.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
