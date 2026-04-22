#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
import weave
from slidevqa_experiment.config import Settings
from weave import weave_client_context
from weave.trace_server import trace_server_interface as tsi

try:
    import wandb_workspaces.reports.v2 as wr
except ImportError as exc:  # pragma: no cover - runtime guidance
    raise SystemExit(
        "wandb-workspaces is required. Run with:\n"
        "  uv run --with wandb-workspaces python scripts/create_wandb_report.py"
    ) from exc


REPORT_TITLE = "Context-RAG SlideVQA: Baseline vs Best Enhanced"
REPORT_DESCRIPTION = "Public report for the best baseline and best enhanced SlideVQA results tracked in Weave."
BASELINE_REPORT_RUN_ID = "slidevqa-report-baseline-best"
ENHANCED_REPORT_RUN_ID = "slidevqa-report-enhanced-best"
REPORT_GROUP = "slidevqa_best_public_report"
BASELINE_COLOR = "#5B6C8D"
ENHANCED_COLOR = "#D97A2B"
WEAVE_PROJECT_URL_TMPL = "https://wandb.ai/{entity}/{project}/weave"


@dataclass
class EvalSelection:
    variant: str
    summary_path: Path
    summary: dict[str, Any]
    results_path: Path
    rows: list[dict[str, Any]]

    @property
    def run_name(self) -> str:
        return self.summary_path.name.removesuffix(".summary.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _entity_project(settings: Settings) -> tuple[str, str]:
    api = wandb.Api()
    viewer = api.viewer
    project_path = settings.weave_project.strip()
    if "/" in project_path:
        entity, project = project_path.split("/", 1)
        return entity, project
    entity = getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    if not entity:
        raise RuntimeError("Unable to infer W&B entity from the current account.")
    return entity, project_path


def _candidate_selections(settings: Settings, variant: str) -> list[EvalSelection]:
    candidates: list[EvalSelection] = []
    for path in settings.eval_runs_dir.glob("*.summary.json"):
        summary = _read_json(path)
        if summary.get("status") != "completed":
            continue
        if summary.get("variant") != variant:
            continue
        if summary.get("dataset") != settings.dataset_name:
            continue
        results_path_raw = str(summary.get("results_path") or "").strip()
        if not results_path_raw:
            continue
        results_path = Path(results_path_raw)
        if not results_path.exists():
            continue
        rows = _read_jsonl(results_path)
        candidates.append(
            EvalSelection(
                variant=variant,
                summary_path=path,
                summary=summary,
                results_path=results_path,
                rows=rows,
            )
        )
    return candidates


def _pick_best_selection(settings: Settings, variant: str) -> EvalSelection:
    candidates = _candidate_selections(settings, variant)
    if not candidates:
        raise RuntimeError(f"No completed eval summaries found for variant='{variant}'.")

    max_num_samples = max(int(item.summary.get("num_samples", 0) or 0) for item in candidates)
    full_candidates = [item for item in candidates if int(item.summary.get("num_samples", 0) or 0) == max_num_samples]

    def sort_key(item: EvalSelection) -> tuple[float, float, float, float, int]:
        summary = item.summary
        return (
            _float(summary.get("f1")),
            _float(summary.get("exact_match")),
            _float(summary.get("recall_at_5")),
            -_float(summary.get("avg_latency_s")),
            int(summary.get("generated_at", 0) or 0),
        )

    return sorted(full_candidates, key=sort_key, reverse=True)[0]


def _make_trajectory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trajectory_rows: list[dict[str, Any]] = []
    for row in rows:
        scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
        ragas = row.get("ragas") if isinstance(row.get("ragas"), dict) else {}
        trajectory_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "question": row.get("question", ""),
                "gold_answers": "\n".join(str(item) for item in row.get("answers", [])),
                "prediction": row.get("prediction", ""),
                "exact_match": _float(scores.get("exact_match")),
                "f1": _float(scores.get("f1")),
                "recall_at_5": _float(scores.get("recall_at_5")),
                "latency_s": _float(row.get("latency_s")),
                "relevant_doc_page_ids": "\n".join(str(item) for item in row.get("relevant_doc_page_ids", [])),
                "retrieved_doc_page_ids": "\n".join(str(item) for item in row.get("retrieved_doc_page_ids", [])),
                "citations": json.dumps(row.get("citations", []), ensure_ascii=False),
                "ragas_faithfulness": ragas.get("faithfulness"),
                "ragas_answer_relevancy": ragas.get("answer_relevancy"),
                "ragas_context_precision": ragas.get("context_precision"),
                "ragas_context_recall": ragas.get("context_recall"),
                "ragas_answer_correctness": ragas.get("answer_correctness"),
            }
        )
    return trajectory_rows


def _make_table(rows: list[dict[str, Any]]) -> wandb.Table:
    if not rows:
        return wandb.Table(columns=["empty"], data=[["no rows"]])
    columns = list(rows[0].keys())
    data = [[row.get(column) for column in columns] for row in rows]
    return wandb.Table(columns=columns, data=data)


def _build_score_change_rows(
    baseline_rows: list[dict[str, Any]],
    enhanced_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    baseline_by_id = {str(row.get("sample_id")): row for row in baseline_rows}
    enhanced_by_id = {str(row.get("sample_id")): row for row in enhanced_rows}

    changed_rows: list[dict[str, Any]] = []
    counts = {
        "improved": 0,
        "regressed": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }

    for sample_id in sorted(set(baseline_by_id) & set(enhanced_by_id)):
        baseline = baseline_by_id[sample_id]
        enhanced = enhanced_by_id[sample_id]
        b_scores = baseline.get("scores") if isinstance(baseline.get("scores"), dict) else {}
        e_scores = enhanced.get("scores") if isinstance(enhanced.get("scores"), dict) else {}

        b_em = _float(b_scores.get("exact_match"))
        e_em = _float(e_scores.get("exact_match"))
        b_f1 = _float(b_scores.get("f1"))
        e_f1 = _float(e_scores.get("f1"))

        if b_em < 1.0 and e_em >= 1.0:
            change_type = "improved"
            counts["improved"] += 1
        elif b_em >= 1.0 and e_em < 1.0:
            change_type = "regressed"
            counts["regressed"] += 1
        elif b_em >= 1.0 and e_em >= 1.0:
            change_type = "both_correct"
            counts["both_correct"] += 1
        else:
            change_type = "both_wrong"
            counts["both_wrong"] += 1

        if change_type not in {"improved", "regressed"}:
            continue

        changed_rows.append(
            {
                "sample_id": sample_id,
                "change_type": change_type,
                "question": baseline.get("question", ""),
                "gold_answers": "\n".join(str(item) for item in baseline.get("answers", [])),
                "baseline_prediction": baseline.get("prediction", ""),
                "enhanced_prediction": enhanced.get("prediction", ""),
                "baseline_exact_match": b_em,
                "enhanced_exact_match": e_em,
                "baseline_f1": b_f1,
                "enhanced_f1": e_f1,
                "delta_f1": e_f1 - b_f1,
                "baseline_retrieved_doc_page_ids": "\n".join(
                    str(item) for item in baseline.get("retrieved_doc_page_ids", [])
                ),
                "enhanced_retrieved_doc_page_ids": "\n".join(
                    str(item) for item in enhanced.get("retrieved_doc_page_ids", [])
                ),
            }
        )

    changed_rows.sort(
        key=lambda row: (
            1 if row["change_type"] == "improved" else 0,
            row["delta_f1"],
        ),
        reverse=True,
    )
    return changed_rows, counts


def _load_build_manifest(settings: Settings, variant: str) -> dict[str, Any]:
    manifest_path = settings.artifact_dir(variant) / "manifest.json"
    if not manifest_path.exists():
        return {}
    return _read_json(manifest_path)


def _load_weave_metadata(
    entity: str,
    project: str,
    baseline: EvalSelection,
    enhanced: EvalSelection,
    baseline_manifest: dict[str, Any],
    enhanced_manifest: dict[str, Any],
) -> dict[str, Any]:
    client = weave.init(f"{entity}/{project}")
    del client  # keep init side effects only
    weave_client = weave_client_context.require_weave_client()

    object_ids = [
        f"eval-{baseline.run_name}",
        f"eval-{enhanced.run_name}",
        "build-baseline",
        "build-enhanced",
    ]
    req = tsi.ObjQueryReq(
        project_id=weave_client.project_id,
        filter=tsi.ObjectVersionFilter(object_ids=object_ids),
    )
    obj_res = weave_client.server.objs_query(req)
    objects: dict[str, list[dict[str, Any]]] = {}
    for obj in obj_res.objs:
        objects.setdefault(obj.object_id, []).append(
            {
                "object_id": obj.object_id,
                "digest": obj.digest,
                "version_index": obj.version_index,
                "created_at": str(obj.created_at),
                "val": obj.val,
            }
        )

    def pick_build_version(variant: str, manifest: dict[str, Any]) -> dict[str, Any]:
        candidates = objects.get(f"build-{variant}", [])
        built_at = manifest.get("built_at")
        for candidate in candidates:
            val = candidate.get("val") if isinstance(candidate.get("val"), dict) else {}
            if val.get("built_at") == built_at:
                return candidate
        return candidates[-1] if candidates else {}

    stats_res = weave_client.server.calls_query_stats(
        tsi.CallsQueryStatsReq(project_id=weave_client.project_id)
    )
    calls_res = weave_client.server.calls_query(
        tsi.CallsQueryReq(project_id=weave_client.project_id, limit=1000)
    )
    op_name_counts: dict[str, int] = {}
    for call in calls_res.calls:
        op_name = str(getattr(call, "op_name", "") or "")
        op_name_counts[op_name] = op_name_counts.get(op_name, 0) + 1

    return {
        "project_url": WEAVE_PROJECT_URL_TMPL.format(entity=entity, project=project),
        "baseline_eval": (objects.get(f"eval-{baseline.run_name}", [{}]) or [{}])[-1],
        "enhanced_eval": (objects.get(f"eval-{enhanced.run_name}", [{}]) or [{}])[-1],
        "baseline_build": pick_build_version("baseline", baseline_manifest),
        "enhanced_build": pick_build_version("enhanced", enhanced_manifest),
        "total_calls": int(getattr(stats_res, "count", 0) or 0),
        "op_name_counts": dict(sorted(op_name_counts.items(), key=lambda item: item[1], reverse=True)),
    }


def _publish_report_run(
    *,
    entity: str,
    project: str,
    run_id: str,
    run_name: str,
    selection: EvalSelection,
    build_manifest: dict[str, Any],
    weave_eval: dict[str, Any],
    weave_build: dict[str, Any],
    trajectory_rows: list[dict[str, Any]],
    extra_summary: dict[str, Any] | None = None,
    extra_tables: dict[str, wandb.Table] | None = None,
) -> str:
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="allow",
        name=run_name,
        group=REPORT_GROUP,
        job_type="report-source",
        tags=["report-source", "slidevqa", selection.variant],
    )
    run.config.update(
        {
            "report_group": REPORT_GROUP,
            "report_run_key": f"{selection.variant}_best",
            "variant": selection.variant,
            "dataset": selection.summary.get("dataset"),
            "source_eval_run_name": selection.run_name,
            "source_summary_path": str(selection.summary_path),
            "source_results_path": str(selection.results_path),
            "source_weave_eval_object_id": weave_eval.get("object_id", ""),
            "source_weave_eval_digest": weave_eval.get("digest", ""),
            "source_weave_build_object_id": weave_build.get("object_id", ""),
            "source_weave_build_digest": weave_build.get("digest", ""),
        },
        allow_val_change=True,
    )

    summary_payload = {
        "variant": selection.variant,
        "num_samples": int(selection.summary.get("num_samples", 0) or 0),
        "exact_match": _float(selection.summary.get("exact_match")),
        "f1": _float(selection.summary.get("f1")),
        "recall_at_5": _float(selection.summary.get("recall_at_5")),
        "avg_latency_s": _float(selection.summary.get("avg_latency_s")),
        "p95_latency_s": _float(selection.summary.get("p95_latency_s")),
        "generated_at": int(selection.summary.get("generated_at", 0) or 0),
        "ragas_enabled": bool(selection.summary.get("ragas_enabled")),
        "ragas_error": str(selection.summary.get("ragas_error") or ""),
        "num_pages": int(build_manifest.get("num_pages", 0) or 0),
        "num_chunks": int(build_manifest.get("num_chunks", 0) or 0),
        "build_built_at": int(build_manifest.get("built_at", 0) or 0),
        "source_weave_eval_object": weave_eval.get("object_id", ""),
        "source_weave_eval_digest": weave_eval.get("digest", ""),
        "source_weave_build_object": weave_build.get("object_id", ""),
        "source_weave_build_digest": weave_build.get("digest", ""),
    }
    if extra_summary:
        summary_payload.update(extra_summary)

    run.summary.update(summary_payload)
    run.summary["trajectory_table"] = _make_table(trajectory_rows)
    for table_name, table in (extra_tables or {}).items():
        run.summary[table_name] = table
    run.finish()
    return run.url


def _selected_runset(entity: str, project: str, selected_run_ids: dict[str, wr.RunSettings], name: str) -> wr.Runset:
    runset = wr.Runset(entity=entity, project=project, name=name, run_settings=selected_run_ids)
    runset._selections_root = 0
    return runset


def _render_weave_object_line(label: str, obj: dict[str, Any]) -> str:
    object_id = obj.get("object_id") or "(missing)"
    digest = obj.get("digest") or "(missing)"
    return f"- {label}: `{object_id}` @ `{digest}`"


def _build_report(
    *,
    entity: str,
    project: str,
    baseline: EvalSelection,
    enhanced: EvalSelection,
    baseline_run_url: str,
    enhanced_run_url: str,
    changed_rows: list[dict[str, Any]],
    change_counts: dict[str, int],
    weave_meta: dict[str, Any],
) -> tuple[str, str]:
    baseline_summary = baseline.summary
    enhanced_summary = enhanced.summary
    delta_em = _float(enhanced_summary.get("exact_match")) - _float(baseline_summary.get("exact_match"))
    delta_f1 = _float(enhanced_summary.get("f1")) - _float(baseline_summary.get("f1"))
    delta_recall = _float(enhanced_summary.get("recall_at_5")) - _float(baseline_summary.get("recall_at_5"))
    delta_latency = _float(enhanced_summary.get("avg_latency_s")) - _float(baseline_summary.get("avg_latency_s"))

    compare_runset = _selected_runset(
        entity,
        project,
        {
            BASELINE_REPORT_RUN_ID: wr.RunSettings(color=BASELINE_COLOR),
            ENHANCED_REPORT_RUN_ID: wr.RunSettings(color=ENHANCED_COLOR),
        },
        name="Selected Report Runs",
    )
    baseline_runset = _selected_runset(
        entity,
        project,
        {BASELINE_REPORT_RUN_ID: wr.RunSettings(color=BASELINE_COLOR)},
        name="Best Baseline",
    )
    enhanced_runset = _selected_runset(
        entity,
        project,
        {ENHANCED_REPORT_RUN_ID: wr.RunSettings(color=ENHANCED_COLOR)},
        name="Best Enhanced",
    )

    top_ops = list(weave_meta.get("op_name_counts", {}).items())[:5]
    top_ops_md = "\n".join(f"- `{name}`: {count}" for name, count in top_ops) if top_ops else "- No traced ops found."

    summary_callout = (
        f"Best enhanced improves EM by {delta_em * 100:.2f} pts and F1 by {delta_f1 * 100:.2f} pts "
        f"over the best baseline, while keeping Recall@5 at {_pct(enhanced_summary.get('recall_at_5'))}. "
        f"Average latency changes by {delta_latency:+.2f}s. "
        f"Score-changing samples: {change_counts['improved']} improved, {change_counts['regressed']} regressed."
    )

    overview_md = f"""
本报告面向 `Context-RAG` 的 `SlideVQA` 实验，对应 Weave 项目为 [{entity}/{project}]({weave_meta['project_url']}).

- 选择规则：在 `data/eval_runs/*.summary.json` 中，仅保留 `status=completed`、`dataset=slidevqa`、且样本数为该变体最大值的评测；再按 `F1 -> EM -> Recall@5 -> Avg Latency -> generated_at` 排序，选出最佳 `baseline` 与最佳 `enhanced`。
- 选中 baseline：`{baseline.run_name}`，W&B run: [slidevqa-report-baseline-best]({baseline_run_url})
- 选中 enhanced：`{enhanced.run_name}`，W&B run: [slidevqa-report-enhanced-best]({enhanced_run_url})
- 本项目当前 Weave workspace 总 call 数：`{weave_meta['total_calls']}`
"""

    highlights_md = f"""
- Baseline best: EM `{_pct(baseline_summary.get('exact_match'))}`, F1 `{_pct(baseline_summary.get('f1'))}`, Recall@5 `{_pct(baseline_summary.get('recall_at_5'))}`, Avg Latency `{_float(baseline_summary.get('avg_latency_s')):.2f}s`
- Enhanced best: EM `{_pct(enhanced_summary.get('exact_match'))}`, F1 `{_pct(enhanced_summary.get('f1'))}`, Recall@5 `{_pct(enhanced_summary.get('recall_at_5'))}`, Avg Latency `{_float(enhanced_summary.get('avg_latency_s')):.2f}s`
- Delta: EM `{delta_em * 100:+.2f}` pts, F1 `{delta_f1 * 100:+.2f}` pts, Recall@5 `{delta_recall * 100:+.2f}` pts, Avg Latency `{delta_latency:+.2f}s`
- Pairwise outcome on 36 shared samples: `{change_counts['improved']}` improved, `{change_counts['regressed']}` regressed, `{change_counts['both_correct']}` both correct, `{change_counts['both_wrong']}` both wrong
"""

    baseline_md = f"""
- Source eval summary: `{baseline.summary_path.name}`
- Source results file: `{baseline.results_path.name}`
{_render_weave_object_line("Weave eval object", weave_meta.get("baseline_eval", {}))}
{_render_weave_object_line("Weave build object", weave_meta.get("baseline_build", {}))}
"""

    enhanced_md = f"""
- Source eval summary: `{enhanced.summary_path.name}`
- Source results file: `{enhanced.results_path.name}`
{_render_weave_object_line("Weave eval object", weave_meta.get("enhanced_eval", {}))}
{_render_weave_object_line("Weave build object", weave_meta.get("enhanced_build", {}))}
"""

    score_change_md = f"""
下表仅保留分数发生变化的样本，共 `{len(changed_rows)}` 条。

- Improved: `{change_counts['improved']}`
- Regressed: `{change_counts['regressed']}`
"""

    notes_md = f"""
- 数据集：`SlideVQA`，当前最佳对比均基于 `36` 条样本。
- 检索范围：每条问题仅在自己的 deck page scope 中检索，避免跨文档污染。
- Baseline：OCR 文本切块 + BM25 + text dense + page-image dense。
- Enhanced：在 baseline 基础上增加 `contextual chunk text` 与 `page proxy`（BM25 + dense）。
- 指标：`Recall@5` 衡量检索；`EM/F1` 衡量答案；`avg_latency_s/p95_latency_s` 衡量端到端时延。
- RAGAS：两条入选记录都写入了 `ragas_enabled=true`，但实际 `ragas_error` 为多模态 judge `qwen3-vl` 不适合结构化 RAGAS，因此本报告不将 RAGAS 作为主比较指标。
- 轨迹说明：当前主 pipeline 只把 `build/eval summary object` 发布到 Weave；逐样本轨迹表来自本地 `data/eval_runs/*.jsonl` 的记录导出后再挂到 W&B run summary table。
- Weave traced ops top-5:
{top_ops_md}
"""

    report = wr.Report(
        entity=entity,
        project=project,
        title=REPORT_TITLE,
        description=REPORT_DESCRIPTION,
        width="fluid",
        blocks=[
            wr.H1("Context-RAG / SlideVQA: Baseline vs Best Enhanced"),
            wr.MarkdownBlock(overview_md.strip()),
            wr.TableOfContents(),
            wr.CalloutBlock(summary_callout),
            wr.H2("Metric Comparison"),
            wr.MarkdownBlock(highlights_md.strip()),
            wr.PanelGrid(
                runsets=[compare_runset],
                hide_run_sets=True,
                panels=[
                    wr.RunComparer(diff_only=True, layout=wr.Layout(x=0, y=0, w=24, h=10)),
                    wr.BarPlot(
                        title="Quality Metrics",
                        metrics=[
                            wr.SummaryMetric("exact_match"),
                            wr.SummaryMetric("f1"),
                            wr.SummaryMetric("recall_at_5"),
                        ],
                        orientation="h",
                        layout=wr.Layout(x=0, y=10, w=12, h=8),
                    ),
                    wr.BarPlot(
                        title="Latency Metrics",
                        metrics=[
                            wr.SummaryMetric("avg_latency_s"),
                            wr.SummaryMetric("p95_latency_s"),
                        ],
                        orientation="h",
                        layout=wr.Layout(x=12, y=10, w=12, h=8),
                    ),
                    wr.BarPlot(
                        title="Build Footprint",
                        metrics=[
                            wr.SummaryMetric("num_pages"),
                            wr.SummaryMetric("num_chunks"),
                        ],
                        orientation="h",
                        layout=wr.Layout(x=0, y=18, w=12, h=8),
                    ),
                    wr.ScatterPlot(
                        title="F1 vs Avg Latency",
                        x=wr.SummaryMetric("avg_latency_s"),
                        y=wr.SummaryMetric("f1"),
                        layout=wr.Layout(x=12, y=18, w=12, h=8),
                    ),
                ],
            ),
            wr.H2("Baseline Record"),
            wr.MarkdownBlock(baseline_md.strip()),
            wr.PanelGrid(
                runsets=[baseline_runset],
                hide_run_sets=True,
                panels=[
                    wr.WeavePanelSummaryTable(
                        table_name="trajectory_table",
                        layout=wr.Layout(x=0, y=0, w=24, h=18),
                    )
                ],
            ),
            wr.H2("Best Enhanced Record"),
            wr.MarkdownBlock(enhanced_md.strip()),
            wr.PanelGrid(
                runsets=[enhanced_runset],
                hide_run_sets=True,
                panels=[
                    wr.WeavePanelSummaryTable(
                        table_name="trajectory_table",
                        layout=wr.Layout(x=0, y=0, w=24, h=18),
                    )
                ],
            ),
            wr.H2("Score-Changing Trajectories"),
            wr.MarkdownBlock(score_change_md.strip()),
            wr.PanelGrid(
                runsets=[enhanced_runset],
                hide_run_sets=True,
                panels=[
                    wr.WeavePanelSummaryTable(
                        table_name="delta_table",
                        layout=wr.Layout(x=0, y=0, w=24, h=14),
                    )
                ],
            ),
            wr.H2("Project Notes"),
            wr.MarkdownBlock(notes_md.strip()),
        ],
    )
    report.save(draft=False)
    share_url = report.enable_share_link()
    return report.url, share_url


def main() -> int:
    settings = Settings.from_env()
    settings.ensure_dirs()
    if settings.wandb_api_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = settings.wandb_api_key
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is required.")

    entity, project = _entity_project(settings)

    baseline = _pick_best_selection(settings, "baseline")
    enhanced = _pick_best_selection(settings, "enhanced")
    baseline_manifest = _load_build_manifest(settings, "baseline")
    enhanced_manifest = _load_build_manifest(settings, "enhanced")
    weave_meta = _load_weave_metadata(
        entity=entity,
        project=project,
        baseline=baseline,
        enhanced=enhanced,
        baseline_manifest=baseline_manifest,
        enhanced_manifest=enhanced_manifest,
    )

    baseline_trajectory_rows = _make_trajectory_rows(baseline.rows)
    enhanced_trajectory_rows = _make_trajectory_rows(enhanced.rows)
    changed_rows, change_counts = _build_score_change_rows(baseline.rows, enhanced.rows)

    baseline_run_url = _publish_report_run(
        entity=entity,
        project=project,
        run_id=BASELINE_REPORT_RUN_ID,
        run_name="SlideVQA Best Baseline",
        selection=baseline,
        build_manifest=baseline_manifest,
        weave_eval=weave_meta.get("baseline_eval", {}),
        weave_build=weave_meta.get("baseline_build", {}),
        trajectory_rows=baseline_trajectory_rows,
    )
    enhanced_run_url = _publish_report_run(
        entity=entity,
        project=project,
        run_id=ENHANCED_REPORT_RUN_ID,
        run_name="SlideVQA Best Enhanced",
        selection=enhanced,
        build_manifest=enhanced_manifest,
        weave_eval=weave_meta.get("enhanced_eval", {}),
        weave_build=weave_meta.get("enhanced_build", {}),
        trajectory_rows=enhanced_trajectory_rows,
        extra_summary={
            "improved_vs_baseline": change_counts["improved"],
            "regressed_vs_baseline": change_counts["regressed"],
            "both_correct_vs_baseline": change_counts["both_correct"],
            "both_wrong_vs_baseline": change_counts["both_wrong"],
        },
        extra_tables={"delta_table": _make_table(changed_rows)},
    )

    report_url, share_url = _build_report(
        entity=entity,
        project=project,
        baseline=baseline,
        enhanced=enhanced,
        baseline_run_url=baseline_run_url,
        enhanced_run_url=enhanced_run_url,
        changed_rows=changed_rows,
        change_counts=change_counts,
        weave_meta=weave_meta,
    )

    print(
        json.dumps(
            {
                "entity": entity,
                "project": project,
                "weave_project_url": weave_meta["project_url"],
                "baseline_run_name": baseline.run_name,
                "enhanced_run_name": enhanced.run_name,
                "baseline_report_run_url": baseline_run_url,
                "enhanced_report_run_url": enhanced_run_url,
                "report_url": report_url,
                "share_url": share_url,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
