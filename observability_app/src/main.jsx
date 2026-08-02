import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ClockIcon,
  ReloadIcon,
} from "@radix-ui/react-icons";
import "./styles.css";

const EMPTY = {
  generated_at: "",
  ssh: { ok: false },
  quota: {},
  jobs: [],
  history: [],
  stats: { gpu_hours_7d: 0, training_streak_days: 0, days: {} },
  remote_artifacts: [],
  training: { metrics: [], errors: [], progress: [] },
  results: [],
  notes: [],
};

function App() {
  const [snapshot, setSnapshot] = useState(EMPTY);
  const [status, setStatus] = useState("loading");
  const [selectedRunId, setSelectedRunId] = useState(null);

  async function loadSnapshot() {
    setStatus("loading");
    try {
      const response = await fetch(`/latest.json?ts=${Date.now()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSnapshot(await response.json());
      setStatus("ready");
    } catch (error) {
      setStatus(error.message || "error");
    }
  }

  useEffect(() => {
    loadSnapshot();
  }, []);

  const issues = useMemo(() => dedupeIssues(snapshot.training?.errors ?? []), [snapshot]);
  const metricsByRun = useMemo(() => groupMetricsByRun(snapshot.training?.metrics ?? []), [snapshot]);
  const issuesByRun = useMemo(() => groupByRun(issues), [issues]);
  const progressByRun = useMemo(() => groupProgressByRun(snapshot.training?.progress ?? []), [snapshot]);
  const model = useMemo(() => deriveModel(snapshot, issues), [snapshot, issues]);
  const runsForInspection = useMemo(
    () => buildInspectableRuns(snapshot.history ?? [], metricsByRun, issuesByRun, progressByRun),
    [snapshot.history, metricsByRun, issuesByRun, progressByRun],
  );

  useEffect(() => {
    if (selectedRunId && !runsForInspection.some((run) => run.job_id === selectedRunId)) {
      setSelectedRunId(null);
    }
  }, [selectedRunId, runsForInspection]);

  if (status === "loading" && !snapshot.generated_at) {
    return <LoadingShell />;
  }

  return (
    <main className="min-h-[100dvh] bg-[#151814] text-[#f5f2ea]">
      <div className="pointer-events-none fixed inset-0 observer-grid opacity-40" />
      <div className="mx-auto grid max-w-[1500px] grid-cols-1 gap-5 px-4 py-4 md:px-6 lg:grid-cols-[minmax(0,1fr)_380px] lg:py-6">
        <section className="min-w-0 space-y-5">
          <Header snapshot={snapshot} status={status} onRefresh={loadSnapshot} />
          <StatusStrip model={model} />
          <LiveQueue jobs={snapshot.jobs} />
          <RunInspector
            runs={runsForInspection}
            metricsByRun={metricsByRun}
            issuesByRun={issuesByRun}
            progressByRun={progressByRun}
            selectedRunId={selectedRunId}
            setSelectedRunId={setSelectedRunId}
          />
        </section>

        <aside className="min-w-0 space-y-5">
          <BlockersPanel artifacts={snapshot.remote_artifacts} issues={issues} />
          <RunHistory
            runs={runsForInspection}
            selectedRunId={selectedRunId}
            setSelectedRunId={setSelectedRunId}
          />
          <ResultPreviews results={snapshot.results} />
        </aside>
      </div>
    </main>
  );
}

function deriveModel(snapshot, issues) {
  const artifacts = snapshot.remote_artifacts ?? [];
  const jobs = snapshot.jobs ?? [];
  const missingArtifacts = artifacts.filter((item) => item.status === "missing").length;
  const blockers = issues.filter((issue) => issue.severity === "blocker").length;

  return {
    sshOk: Boolean(snapshot.ssh?.ok),
    queueLabel: queueLabel(jobs),
    blockerCount: missingArtifacts + blockers,
    scratch: snapshot.quota?.scratch_percent ?? "unknown",
    home: snapshot.quota?.home_percent ?? "unknown",
    gpuHours: snapshot.stats?.gpu_hours_7d ?? 0,
    streak: snapshot.stats?.training_streak_days ?? 0,
  };
}

function Header({ snapshot, status, onRefresh }) {
  return (
    <header className="rounded-[28px] border border-white/10 bg-[#f5f2ea] text-[#171a16] shadow-[0_30px_80px_-50px_rgba(0,0,0,.8)]">
      <div className="grid gap-5 p-5 md:grid-cols-[280px_minmax(0,1fr)] md:items-center md:p-6">
        <div className="min-w-0">
          <p className="font-mono text-xs font-semibold uppercase text-[#147d73]">SALLM Observer</p>
          <h1 className="mt-2 text-3xl font-semibold leading-none md:text-4xl">Cluster operations</h1>
          <p className="mt-3 max-w-[520px] text-sm leading-6 text-[#5d645d]">SSH, Slurm, quota, blockers, and selected-run evidence from the latest local snapshot.</p>
        </div>
        <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
          <div className="grid gap-2 sm:grid-cols-2">
            <StatusLine label="Snapshot" value={formatTimestamp(snapshot.generated_at)} />
            <StatusLine label="Refresh" value={status === "ready" ? "current file loaded" : status} />
          </div>
          <button
            className="inline-flex h-11 items-center justify-center gap-2 rounded-full bg-[#147d73] px-5 text-sm font-semibold text-white transition hover:bg-[#0f6b62] active:translate-y-px"
            onClick={onRefresh}
          >
            <ReloadIcon />
            Reload
          </button>
        </div>
      </div>
    </header>
  );
}

function StatusStrip({ model }) {
  const stats = [
    ["SSH", model.sshOk ? "connected" : "offline", model.sshOk ? "good" : "bad"],
    ["Queue", model.queueLabel, "neutral"],
    ["Scratch", pct(model.scratch), Number(model.scratch) >= 85 ? "warn" : "neutral"],
    ["GPU hours", fmt(model.gpuHours), "neutral"],
    ["Streak", `${model.streak}d`, "neutral"],
    ["Blockers", model.blockerCount, model.blockerCount ? "warn" : "good"],
  ];

  return (
    <section className="grid grid-cols-2 overflow-hidden rounded-[24px] border border-white/10 bg-white/[.055] md:grid-cols-3 xl:grid-cols-6">
      {stats.map(([label, value, tone]) => (
        <div key={label} className="border-b border-r border-white/10 p-4 last:border-r-0 md:p-5">
          <p className="mb-2 font-mono text-[11px] uppercase text-[#9da79b]">{label}</p>
          <p className={`font-mono text-2xl font-semibold ${toneClass(tone)}`}>{value}</p>
        </div>
      ))}
    </section>
  );
}

function LiveQueue({ jobs }) {
  return (
    <section className="panel">
      <SectionTitle title="Live queue" detail="squeue, quota first" />
      <div className="grid gap-3 md:hidden">
        {jobs.length ? (
          jobs.map((job) => (
            <div key={job.job_id} className="side-row">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div>
                  <p className="mono text-[#8e988c]">{job.job_id}</p>
                  <p className="font-semibold">{job.name}</p>
                </div>
                <Pill state={job.state}>{job.state}</Pill>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <StatusLine label="Partition" value={job.partition} />
                <StatusLine label="Elapsed" value={job.elapsed} />
                <StatusLine label="Reason" value={job.reason} />
                <StatusLine label="GRES" value={job.gres} />
              </div>
            </div>
          ))
        ) : (
          <EmptyState text="No live jobs in this snapshot." />
        )}
      </div>
      <div className="hidden overflow-x-auto md:block">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>State</th>
              <th>Partition</th>
              <th>Elapsed</th>
              <th>Reason</th>
              <th>GRES</th>
            </tr>
          </thead>
          <tbody>
            {jobs.length ? (
              jobs.map((job) => (
                <tr key={job.job_id}>
                  <td className="mono">{job.job_id}</td>
                  <td>{job.name}</td>
                  <td><Pill state={job.state}>{job.state}</Pill></td>
                  <td>{job.partition}</td>
                  <td>{job.elapsed}</td>
                  <td>{job.reason}</td>
                  <td className="mono">{job.gres}</td>
                </tr>
              ))
            ) : (
              <EmptyRow cells={7} text="No live jobs in this snapshot." />
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RunInspector({ runs, metricsByRun, issuesByRun, progressByRun, selectedRunId, setSelectedRunId }) {
  const selectedRun = runs.find((run) => run.job_id === selectedRunId) ?? null;
  const selectedMetrics = selectedRun ? metricsByRun[selectedRun.job_id] ?? [] : [];
  const selectedIssues = selectedRun ? issuesByRun[selectedRun.job_id] ?? [] : [];
  const selectedProgress = selectedRun ? progressByRun[selectedRun.job_id] ?? [] : [];
  const metricRuns = runs.filter((run) => run.metricCount > 0).length;

  return (
    <section className="panel">
      <SectionTitle title="Run inspector" detail={`${metricRuns} runs with parsed metrics`} />
      <div className="grid gap-5 lg:grid-cols-[320px_minmax(0,1fr)] 2xl:grid-cols-[340px_minmax(0,1fr)]">
        <div className="run-list">
          {runs.map((run) => (
            <button
              key={run.job_id}
              className={`run-button ${selectedRunId === run.job_id ? "run-button-selected" : ""}`}
              onClick={() => setSelectedRunId(run.job_id)}
            >
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold">{run.name || `job ${run.job_id}`}</span>
                <span className="mono mt-1 block text-[#8e988c]">{run.job_id} · {run.state}</span>
                <span className="run-meta">{formatRunTime(run.start)} · {formatDuration(run.elapsed_seconds)} · exit {run.exit ?? "n/a"}</span>
              </span>
              <span className="grid justify-items-end gap-1">
                <span className="metric-badge">{run.metricCount || "no"} pts</span>
                {run.issueCount > 0 && <span className="metric-badge metric-badge-warn">{run.issueCount} issues</span>}
              </span>
            </button>
          ))}
          {!runs.length && <EmptyState text="No past runs found in this snapshot." />}
        </div>

        <div className="min-w-0">
          {selectedRun ? (
            <div className="space-y-5">
              <div className="inspector-summary">
                <StatusLine label="Job" value={selectedRun.job_id} />
                <StatusLine label="State" value={selectedRun.state} />
                <StatusLine label="Partition" value={selectedRun.partition ?? "unknown"} />
                <StatusLine label="GPU time" value={formatGpuHours(selectedRun.gpu_hours)} />
                <StatusLine label="Elapsed" value={formatDuration(selectedRun.elapsed_seconds)} />
                <StatusLine label="Start" value={formatRunTime(selectedRun.start)} />
                <StatusLine label="End" value={formatRunTime(selectedRun.end)} />
                <StatusLine label="Exit" value={selectedRun.exit ?? "n/a"} />
                <StatusLine label="Metric source" value={selectedMetrics[0]?.source ?? "none parsed"} />
              </div>
              <RunLossChart metrics={selectedMetrics} />
              <MetricsTable
                metrics={selectedMetrics}
                title="Selected run metrics"
                detail={selectedMetrics.length ? "newest parsed rows" : "no Trainer rows"}
              />
              <RunSignals issues={selectedIssues} progress={selectedProgress} />
            </div>
          ) : (
            <EmptyState text="Select a past run to inspect loss and recently printed metrics." />
          )}
        </div>
      </div>
    </section>
  );
}

function RunSignals({ issues, progress }) {
  const rows = progress.slice(-3).reverse();
  return (
    <div>
      <SectionTitle title="Selected run signals" detail={`${issues.length} issues, ${progress.length} progress rows`} />
      <div className="grid gap-3 md:grid-cols-2">
        <div className="side-row">
          <p className="mb-3 text-sm font-semibold">Progress</p>
          {rows.length ? (
            <div className="space-y-3">
              {rows.map((item, index) => (
                <div key={`${item.source}-${item.step}-${index}`}>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="mono text-[#8e988c]">{item.step}/{item.total}</span>
                    <span className="mono">{item.pct}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full rounded-full bg-[#63b8aa]" style={{ width: `${item.pct}%` }} />
                  </div>
                  <p className="mono mt-2 text-xs text-[#8e988c]">ETA {item.eta}</p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState text="No progress parsed for this run." />
          )}
        </div>
        <div className="side-row">
          <p className="mb-3 text-sm font-semibold">Issues</p>
          {issues.length ? (
            <div className="space-y-3">
              {issues.slice(-4).reverse().map((item) => (
                <div key={`${item.source}-${item.line}`}>
                  <Pill state={item.severity}>{item.severity}</Pill>
                  <p className="mono mt-2 break-words">{item.line}</p>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState text="No warnings or blockers parsed for this run." />
          )}
        </div>
      </div>
    </div>
  );
}

function RunLossChart({ metrics }) {
  const values = metrics.map(lossValue).filter((value) => typeof value === "number");
  const points = toPolyline(values, 680, 240);

  return (
    <div>
      <SectionTitle title="Loss trend" detail={`${values.length} selected-run points`} />
      <div className="relative h-[280px] overflow-hidden rounded-[22px] border border-white/10 bg-[#10130f]">
        <div className="absolute inset-0 chart-grid" />
        {points ? (
          <svg className="relative h-full w-full" viewBox="0 0 680 240" role="img" aria-label="loss trend">
            <path d={points} fill="none" stroke="#63b8aa" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
            <path d={`${points} L680 240 L0 240 Z`} fill="url(#lossFill)" opacity=".28" />
            <defs>
              <linearGradient id="lossFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#63b8aa" />
                <stop offset="100%" stopColor="#63b8aa" stopOpacity="0" />
              </linearGradient>
            </defs>
          </svg>
        ) : (
          <EmptyState text="No loss series parsed for this run." />
        )}
      </div>
    </div>
  );
}

function MetricsTable({ metrics, title = "Recent metrics", detail = "latest Trainer dicts" }) {
  const rows = metrics.slice(-8).reverse();
  return (
    <div>
      <SectionTitle title={title} detail={detail} />
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Loss</th>
              <th>Accuracy</th>
              <th>Epoch</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((row, index) => (
                <tr key={`${row.source}-${index}`}>
                  <td className="mono">{fmt(row.loss ?? row.eval_loss)}</td>
                  <td className="mono">{fmt(row.mean_token_accuracy)}</td>
                  <td className="mono">{fmt(row.epoch)}</td>
                  <td className="truncate-cell">{row.source}</td>
                </tr>
              ))
            ) : (
              <EmptyRow cells={4} text="No parsed metrics yet." />
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ResultPreviews({ results }) {
  const rows = results
    .map((item) => ({ ...item, metricNumbers: metricNumbers(item.numbers) }))
    .filter((item) => item.metricNumbers.length)
    .slice(0, 5);

  return (
    <section className="panel">
      <SectionTitle title="Result previews" detail="metric-looking JSON keys" />
      <div className="grid gap-3">
        {rows.map((item) => (
          <div key={item.path} className="result-card rounded-2xl border border-white/10 bg-white/[.035] p-4">
            <p className="truncate font-mono text-xs text-[#b8c1b6]">{item.path}</p>
            <p className="mt-3 font-mono text-xs leading-6 text-[#f5f2ea]">{formatNumbers(item.metricNumbers)}</p>
          </div>
        ))}
        {!rows.length && <EmptyState text="No metric-like result numbers found." />}
      </div>
    </section>
  );
}

function BlockersPanel({ artifacts, issues }) {
  const missing = artifacts.filter((item) => item.status === "missing").length;
  const blockers = issues.filter((item) => item.severity === "blocker");
  const warnings = issues.filter((item) => item.severity !== "blocker");
  const blockerTotal = missing + blockers.length;

  return (
    <section className="panel">
      <SectionTitle title="Blockers" detail={`${blockerTotal} blockers: ${missing} missing, ${blockers.length} log; ${warnings.length} warnings`} />
      <div className="space-y-3">
        {artifacts.map((item) => (
          <div key={item.path} className="side-row">
            <Pill state={item.status}>{item.status}</Pill>
            <p className="mono break-words">{item.path}</p>
          </div>
        ))}
        {issues.slice(-6).reverse().map((item) => (
          <div key={`${item.source}-${item.line}`} className="side-row">
            <Pill state={item.severity}>{item.severity}</Pill>
            <p className="mono break-words">{item.line}</p>
            <p className="truncate text-xs text-[#8e988c]">{item.source}</p>
          </div>
        ))}
        {!artifacts.length && !issues.length && <EmptyState text="No blockers in this snapshot." />}
      </div>
    </section>
  );
}

function RunHistory({ runs, selectedRunId, setSelectedRunId }) {
  return (
    <section className="panel">
      <SectionTitle title="GPU use" detail="sacct last seven days" />
      <div className="space-y-2">
        {runs.slice(0, 14).map((run) => (
          <button
            key={run.job_id}
            className={`run-button run-button-compact ${selectedRunId === run.job_id ? "run-button-selected" : ""}`}
            onClick={() => setSelectedRunId(run.job_id)}
          >
            <span className="min-w-0">
              <span className="block truncate text-sm">{run.name}</span>
              <span className="mono text-xs text-[#8e988c]">{run.job_id} · {run.state} · {formatRunTime(run.start)}</span>
            </span>
            <span className="mono text-sm text-[#f5f2ea]">{formatGpuHours(run.gpu_hours)}</span>
          </button>
        ))}
        {!runs.length && <EmptyState text="No sacct runs in this snapshot." />}
      </div>
    </section>
  );
}

function StatusLine({ label, value }) {
  return (
    <div>
      <p className="font-mono text-[11px] uppercase text-[#767d75]">{label}</p>
      <p className="mt-1 break-words font-mono text-sm">{value}</p>
    </div>
  );
}

function SectionTitle({ title, detail }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-4">
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="font-mono text-xs text-[#8e988c]">{detail}</p>
    </div>
  );
}

function Pill({ state, children }) {
  return <span className={`pill ${stateTone(state)}`}>{children}</span>;
}

function EmptyRow({ cells, text }) {
  return (
    <tr>
      <td colSpan={cells} className="py-8 text-center text-[#8e988c]">{text}</td>
    </tr>
  );
}

function EmptyState({ text }) {
  return <div className="rounded-2xl border border-dashed border-white/15 p-8 text-center text-sm text-[#8e988c]">{text}</div>;
}

function LoadingShell() {
  return (
    <main className="grid min-h-[100dvh] place-items-center bg-[#151814] text-[#f5f2ea]">
      <div className="w-[min(520px,90vw)] rounded-[28px] border border-white/10 bg-white/[.055] p-8">
        <ClockIcon className="mb-6 h-7 w-7 animate-pulse text-[#63b8aa]" />
        <h1 className="text-3xl font-semibold">Loading observer</h1>
        <div className="mt-8 space-y-3">
          <div className="h-4 w-5/6 rounded-full bg-white/10 shimmer" />
          <div className="h-4 w-2/3 rounded-full bg-white/10 shimmer" />
          <div className="h-4 w-4/6 rounded-full bg-white/10 shimmer" />
        </div>
      </div>
    </main>
  );
}

function stateTone(value = "") {
  const state = String(value).toLowerCase();
  if (state.includes("run") || state.includes("complete") || state.includes("present") || state.includes("connected")) return "pill-good";
  if (state.includes("pend") || state.includes("missing") || state.includes("resource") || state.includes("warn")) return "pill-watch";
  if (state.includes("blocker") || state.includes("fail") || state.includes("error") || state.includes("cancel") || state.includes("timeout")) return "pill-bad";
  return "pill-neutral";
}

function queueLabel(jobs = []) {
  if (!jobs.length) return "empty";
  const running = jobs.filter((job) => /running/i.test(job.state)).length;
  const pending = jobs.filter((job) => /pending/i.test(job.state)).length;
  if (running && pending) return `${running} running, ${pending} pending`;
  if (running) return `${running} running`;
  if (pending) return `${pending} pending`;
  return `${jobs.length} queued`;
}

function groupMetricsByRun(metrics = []) {
  const byRunAndSource = {};

  for (const row of metrics) {
    const runId = runIdFromSource(row.source);
    if (!runId) continue;
    const source = row.source || "unknown";
    byRunAndSource[runId] ??= {};
    byRunAndSource[runId][source] ??= [];
    byRunAndSource[runId][source].push(row);
  }

  return Object.fromEntries(
    Object.entries(byRunAndSource).map(([runId, bySource]) => {
      const richestSource = Object.values(bySource).sort((a, b) => b.length - a.length)[0] ?? [];
      return [runId, richestSource];
    }),
  );
}

function dedupeIssues(errors = []) {
  const seen = new Map();
  for (const error of errors) {
    const line = String(error.line ?? "");
    const source = String(error.source ?? "");
    const runId = runIdFromSource(source);
    const key = `${runId ?? source}\n${line}`;
    if (!line || seen.has(key)) continue;
    seen.set(key, {
      ...error,
      source,
      line,
      run_id: runId,
      severity: classifyIssue(line),
    });
  }
  return Array.from(seen.values());
}

function classifyIssue(line) {
  if (/warning:.*retrying|oom.*retrying|auto batch detection hit oom|generation oom/i.test(line)) {
    return "warning";
  }
  if (/traceback|exception|assertionerror|valueerror|error|failed|cancelled|time limit|no such file|cuda out of memory|oom/i.test(line)) {
    return "blocker";
  }
  return "warning";
}

function groupByRun(items = []) {
  const grouped = {};
  for (const item of items) {
    const runId = item.run_id ?? runIdFromSource(item.source);
    if (!runId) continue;
    grouped[runId] ??= [];
    grouped[runId].push(item);
  }
  return grouped;
}

function groupProgressByRun(progress = []) {
  const seen = new Map();
  for (const item of progress) {
    const runId = runIdFromSource(item.source);
    if (!runId) continue;
    const key = `${item.source}\n${item.step}\n${item.total}\n${item.pct}`;
    if (seen.has(key)) continue;
    seen.set(key, { ...item, run_id: runId });
  }
  return groupByRun(Array.from(seen.values()));
}

function buildInspectableRuns(history = [], metricsByRun = {}, issuesByRun = {}, progressByRun = {}) {
  const seen = new Set();
  const rows = [];

  for (const run of history) {
    const jobId = String(run.job_id ?? "");
    const state = String(run.state ?? "");
    const metricCount = metricsByRun[jobId]?.length ?? 0;
    const issueCount = issuesByRun[jobId]?.length ?? 0;
    const progressCount = progressByRun[jobId]?.length ?? 0;
    const isWaiting = /pending|running/i.test(state);
    if (!jobId || seen.has(jobId) || (isWaiting && metricCount + issueCount + progressCount === 0)) continue;
    seen.add(jobId);
    rows.push({ ...run, job_id: jobId, metricCount, issueCount, progressCount });
  }

  const extraRunIds = new Set([...Object.keys(metricsByRun), ...Object.keys(issuesByRun), ...Object.keys(progressByRun)]);
  for (const jobId of extraRunIds) {
    if (seen.has(jobId)) continue;
    rows.push({
      job_id: jobId,
      name: `slurm-${jobId}`,
      state: "LOG ONLY",
      gpu_hours: undefined,
      elapsed_seconds: undefined,
      start: "",
      metricCount: metricsByRun[jobId]?.length ?? 0,
      issueCount: issuesByRun[jobId]?.length ?? 0,
      progressCount: progressByRun[jobId]?.length ?? 0,
    });
  }

  return rows.sort((a, b) => {
    const metricDelta = Number(b.metricCount > 0) - Number(a.metricCount > 0);
    if (metricDelta) return metricDelta;
    return runTime(b) - runTime(a);
  });
}

function runIdFromSource(source = "") {
  const match = String(source).match(/(?:^|\/)slurm-(\d+)\.out$/);
  return match?.[1] ?? null;
}

function runTime(run) {
  const value = run.end || run.start;
  const time = value ? new Date(value).getTime() : Number(run.job_id);
  return Number.isFinite(time) ? time : Number(run.job_id) || 0;
}

function lossValue(row) {
  if (typeof row.loss === "number") return row.loss;
  if (typeof row.eval_loss === "number") return row.eval_loss;
  return undefined;
}

function toneClass(tone) {
  return {
    good: "text-[#63b8aa]",
    bad: "text-[#f0a18f]",
    warn: "text-[#e3ba7d]",
    neutral: "text-[#f5f2ea]",
  }[tone] || "text-[#f5f2ea]";
}

function toPolyline(values, width, height) {
  if (values.length < 2) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values.map((value, index) => {
    const x = (index / (values.length - 1)) * width;
    const y = height - ((value - min) / span) * (height - 34) - 17;
    return `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
}

function fmt(value) {
  if (typeof value !== "number") return value ?? "none";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatGpuHours(value) {
  return typeof value === "number" ? `${fmt(value)}h` : "none";
}

function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds <= 0) return "none";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function formatRunTime(value) {
  if (!value || ["Unknown", "None", "N/A"].includes(value)) return value || "none";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toISOString().slice(0, 16).replace("T", " ");
}

function pct(value) {
  return typeof value === "number" ? `${value.toFixed(1)}%` : String(value);
}

function metricNumbers(items = []) {
  return items.filter(([key]) => {
    const name = String(key);
    if (/config|fewshot|n_tokens|n_sentences|label_counts|_count|counts/i.test(name)) return false;
    return /acc|accuracy|f1|bleu|chrf|exact|rouge|loss|perplex|stderr|wer|cer|score|token/i.test(name);
  });
}

function formatNumbers(items = []) {
  return items.slice(0, 4).map(([key, value]) => `${key}=${fmt(value)}`).join(" · ") || "no numeric summary";
}

function formatTimestamp(value) {
  if (!value) return "not loaded";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toISOString().replace("T", " ").replace(".000Z", "Z").replace(":00Z", "Z");
}

createRoot(document.getElementById("root")).render(<App />);
