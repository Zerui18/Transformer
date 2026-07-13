'use strict';

/* zlab experiment manager UI.
   Polls /api/status every 2s, live metrics for the running experiment every 5s.
   All mutations POST JSON {index, name} — the server refuses stale-index actions. */

const $ = (sel) => document.querySelector(sel);

const STATE_NAMES = { 0: 'queuing', 1: 'running', 2: 'completed', 3: 'stopped', 4: 'failed' };
const STATE_ICONS = { 0: '○', 1: '▶', 2: '✓', 3: '⏸', 4: '✕' };
const STATUS_POLL_MS = 2000;
const METRICS_POLL_MS = 5000;

let status = null;            // last /api/status payload
let serverTimeOffset = 0;     // server_time - Date.now()/1000 at receive time
let currentMetricsName = null;// which experiment the rendered charts belong to
const renderCache = {};       // section id -> last rendered HTML signature
const expandedErrs = new Set();

/* ---------------- helpers ---------------- */

function esc(value) {
	return String(value ?? '').replace(/[&<>"']/g, (c) => (
		{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function fmtBytes(n) {
	if (n == null) return '—';
	const units = ['B', 'KB', 'MB', 'GB', 'TB'];
	let i = 0;
	while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
	return `${n >= 100 ? n.toFixed(0) : n.toFixed(1)} ${units[i]}`;
}

function fmtDur(seconds) {
	if (seconds == null || !isFinite(seconds) || seconds < 0) return '—';
	const s = Math.floor(seconds);
	const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
	const pad = (x) => String(x).padStart(2, '0');
	return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

function fmtClock(ts) {
	if (ts == null) return '—';
	return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function fmtNum(v) {
	if (v == null || !isFinite(v)) return '—';
	const a = Math.abs(v);
	if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
	if (a >= 100) return v.toFixed(1);
	if (a >= 1) return v.toFixed(2);
	return v.toPrecision(3);
}

function serverNow() {
	return Date.now() / 1000 + serverTimeOffset;
}

function cssVar(name) {
	return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

async function api(path, body) {
	const opts = body === undefined ? {} : {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	};
	const res = await fetch(path, opts);
	const data = await res.json();
	if (!data.success) throw new Error(data.error || 'Request failed');
	return data;
}

function toast(message, kind = 'err') {
	const el = document.createElement('div');
	el.className = `toast ${kind}`;
	el.textContent = message;
	$('#toasts').appendChild(el);
	setTimeout(() => el.remove(), 5000);
}

function confirmDialog(title, text) {
	return new Promise((resolve) => {
		$('#confirm-title').textContent = title;
		$('#confirm-text').textContent = text;
		const modal = $('#confirm-modal');
		const yes = $('#confirm-yes');
		const done = (result) => { modal.close(); yes.onclick = null; modal.oncancel = null; resolve(result); };
		yes.onclick = () => done(true);
		modal.oncancel = () => done(false);
		modal.querySelector('[data-action="confirm-no"]').onclick = () => done(false);
		modal.showModal();
	});
}

async function act(promise, okMessage) {
	try {
		await promise;
		if (okMessage) toast(okMessage, 'ok');
		await refresh();
	} catch (e) {
		toast(e.message);
		refresh();
	}
}

/* ---------------- experiment row rendering ---------------- */

function chip(state) {
	const name = STATE_NAMES[state] ?? '?';
	return `<span class="chip chip-${name}"><i>${STATE_ICONS[state] ?? ''}</i>${name}</span>`;
}

function modelSummary(exp) {
	const model = exp.config?.model || {};
	const ia = model.init_args || {};
	const parts = [];
	if (model.class) parts.push(model.class);
	if (ia.attention_type) parts.push(ia.attention_type);
	if (ia.n_blocks != null) parts.push(`${ia.n_blocks} blk`);
	if (ia.n_heads != null) parts.push(`${ia.n_heads} h`);
	if (ia.emb_dim != null) parts.push(`d${ia.emb_dim}`);
	const epochs = exp.config?.trainer?.max_epochs;
	if (epochs != null) parts.push(`${epochs} ep`);
	return parts.join(' · ');
}

function runtimeText(exp) {
	if (exp.launched_at == null) return '';
	const end = exp.finished_at ?? serverNow();
	return fmtDur(end - exp.launched_at);
}

function expRow(exp, bucket, index, buttons, extra = '') {
	const pos = bucket === 'queued' ? `<span class="exp-pos">${index + 1}</span>` : '';
	return `<div class="exp-row" data-bucket="${bucket}" data-index="${index}" data-exp-name="${esc(exp.name)}">
		${pos}
		${chip(exp.state)}
		<div class="exp-main">
			<div class="exp-name" data-action="details" data-bucket="${bucket}" data-key="${index}" data-exp-name="${esc(exp.name)}" title="${esc(exp.name)}">${esc(exp.name)}</div>
			<div class="exp-sub">${esc(modelSummary(exp))}</div>
			${extra}
		</div>
		<div class="exp-actions">${buttons}</div>
	</div>`;
}

function iconBtn(action, index, name, label, title, extraCls = '') {
	return `<button class="btn btn-ghost btn-sm btn-icon ${extraCls}" data-action="${action}" data-index="${index}" data-name="${esc(name)}" title="${title}">${label}</button>`;
}

/* ---------------- section rendering ---------------- */

function setHtml(id, html) {
	if (renderCache[id] === html) return;
	renderCache[id] = html;
	$(`#${id}`).innerHTML = html;
}

function meterTile(label, valueHtml, pct, sub, { severity = true, color = null } = {}) {
	let fill = color || cssVar('--s1');
	if (severity && pct != null) {
		if (pct >= 90) fill = cssVar('--crit');
		else if (pct >= 75) fill = cssVar('--warn');
	}
	const meter = pct == null ? '' :
		`<div class="meter" style="background: color-mix(in srgb, ${fill} 15%, var(--surface))">
			<i style="width:${Math.max(0, Math.min(100, pct)).toFixed(1)}%; background:${fill}"></i>
		</div>`;
	return `<div class="tile">
		<div class="tile-label">${esc(label)}</div>
		<div class="tile-value">${valueHtml}</div>
		${meter}
		${sub ? `<div class="tile-sub">${sub}</div>` : ''}
	</div>`;
}

function renderMachine() {
	const m = status.machine || {};
	const cpu = m.cpu || {};
	const tiles = [];
	tiles.push(meterTile('CPU', `${cpu.cpu_percent != null ? cpu.cpu_percent.toFixed(0) : '—'}<small>%</small>`,
		cpu.cpu_percent,
		`${cpu.cpu_count ?? '—'} cores${cpu.load_avg ? ` · load ${cpu.load_avg.map((x) => x.toFixed(1)).join(' / ')}` : ''}`));
	tiles.push(meterTile('Memory', `${cpu.mem_percent != null ? cpu.mem_percent.toFixed(0) : '—'}<small>%</small>`,
		cpu.mem_percent,
		cpu.mem_total ? `${fmtBytes(cpu.mem_used)} of ${fmtBytes(cpu.mem_total)}` : 'psutil not installed'));
	const disk = m.disk || {};
	const diskPct = disk.total ? (disk.used / disk.total) * 100 : null;
	tiles.push(meterTile('Disk (experiments)', `${disk.free != null ? fmtBytes(disk.free) : '—'}<small> free</small>`,
		diskPct, disk.total ? `${fmtBytes(disk.used)} of ${fmtBytes(disk.total)} used` : ''));
	const gpus = m.gpus || [];
	if (gpus.length === 0) {
		tiles.push(meterTile('GPU', '<small>none detected</small>', null, 'no NVIDIA GPU / driver on this machine'));
	}
	for (const gpu of gpus) {
		// high utilisation is healthy on a training box — keep the accent hue, no severity ramp
		tiles.push(meterTile(`GPU ${gpu.index} · ${gpu.name ?? ''}`,
			`${gpu.util_percent != null ? Number(gpu.util_percent).toFixed(0) : '—'}<small>% util</small>`,
			gpu.util_percent, [
				gpu.temperature != null ? `${gpu.temperature.toFixed ? gpu.temperature.toFixed(0) : gpu.temperature}°C` : null,
				gpu.power_draw != null ? `${Number(gpu.power_draw).toFixed(0)}${gpu.power_limit ? ` / ${Number(gpu.power_limit).toFixed(0)}` : ''} W` : null,
			].filter(Boolean).join(' · '), { severity: false }));
		const memPct = gpu.mem_total ? (gpu.mem_used / gpu.mem_total) * 100 : null;
		tiles.push(meterTile(`GPU ${gpu.index} memory`,
			`${memPct != null ? memPct.toFixed(0) : '—'}<small>%</small>`,
			memPct, gpu.mem_total ? `${fmtBytes(gpu.mem_used)} of ${fmtBytes(gpu.mem_total)}` : ''));
	}
	setHtml('machine', tiles.join(''));
	$('#host-label').textContent = m.hostname ? `${m.hostname} · ${m.platform ?? ''}` : '';
}

function renderCurrent() {
	const exp = status.current;
	if (!exp) {
		currentMetricsName = null;
		setHtml('current', `<div class="current-idle">No experiment running — the next queued experiment starts automatically.</div>`);
		return;
	}
	// the shell contains only slowly-changing content; volatile progress values are
	// patched into it below, so buttons are not rebuilt mid-click every poll
	const shell = `
		<div class="current-head">
			${chip(exp.state)}
			<span class="current-name" data-action="details" data-bucket="current" data-key="0" data-exp-name="${esc(exp.name)}">${esc(exp.name)}</span>
			<span class="current-spacer"></span>
			<button class="btn btn-ghost btn-sm" data-action="clone" data-bucket="current" data-key="0" data-exp-name="${esc(exp.name)}">Clone</button>
			<button class="btn btn-danger btn-sm" data-action="stop-current">Stop</button>
		</div>
		<div class="current-meta">
			<span>${esc(modelSummary(exp))}</span>
			<span>elapsed <span data-elapsed="${exp.launched_at ?? ''}">—</span></span>
			<span title="${esc(exp.directory)}">${esc(exp.directory)}</span>
		</div>
		${exp.state !== 0 ? `
		<div class="progress-wrap">
			<div class="progress-info"><span id="prog-left"></span><span id="prog-right"></span></div>
			<div class="progress"><i id="prog-bar"></i></div>
		</div>` : '<div class="current-meta"><span>initializing (building dataloaders / model)…</span></div>'}
		<div class="charts" id="current-charts"></div>`;
	if (renderCache['current'] !== shell) {
		const oldCharts = $('#current-charts');
		const keepCharts = oldCharts && currentMetricsName === exp.name ? oldCharts : null;
		renderCache['current'] = shell;
		$('#current').innerHTML = shell;
		if (keepCharts) $('#current-charts').replaceWith(keepCharts);
	}
	// patch the volatile progress values without rebuilding the DOM
	const prog = exp.progress || {};
	const maxEpochs = exp.config?.trainer?.max_epochs;
	const epochFrac = prog.batches_per_epoch > 0 ? prog.batch / prog.batches_per_epoch : 0;
	const overallPct = maxEpochs ? Math.min(100, ((prog.epoch + epochFrac) / maxEpochs) * 100) : null;
	if ($('#prog-left')) {
		$('#prog-left').textContent = `epoch ${prog.epoch + 1}${maxEpochs ? ` / ${maxEpochs}` : ''} · batch ${prog.batch}${prog.batches_per_epoch ? ` / ${prog.batches_per_epoch}` : ''}`;
		$('#prog-right').textContent = `step ${(prog.global_step ?? 0).toLocaleString()}${overallPct != null ? ` · ${overallPct.toFixed(0)}%` : ''}`;
		$('#prog-bar').style.width = `${overallPct ?? 0}%`;
	}
	if (currentMetricsName !== exp.name) {
		currentMetricsName = exp.name;
		refreshCurrentMetrics();
	}
}

function renderQueue() {
	const queue = status.queued || [];
	$('#queue-count').textContent = queue.length ? `(${queue.length})` : '';
	if (!queue.length) { setHtml('queue', '<div class="empty">Queue is empty.</div>'); return; }
	const rows = queue.map((exp, i) => expRow(exp, 'queued', i, [
		iconBtn('queue-top', i, exp.name, '⤒', 'move to front'),
		iconBtn('queue-up', i, exp.name, '↑', 'move up'),
		iconBtn('queue-down', i, exp.name, '↓', 'move down'),
		iconBtn('clone-row', i, exp.name, '⧉', 'clone into a new experiment'),
		iconBtn('queue-stop', i, exp.name, '✕', 'remove from queue (to stopped)', 'btn-danger-text'),
	].join('')));
	setHtml('queue', rows.join(''));
}

function renderStopped() {
	const list = status.stopped || [];
	$('#stopped-count').textContent = list.length ? `(${list.length})` : '';
	if (!list.length) { setHtml('stopped', '<div class="empty">Nothing stopped.</div>'); return; }
	setHtml('stopped', list.map((exp, i) => expRow(exp, 'stopped', i, [
		iconBtn('stopped-enqueue', i, exp.name, '↻', 'requeue'),
		iconBtn('clone-row', i, exp.name, '⧉', 'clone into a new experiment'),
		iconBtn('stopped-remove', i, exp.name, '🗑', 'remove and DELETE its folder', 'btn-danger-text'),
	].join(''), exp.launched_at ? `<div class="exp-sub">ran ${runtimeText(exp)}</div>` : '')).join(''));
}

function renderFailed() {
	const list = status.failed || [];
	$('#failed-count').textContent = list.length ? `(${list.length})` : '';
	if (!list.length) { setHtml('failed', '<div class="empty">No failures.</div>'); return; }
	setHtml('failed', list.map((exp, i) => {
		const errId = `${exp.name}`;
		const open = expandedErrs.has(errId);
		const err = exp.err_buffer ? `<div class="exp-err ${open ? 'open' : ''}" data-action="toggle-err" data-err-id="${esc(errId)}" title="click to expand">${esc(exp.err_buffer)}</div>` : '';
		return expRow(exp, 'failed', i, [
			iconBtn('failed-enqueue', i, exp.name, '↻', 'retry (requeue)'),
			iconBtn('clone-row', i, exp.name, '⧉', 'clone into a new experiment'),
			iconBtn('failed-remove', i, exp.name, '🗑', 'remove and DELETE its folder', 'btn-danger-text'),
		].join(''), err);
	}).join(''));
}

function renderCompleted() {
	const list = status.completed || [];
	$('#completed-count').textContent = list.length ? `(${list.length})` : '';
	if (!list.length) { setHtml('completed', '<div class="empty">Nothing completed yet.</div>'); return; }
	setHtml('completed', list.map((exp, i) => expRow(exp, 'completed', i, [
		iconBtn('clone-row', i, exp.name, '⧉', 'clone into a new experiment'),
		iconBtn('completed-clear', i, exp.name, '✕', 'clear from list (folder stays on disk)'),
	].join(''), `<div class="exp-sub">ran ${runtimeText(exp)} · finished ${fmtClock(exp.finished_at)}</div>`)).join(''));
}

function renderDisk() {
	const list = status.disk || [];
	$('#disk-panel').hidden = list.length === 0;
	$('#disk-count').textContent = list.length ? `(${list.length})` : '';
	if (!list.length) return;
	setHtml('disk', list.map((folder) => `<div class="exp-row">
		<div class="exp-main">
			<div class="exp-name" data-action="details" data-bucket="disk" data-key="${esc(folder.name)}">${esc(folder.name)}</div>
			<div class="exp-sub">modified ${new Date(folder.mtime * 1000).toLocaleString()}</div>
		</div>
		<div class="exp-actions">
			<button class="btn btn-ghost btn-sm" data-action="disk-resume" data-name="${esc(folder.name)}" title="requeue this folder, resuming from its last checkpoint if present">Resume</button>
			<button class="btn btn-ghost btn-sm btn-icon" data-action="clone" data-bucket="disk" data-key="${esc(folder.name)}" title="clone into a new experiment">⧉</button>
			<button class="btn btn-ghost btn-sm btn-icon btn-danger-text" data-action="disk-delete" data-name="${esc(folder.name)}" title="DELETE this folder from disk">🗑</button>
		</div>
	</div>`).join(''));
}

function renderAll() {
	renderMachine();
	renderCurrent();
	renderQueue();
	renderStopped();
	renderFailed();
	renderCompleted();
	renderDisk();
}

/* ---------------- polling ---------------- */

let refreshSeq = 0;
async function refresh() {
	const seq = ++refreshSeq;
	try {
		const data = await api('/api/status');
		if (seq !== refreshSeq) return; // a newer poll already landed — don't regress state
		serverTimeOffset = data.server_time - Date.now() / 1000;
		status = data;
		$('#conn').className = 'conn ok';
		$('#conn-label').textContent = 'live';
		renderAll();
	} catch (e) {
		if (seq !== refreshSeq) return;
		$('#conn').className = 'conn down';
		$('#conn-label').textContent = 'server unreachable';
	}
}

async function refreshCurrentMetrics() {
	const name = status?.current?.name;
	if (!name) return;
	const box = $('#current-charts');
	if (!box) return;
	try {
		const data = await api(`/api/experiments/current/0/metrics?max_points=400&name=${encodeURIComponent(name)}`);
		if (status?.current?.name !== name || !document.contains(box)) return; // slot advanced mid-flight
		renderMetricCharts(box, data.metrics, { emptyText: 'No metrics logged yet.' });
	} catch (e) { /* current may have just finished — next status poll handles it */ }
}

let statusTimer = null, metricsTimer = null, tickTimer = null;
function startTimers() {
	if (statusTimer) return;
	statusTimer = setInterval(refresh, STATUS_POLL_MS);
	metricsTimer = setInterval(refreshCurrentMetrics, METRICS_POLL_MS);
	tickTimer = setInterval(() => {
		document.querySelectorAll('[data-elapsed]').forEach((el) => {
			const launched = parseFloat(el.dataset.elapsed);
			if (isFinite(launched)) el.textContent = fmtDur(serverNow() - launched);
		});
	}, 1000);
}
function stopTimers() {
	clearInterval(statusTimer); clearInterval(metricsTimer); clearInterval(tickTimer);
	statusTimer = metricsTimer = tickTimer = null;
}
document.addEventListener('visibilitychange', () => {
	if (document.hidden) stopTimers();
	else { startTimers(); refresh(); refreshCurrentMetrics(); }
});

/* ---------------- charts ---------------- */

const SKIP_TAGS = new Set(['epoch', 'hp_metric']);

function groupMetrics(metrics) {
	const tags = Object.keys(metrics).filter((t) => !SKIP_TAGS.has(t)).sort();
	const groups = [];
	const grab = (predicate, title) => {
		const members = tags.filter(predicate);
		if (members.length) groups.push({ title, tags: members });
		return new Set(members);
	};
	const used = new Set();
	for (const t of grab((t) => /loss/i.test(t), 'loss')) used.add(t);
	for (const t of grab((t) => /^bleu/i.test(t) && !used.has(t), 'BLEU')) used.add(t);
	for (const t of tags) if (!used.has(t)) groups.push({ title: t, tags: [t] });
	return groups;
}

function renderMetricCharts(container, metrics, { emptyText = 'No metrics.' } = {}) {
	const groups = groupMetrics(metrics);
	if (!groups.length) {
		container.innerHTML = `<div class="chart-empty">${esc(emptyText)}</div>`;
		return;
	}
	container.innerHTML = groups.map((g, i) => `<div class="chart-box"><div class="chart-title">${esc(g.title)}</div><div class="chart-slot" data-slot="${i}"></div></div>`).join('');
	const seriesColors = ['--s1', '--s2', '--s3', '--s5', '--s6', '--s4'];
	groups.forEach((group, i) => {
		const series = group.tags.map((tag, j) => ({
			name: tag,
			color: cssVar(seriesColors[j % seriesColors.length]),
			steps: metrics[tag].steps,
			values: metrics[tag].values,
		}));
		drawLineChart(container.querySelector(`[data-slot="${i}"]`), series);
	});
}

function niceTicks(min, max, count) {
	const span = max - min || Math.abs(max) || 1;
	const step0 = span / Math.max(1, count);
	const mag = Math.pow(10, Math.floor(Math.log10(step0)));
	const norm = step0 / mag;
	const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
	const start = Math.ceil(min / step) * step;
	const ticks = [];
	for (let v = start; v <= max + step * 1e-6; v += step) ticks.push(v);
	return ticks;
}

function drawLineChart(el, series) {
	series = series.filter((s) => s.steps.length > 0);
	if (!series.length) { el.innerHTML = '<div class="chart-empty">no data</div>'; return; }
	const W = Math.max(280, el.clientWidth || el.parentElement.clientWidth || 420);
	const H = 170;
	const M = { l: 46, r: 52, t: 10, b: 22 };
	const pw = W - M.l - M.r, ph = H - M.t - M.b;
	let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
	for (const s of series) {
		xmin = Math.min(xmin, s.steps[0]); xmax = Math.max(xmax, s.steps[s.steps.length - 1]);
		for (const v of s.values) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }
	}
	if (xmin === xmax) { xmin -= 1; xmax += 1; }
	const ypad = (ymax - ymin) * 0.08 || Math.abs(ymax) * 0.1 || 1;
	ymin -= ypad; ymax += ypad;
	const X = (v) => M.l + ((v - xmin) / (xmax - xmin)) * pw;
	const Y = (v) => M.t + ph - ((v - ymin) / (ymax - ymin)) * ph;

	const grid = cssVar('--grid'), baseline = cssVar('--baseline'), surface = cssVar('--surface');
	let svg = '';
	// hairline horizontal gridlines + y tick labels
	for (const t of niceTicks(ymin, ymax, 4)) {
		svg += `<line x1="${M.l}" y1="${Y(t)}" x2="${M.l + pw}" y2="${Y(t)}" stroke="${grid}" stroke-width="1"/>`;
		svg += `<text x="${M.l - 6}" y="${Y(t) + 3.5}" text-anchor="end">${fmtNum(t)}</text>`;
	}
	// baseline + x ticks
	svg += `<line x1="${M.l}" y1="${M.t + ph}" x2="${M.l + pw}" y2="${M.t + ph}" stroke="${baseline}" stroke-width="1"/>`;
	for (const t of niceTicks(xmin, xmax, 4)) {
		const label = t >= 1000 ? `${(t / 1000).toFixed(t >= 10000 ? 0 : 1)}k`
			: (Number.isInteger(t) ? t : fmtNum(t)); // avoid float-accumulation garbage like 1.2000000000000002
		svg += `<text x="${X(t)}" y="${M.t + ph + 14}" text-anchor="middle">${label}</text>`;
	}
	// area wash for a single series
	if (series.length === 1) {
		const s = series[0];
		const pts = s.steps.map((x, i) => `${X(x).toFixed(1)},${Y(s.values[i]).toFixed(1)}`).join(' ');
		svg += `<polygon points="${X(s.steps[0]).toFixed(1)},${M.t + ph} ${pts} ${X(s.steps[s.steps.length - 1]).toFixed(1)},${M.t + ph}" fill="${s.color}" fill-opacity="0.1"/>`;
	}
	// 2px lines with round joins + end marker with surface ring
	for (const s of series) {
		const pts = s.steps.map((x, i) => `${X(x).toFixed(1)},${Y(s.values[i]).toFixed(1)}`).join(' ');
		svg += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
		const lx = X(s.steps[s.steps.length - 1]), ly = Y(s.values[s.values.length - 1]);
		svg += `<circle cx="${lx}" cy="${ly}" r="4" fill="${s.color}" stroke="${surface}" stroke-width="2"/>`;
	}
	// selective direct end labels (ink, not series-colored); drop colliding ones — legend + tooltip carry the rest
	const ends = series.map((s) => ({ y: Y(s.values[s.values.length - 1]), v: s.values[s.values.length - 1] }))
		.sort((a, b) => a.y - b.y);
	let lastLabelY = -Infinity;
	for (const end of ends) {
		if (end.y - lastLabelY < 13) continue;
		lastLabelY = end.y;
		svg += `<text class="endlabel" x="${M.l + pw + 8}" y="${end.y + 3.5}">${fmtNum(end.v)}</text>`;
	}
	// crosshair (populated on hover)
	svg += `<line class="xhair" x1="0" y1="${M.t}" x2="0" y2="${M.t + ph}" stroke="${baseline}" stroke-width="1" visibility="hidden"/>`;
	svg += `<rect class="hover-target" x="${M.l}" y="${M.t}" width="${pw}" height="${ph}" fill="transparent"/>`;

	const legend = series.length >= 2
		? `<div class="legend">${series.map((s) => `<span class="legend-item"><span class="legend-dot" style="background:${s.color}"></span>${esc(s.name)}</span>`).join('')}</div>`
		: '';
	el.innerHTML = `${legend}<div class="chart-wrap">
		<svg class="chart-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${svg}</svg>
		<div class="chart-tip"></div>
	</div>`;

	// hover layer: crosshair + shared tooltip at nearest step
	const wrap = el.querySelector('.chart-wrap');
	const tip = el.querySelector('.chart-tip');
	const xhair = el.querySelector('.xhair');
	const target = el.querySelector('.hover-target');
	const nearest = (s, xv) => {
		let lo = 0, hi = s.steps.length - 1;
		while (hi - lo > 1) { const mid = (lo + hi) >> 1; (s.steps[mid] < xv) ? lo = mid : hi = mid; }
		return (xv - s.steps[lo] <= s.steps[hi] - xv) ? lo : hi;
	};
	target.addEventListener('mousemove', (ev) => {
		const box = wrap.getBoundingClientRect();
		const scale = W / box.width;
		const xv = xmin + (((ev.clientX - box.left) * scale - M.l) / pw) * (xmax - xmin);
		const rows = series.map((s) => {
			const i = nearest(s, xv);
			return { s, step: s.steps[i], value: s.values[i] };
		});
		const anchor = rows[0];
		xhair.setAttribute('x1', X(anchor.step)); xhair.setAttribute('x2', X(anchor.step));
		xhair.setAttribute('visibility', 'visible');
		tip.innerHTML = `<div class="tip-x">step ${anchor.step.toLocaleString()}</div>` +
			rows.map((r) => `<div class="tip-row"><span class="legend-dot" style="background:${r.s.color}"></span>${esc(r.s.name)} <b>${fmtNum(r.value)}</b></div>`).join('');
		tip.style.display = 'block';
		const px = X(anchor.step) / scale;
		tip.style.left = `${px + (px > box.width * 0.6 ? -tip.offsetWidth - 12 : 12)}px`;
		tip.style.top = `${Math.max(0, (ev.clientY - box.top) - tip.offsetHeight - 8)}px`;
	});
	target.addEventListener('mouseleave', () => {
		tip.style.display = 'none';
		xhair.setAttribute('visibility', 'hidden');
	});
}

/* ---------------- create / clone modal ---------------- */

async function openCreate(prefill = null) {
	$('#create-error').hidden = true;
	$('#create-title').textContent = prefill ? `New experiment (from ${prefill.source})` : 'New experiment';
	$('#create-name').value = prefill?.name ?? '';
	$('#yaml-model').value = prefill?.model ?? $('#yaml-model').value;
	$('#yaml-dls').value = prefill?.dls ?? $('#yaml-dls').value;
	$('#yaml-trainer').value = prefill?.trainer ?? $('#yaml-trainer').value;
	$('#resume-dir').value = '';
	$('#resume-ckpt').value = '';
	$('#create-modal').showModal();
	try {
		const data = await api('/api/templates');
		const select = $('#template-select');
		const current = select.value;
		select.innerHTML = '<option value="">—</option>' + data.templates.map((t) => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
		select.value = current;
	} catch (e) { /* templates are a convenience — ignore */ }
}

function nameGuardQuery(name) {
	return name ? `?name=${encodeURIComponent(name)}` : '';
}

async function clone(bucket, key, name) {
	try {
		const data = await api(`/api/experiments/${encodeURIComponent(bucket)}/${encodeURIComponent(key)}/yaml${nameGuardQuery(name)}`);
		openCreate({ source: data.name, name: `${data.name}-v2`, model: data.model, dls: data.dls, trainer: data.trainer });
	} catch (e) { toast(e.message); }
}

async function submitCreate(ev) {
	ev.preventDefault();
	const errBox = $('#create-error');
	errBox.hidden = true;
	try {
		await api('/api/queue/create', {
			name: $('#create-name').value.trim(),
			model: $('#yaml-model').value,
			dls: $('#yaml-dls').value,
			trainer: $('#yaml-trainer').value,
			resume_from_directory: $('#resume-dir').value.trim(),
			resume_from_checkpoint: $('#resume-ckpt').value.trim(),
		});
		$('#create-modal').close();
		toast('Experiment enqueued.', 'ok');
		refresh();
	} catch (e) {
		errBox.textContent = e.message;
		errBox.hidden = false;
	}
}

/* ---------------- details modal ---------------- */

async function openDetails(bucket, key, name) {
	const modal = $('#details-modal');
	const body = $('#details-body');
	$('#details-title').textContent = 'Loading…';
	body.innerHTML = '';
	modal.showModal();
	try {
		const guard = nameGuardQuery(name);
		const [yamlData, ckptData, metricsData] = await Promise.all([
			api(`/api/experiments/${encodeURIComponent(bucket)}/${encodeURIComponent(key)}/yaml${guard}`),
			api(`/api/experiments/${encodeURIComponent(bucket)}/${encodeURIComponent(key)}/checkpoints${guard}`),
			api(`/api/experiments/${encodeURIComponent(bucket)}/${encodeURIComponent(key)}/metrics${guard ? guard + '&' : '?'}max_points=500`),
		]);
		$('#details-title').textContent = yamlData.name;
		const ckpts = ckptData.checkpoints;
		body.innerHTML = `
			<div class="details-section"><h3>Metrics</h3><div class="charts" id="details-charts"></div></div>
			<div class="details-section"><h3>Checkpoints ${ckpts.length ? `(${ckpts.length})` : ''}</h3>
				${ckpts.length ? `<table class="ckpt"><tr><th>file</th><th>size</th><th>modified</th></tr>
					${ckpts.map((c) => `<tr><td>${esc(c.name)}</td><td>${fmtBytes(c.size)}</td><td>${new Date(c.mtime * 1000).toLocaleString()}</td></tr>`).join('')}
				</table>` : '<div class="empty">No checkpoints.</div>'}
			</div>
			<div class="details-section"><h3>Config</h3>
				<div class="details-yaml-grid">
					${['model', 'dls', 'trainer'].map((k) => `<div><div class="yaml-name">${k}.yaml</div><pre>${esc(yamlData[k])}</pre></div>`).join('')}
				</div>
			</div>`;
		renderMetricCharts($('#details-charts'), metricsData.metrics, { emptyText: 'No metrics recorded.' });
	} catch (e) {
		$('#details-title').textContent = 'Error';
		body.innerHTML = `<div class="form-error">${esc(e.message)}</div>`;
	}
}

/* ---------------- actions ---------------- */

document.addEventListener('click', async (ev) => {
	const btn = ev.target.closest('[data-action]');
	if (!btn) return;
	const action = btn.dataset.action;
	const index = btn.dataset.index != null ? parseInt(btn.dataset.index, 10) : null;
	const name = btn.dataset.name;

	switch (action) {
		case 'open-create': return openCreate();
		case 'close-create': return $('#create-modal').close();
		case 'close-details': return $('#details-modal').close();
		case 'details': return openDetails(btn.dataset.bucket, btn.dataset.key, btn.dataset.expName);
		case 'clone': return clone(btn.dataset.bucket, btn.dataset.key, btn.dataset.expName);
		case 'clone-row': {
			const row = btn.closest('.exp-row');
			return clone(row.dataset.bucket, index, row.dataset.expName);
		}
		case 'toggle-err': {
			const id = btn.dataset.errId;
			expandedErrs.has(id) ? expandedErrs.delete(id) : expandedErrs.add(id);
			btn.classList.toggle('open');
			return;
		}
		case 'stop-current': {
			const targetName = status?.current?.name;
			if (!targetName) return;
			if (!await confirmDialog('Stop current experiment?', `"${targetName}" stops at the end of the batch and moves to Stopped — it can be requeued later.`)) return;
			// send the confirmed name so the server refuses if the slot advanced meanwhile
			return act(api('/api/current/stop', { name: targetName }), 'Stop requested.');
		}
		case 'queue-top': return act(api('/api/queue/move', { src: index, dst: 0, name }));
		case 'queue-up': return act(api('/api/queue/move', { src: index, dst: Math.max(0, index - 1), name }));
		case 'queue-down': return act(api('/api/queue/move', { src: index, dst: Math.min((status?.queued?.length ?? 1) - 1, index + 1), name }));
		case 'queue-stop': return act(api('/api/queue/stop', { index, name }));
		case 'queue-stop-all': {
			if (!await confirmDialog('Stop all queued?', 'All queued experiments move to Stopped (they can be requeued).')) return;
			return act(api('/api/queue/stop_all', {}));
		}
		case 'stopped-enqueue': return act(api('/api/stopped/enqueue', { index, name }));
		case 'stopped-enqueue-all': return act(api('/api/stopped/enqueue_all', {}));
		case 'stopped-remove': {
			if (!await confirmDialog('Remove stopped experiment?', `"${name}" is removed and its folder (checkpoints, logs) is DELETED from disk.`)) return;
			return act(api('/api/stopped/remove', { index, name }));
		}
		case 'stopped-remove-all': {
			if (!await confirmDialog('Remove ALL stopped experiments?', 'Their folders (checkpoints, logs) are DELETED from disk.')) return;
			return act(api('/api/stopped/remove_all', {}));
		}
		case 'failed-enqueue': return act(api('/api/failed/enqueue', { index, name }));
		case 'failed-enqueue-all': return act(api('/api/failed/enqueue_all', {}));
		case 'failed-remove': {
			if (!await confirmDialog('Remove failed experiment?', `"${name}" is removed and its folder is DELETED from disk.`)) return;
			return act(api('/api/failed/remove', { index, name }));
		}
		case 'failed-remove-all': {
			if (!await confirmDialog('Remove ALL failed experiments?', 'Their folders are DELETED from disk.')) return;
			return act(api('/api/failed/remove_all', {}));
		}
		case 'completed-clear': return act(api('/api/completed/clear', { index, name }));
		case 'completed-clear-all': return act(api('/api/completed/clear_all', {}));
		case 'disk-resume': return act(api('/api/disk/resume', { name }), `Resuming ${name}.`);
		case 'disk-delete': {
			if (!await confirmDialog('Delete folder from disk?', `"${name}" and everything in it (checkpoints, logs) is permanently DELETED.`)) return;
			return act(api('/api/disk/delete', { name }));
		}
	}
});

$('#create-form').addEventListener('submit', submitCreate);
$('#template-select').addEventListener('change', async (ev) => {
	if (!ev.target.value) return;
	try {
		const data = await api(`/api/templates/${encodeURIComponent(ev.target.value)}`);
		$('#yaml-model').value = data.model;
		$('#yaml-dls').value = data.dls;
		$('#yaml-trainer').value = data.trainer;
	} catch (e) { toast(e.message); }
});

/* ---------------- boot ---------------- */

startTimers();
refresh();
