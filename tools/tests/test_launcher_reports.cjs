// Run with: node --test tools/tests/test_launcher_reports.cjs
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Minimal DOM, as in test_launcher_crossval.cjs: runs the shipped app.js without its startup.
let registry = new Map();

class Element {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.events = {};
    this.className = '';
    this.text = '';
    this.dataset = {};
    this.style = {};
    this.classList = {
      add: c => { this.className += ' ' + c; },
      remove: c => { this.className = this.className.split(' ').filter(x => x !== c).join(' '); },
      contains: c => this.className.split(' ').includes(c),
      toggle: (c, on) => on ? this.classList.add(c) : this.classList.remove(c),
    };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(c => c.textContent).join(''); }
  setAttribute(key, value) { this.attrs[key] = String(value); if (key === 'id') registry.set(String(value), this); }
  addEventListener(event, fn) { this.events[event] = fn; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = []; this.text = ''; this.append(...children); }
  scrollIntoView() {}
  querySelectorAll(selector) {
    const tags = selector.split(',').map(s => s.trim().toUpperCase());
    return this.children.flatMap(c => [...(tags.includes(c.tagName) ? [c] : []), ...c.querySelectorAll(selector)]);
  }
  byClass(name) {
    return this.children.flatMap(c => [...(c.className.split(' ').includes(name) ? [c] : []), ...c.byClass(name)]);
  }
  byTag(tag) {
    return this.children.flatMap(c => [...(c.tagName === tag.toUpperCase() ? [c] : []), ...c.byTag(tag)]);
  }
}

function fixture(responses = {}) {
  const elements = registry = new Map();
  const document = {
    querySelector: () => ({ content: 'test-token' }),
    getElementById: id => {
      if (!elements.has(id)) elements.set(id, new Element());
      return elements.get(id);
    },
    createElement: tag => new Element(tag),
    createTextNode: text => { const el = new Element('#text'); el.textContent = text; return el; },
    querySelectorAll: () => [],
  };
  const calls = [];
  const fetch = async (url, init) => {
    calls.push({ url, body: init?.body ? JSON.parse(init.body) : null });
    const route = Object.keys(responses).find(prefix => url.startsWith(prefix));
    if (!route) throw new Error('offline');
    return { ok: true, status: 200, json: async () => responses[route] };
  };
  const context = vm.createContext({ document, Node: Element, console, setTimeout, clearTimeout, Date, fetch, URLSearchParams,
    confirm: () => true });
  const file = path.join(__dirname, '../../launcher/static/app.js');
  const source = fs.readFileSync(file, 'utf8');
  const init = source.lastIndexOf('\ninit().catch(');
  assert.ok(init > 0, 'Only omit application startup in the test fixture');
  vm.runInContext(source.slice(0, init), context, { filename: file });
  vm.runInContext('renderForm = () => {}; scheduleValidate = () => {}; updateJobDetail = () => {}; pollJobs = async () => {};', context);
  const api = vm.runInContext(`({ S, showResultTab, renderReportTab, reportContent, reportDialog, testedCheckpoint, reportFor })`, context);
  return { ...api, el: document.getElementById, calls };
}

const csv = (header, rows) => ({ header, rows: rows.map(r => r.map(String)), truncated: false });
const GLOBAL = csv(['metric', 'metric_key', 'unit', 'group', 'label', 'repetition', 'view', 'split', 'recordings', 'samples',
  'mae', 'rmse', 'bias', 'pearson', 'median_recording_mae', 'within_5bpm_pct'], [
  ['Heart rate', 'hr', 'bpm', 'All recordings', '', '', '', '', 4, 300, 8.5, 11, -2, 0.4, 7, 55],
  ['Heart rate', 'hr', 'bpm', 'test participants', '', '', '', 'test', 2, 150, 10.1, 12, -3, 0.3, 9, 50],
  ['RR / inter-beat interval', 'ibi', 'ms', 'All recordings', '', '', '', '', 4, 400, 96, 120, -10, 0.1, 90, ''],
]);
const MATRIX = csv(['metric_key', 'unit', 'label', 'all', 'test', 'training'], [
  ['hr', 'bpm', 'all', 8.5, 10.1, 7.0], ['hr', 'bpm', 'curl', 9.0, 11.0, 7.5], ['ibi', 'ms', 'all', 96, 100, 90],
]);
const RECORDINGS = csv(['experiment', 'participant', 'split', 'label', 'repetition', 'view', 'condition', 'recording',
  'duration_s', 'hr_mae_bpm', 'rmssd_mae_ms', 'ibi_mae_ms'], [
  ['jzc', 'jzc01', 'training', 'curl', 1, 'front', 'Curl 1', 'jzc01_curl1', 70, 7, 50, 90],
  ['jzc', 'jzc02', 'test', 'curl', 1, 'front', 'Curl 1', 'jzc02_curl1', 80, 11, 60, 100],
]);
const DETAIL = {
  path: 'C:/runs/x/report/Epoch16', scope: 'all',
  run_config: { title: 'RhythmFormer · toy_Epoch16', checkpoint: 'C:/runs/x/PreTrainedModels/toy_Epoch16.pth',
    recordings: ['jzc01_curl1', 'jzc02_curl1'], failures: {}, splits: { test: ['jzc02'], training: ['jzc01'] },
    selection: { labels: [], views: ['front'] } },
  tables: { global: GLOBAL, matrix: MATRIX, recordings: RECORDINGS },
  summary_figures: [{ name: 'summary/overview.png', path: 'C:/r/summary/overview.png' }],
  mosaics: [{ name: 'mosaics/hr_curl1.png', path: 'C:/r/mosaics/hr_curl1.png' }, { name: 'mosaics/rr_curl1.png', path: 'C:/r/mosaics/rr_curl1.png' }],
  participants: { jzc01: [{ name: 'participants/jzc01/pulse.png', path: 'p1' }, { name: 'participants/jzc01/hr.png', path: 'h1' }],
    jzc02: [{ name: 'participants/jzc02/hr.png', path: 'h2' }] },
  files: { 'REPORT.md': 'C:/r/REPORT.md' }, test_set: 'C:/r/test_set',
};
const REPORT = { name: 'Epoch16', path: 'C:/runs/x/report/Epoch16', checkpoint: 'C:/runs/x/PreTrainedModels/toy_Epoch16.pth',
  test_set: true, modified: 1, summary: { hr: { mae: 8.5, within_pct: 55, recordings: 4 }, ibi: { mae: 96 } },
  test_summary: { hr: { mae: 10.1, recordings: 2 } } };

const texts = node => node.byTag('img').map(img => img.attrs.alt);

test('the report content shows the tables, figures and participants', () => {
  const app = fixture();
  app.S.results.report = {};
  const parts = app.reportContent(REPORT, DETAIL, 'all');
  const page = { children: parts, textContent: parts.map(p => p.textContent).join('|') };
  assert.match(page.textContent, /MAE by movement and split/);
  assert.match(page.textContent, /Pooled results/);
  assert.match(page.textContent, /8\.50 bpm/);
  const images = parts.flatMap(texts);
  assert.ok(images.includes('mosaics/hr_curl1.png') && !images.includes('mosaics/rr_curl1.png'), 'HR mosaics first');
  // The participant's figures come in a fixed order: hr before pulse.
  const participant = images.filter(name => name.startsWith('participants/jzc01'));
  assert.deepEqual(Array.from(participant), ['participants/jzc01/hr.png', 'participants/jzc01/pulse.png']);
  // Clicking a recording row selects its participant.
  const rows = parts.flatMap(p => p.byTag('tr')).filter(tr => tr.events.click && tr.textContent.includes('jzc02'));
  rows[0].events.click();
  assert.equal(app.S.results.report.participant, 'jzc02');
  // Test participants only: no split matrix.
  const test = app.reportContent(REPORT, { ...DETAIL, scope: 'test' }, 'test');
  assert.doesNotMatch(test.map(p => p.textContent).join('|'), /MAE by movement and split/);
});

test('checkpoints offer a report and link the existing one', () => {
  const app = fixture();
  app.S.results.experiments = [{ path: 'C:/runs/x', jobs: ['job1'] }];
  app.S.results.selected = 'C:/runs/x';
  app.S.jobs = [{ id: 'job1', mode: 'train_and_test', status: 'succeeded', progress: { best_epoch: 16 }, epochs_total: 30,
    name: 'Train', backend: 'local', created: '2026-10-29T16:20:05' }];
  app.S.results.detail = { files: [
    { kind: 'checkpoint', name: 'PreTrainedModels/toy_Epoch2.pth', path: 'C:/runs/x/PreTrainedModels/toy_Epoch2.pth' },
    { kind: 'checkpoint', name: 'PreTrainedModels/toy_Epoch16.pth', path: 'C:/runs/x/PreTrainedModels/toy_Epoch16.pth' }],
    reports: [REPORT], cross_validation: false };
  app.S.results.report = {};
  app.showResultTab('Overview');
  const body = app.el('resultBody');
  const rows = body.byTag('tr');
  assert.deepEqual(rows.map(r => r.children[0].children[0].textContent), ['PreTrainedModels/toy_Epoch2.pth', 'PreTrainedModels/toy_Epoch16.pth']);
  assert.match(rows[1].textContent, /tested/);
  assert.match(rows[1].textContent, /View report/);
  assert.doesNotMatch(rows[0].textContent, /View report|tested/);
  assert.equal(app.testedCheckpoint(app.S.results.experiments[0]), '_Epoch16.pth');
});

test('the report dialog previews the request and queues the job', async () => {
  const app = fixture({
    '/api/reports/preview': { output: 'C:/runs/x/report/Epoch16', exists: false, backend: 'local', model: 'RhythmFormer',
      mode: 'train_and_test', default_labels: ['curl'], experiments: ['JZC'], views: ['front'], data_path: 'F:/data',
      data_available: true, cache: 'C:/cache', labels: { curl: 33, resting: 33 }, participants: ['jzc01'] },
    '/api/reports': { job: { id: 'job2' } },
  });
  app.S.results.experiments = [{ path: 'C:/runs/x', jobs: ['job1'] }];
  app.S.results.selected = 'C:/runs/x';
  app.S.jobs = [{ id: 'job1', mode: 'train_and_test', name: 'Train', backend: 'local', created: '2026-10-29T16:20:05' }];
  app.S.boot = { configs: [], settings: { default_backend: 'docker' } };
  await app.reportDialog('C:/runs/x/PreTrainedModels/toy_Epoch16.pth');
  await new Promise(resolve => setTimeout(resolve, 0));
  const preview = app.calls.find(c => c.url.startsWith('/api/reports/preview'));
  assert.equal(preview.body.config, 'launcher_data/jobs/job1/config.yaml');
  assert.equal(preview.body.backend, 'local', 'the backend of the job that trained the checkpoint');
  assert.equal(app.el('reportGo').disabled, false, 'enabled once the preview succeeded');
  const chips = app.el('modalBody').byClass('chip');
  assert.equal(chips.length, 2);
  chips[1].events.click();
  await app.el('reportGo').events.click();
  const queued = app.calls.find(c => c.url === '/api/reports');
  assert.equal(queued.body.labels, 'curl, resting', 'the config default plus the clicked cache label');
});
