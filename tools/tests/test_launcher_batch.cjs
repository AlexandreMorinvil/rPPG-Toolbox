// Run with: node --test tools/tests/test_launcher_batch.cjs
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Minimal DOM, as in test_launcher_crossval.cjs: runs the shipped app.js without its startup.
// Elements created with an id are found by getElementById, as in a browser.
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
  querySelectorAll() { return []; }
  all(predicate) { return this.children.flatMap(c => [...(predicate(c) ? [c] : []), ...c.all(predicate)]); }
  byClass(name) { return this.all(c => c.className.split(' ').includes(name)); }
  byTag(tag) { return this.all(c => c.tagName === tag.toUpperCase()); }
}

function fixture() {
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
  const fetch = async () => { throw new Error('offline'); };
  const context = vm.createContext({ document, Node: Element, console, setTimeout, clearTimeout, Date, fetch, confirm: () => true });
  const file = path.join(__dirname, '../../launcher/static/app.js');
  const source = fs.readFileSync(file, 'utf8');
  const init = source.lastIndexOf('\ninit().catch(');
  assert.ok(init > 0, 'Only omit application startup in the test fixture');
  vm.runInContext(source.slice(0, init), context, { filename: file });
  vm.runInContext('renderForm = () => {}; scheduleValidate = () => {}; updateJobDetail = () => {};', context);
  const api = vm.runInContext(`({ S, renderJobList, renderBatchSelect, renderBatchChecks, updateBatchDetail, batchProgress, headlineMae })`, context);
  api.S.boot = { schema: { mode_labels: { train_and_test: 'Training', unsupervised_method: 'Unsupervised' } }, platform: 'nt',
    settings: { default_backend: 'docker', max_parallel_jobs: 2 }, configs: [
      { path: 'configs/thesis/b1/UBFC/physnet_face.yaml', group: 'local', mode: 'train_and_test', model: 'Physnet', train: 'UBFC-rPPG', test: 'UBFC-rPPG', backend: 'local' },
      { path: 'configs/thesis/b1/UBFC/pos_face.yaml', group: 'local', mode: 'unsupervised_method', model: '', train: '', test: 'UBFC-rPPG', backend: 'local' },
      { path: 'configs/thesis/b1/PURE/physnet_face.yaml', group: 'local', mode: 'train_and_test', model: 'Physnet', train: 'PURE', test: 'PURE', backend: 'local' },
      { path: 'configs/other.yaml', group: 'local', mode: 'train_and_test', model: 'TSCAN', train: 'PURE', test: 'PURE', backend: 'docker' },
    ] };
  return { ...api, el: document.getElementById };
}

const newBatch = (f, selected = []) => {
  f.S.batch = { selected: new Set(selected), query: '', backend: null, answers: {}, secrets: {}, name: '', result: null,
    excluded: new Set(), filter: null, seq: 0, nodes: [] };
  return f.S.batch;
};
const checkbox = node => node.children.find(c => c.tagName === 'INPUT');
const folderBox = (f, name) => f.el('batchTree').byTag('summary').find(s => s.textContent.startsWith(name)).children[0];

test('a folder checkbox selects every config below it; partly selected folders are indeterminate', () => {
  const f = fixture();
  const b = newBatch(f);
  f.renderBatchSelect();
  assert.equal(f.el('batchCheck').disabled, true, 'nothing to check yet');
  const ubfc = folderBox(f, 'UBFC');
  ubfc.checked = true;
  ubfc.events.change();
  assert.deepEqual([...b.selected].sort(), ['configs/thesis/b1/UBFC/physnet_face.yaml', 'configs/thesis/b1/UBFC/pos_face.yaml']);
  assert.equal(f.el('batchCheck').textContent, 'Check 2 configs');
  assert.equal(f.el('batchCount').textContent, '2 configs selected');
  const b1 = folderBox(f, 'b1');
  assert.equal(b1.checked, false);
  assert.equal(b1.indeterminate, true, 'one of the two dataset folders is selected');
  assert.equal(ubfc.checked, true);
  assert.equal(b.radios.find(r => r.value === 'local').radio.checked, true, 'the backend follows the selected configs (local paths)');
  assert.equal(b.radios.find(r => r.value === 'docker').radio.checked, false);
  b1.checked = true;
  b1.events.change();
  assert.equal(b.selected.size, 3);
  assert.equal(b1.indeterminate, false);
});

test('the search keeps the selection and folders open on what is selected', () => {
  const f = fixture();
  const b = newBatch(f, ['configs/other.yaml']);
  b.query = 'pure';
  f.renderBatchSelect();
  const files = f.el('batchTree').byClass('batch-file').map(n => n.attrs.title);
  assert.deepEqual(files, ['configs/thesis/b1/PURE/physnet_face.yaml', 'configs/other.yaml']);
  assert.equal(checkbox(f.el('batchTree').byClass('batch-file')[1]).checked, true);
  assert.equal(f.el('batchCount').textContent, '1 config selected');
});

const result = () => ({
  name: 'thesis/b1', folder: 'configs/thesis/b1', launchable: 3, can_launch: true,
  counts: { ready: 2, warn: 1, blocked: 2 },
  checks: [{ id: 'policy', title: 'Run order', status: 'answered', message: '3 jobs', options: [{ value: 'queue', label: 'One after another' }] }],
  issues: [
    { status: 'error', message: 'folder of experiment JZC not found', configs: ['JZC/a', 'JZC/b'] },
    { status: 'error', message: 'Only one config', configs: ['JZC/b'] },
  ],
  configs: [
    { path: 'p/a', name: 'UBFC/a', status: 'ready', errors: [], warnings: [], model: 'Physnet', datasets: 'UBFC-rPPG', description: 'First' },
    { path: 'p/b', name: 'UBFC/b', status: 'warn', errors: [], warnings: ['usually 72 px'], model: 'Physnet', datasets: 'UBFC-rPPG' },
    { path: 'p/c', name: 'UBFC/c', status: 'ready', errors: [], warnings: [], model: 'POS', datasets: 'UBFC-rPPG' },
    { path: 'p/d', name: 'JZC/a', status: 'blocked', errors: ['folder of experiment JZC not found'], warnings: [] },
    { path: 'p/e', name: 'JZC/b', status: 'blocked', errors: ['folder of experiment JZC not found', 'Only one config'], warnings: [] },
  ],
});

test('the checks list the configs, the shared problems and launch the ready configs', () => {
  const f = fixture();
  const b = newBatch(f);
  b.result = result();
  f.el('batchLaunch');
  f.renderBatchChecks();
  const body = f.el('batchBody');
  assert.equal(f.el('batchWill').textContent, '3 of 5 configs will be launched');
  assert.equal(f.el('batchLaunch').textContent, 'Launch 3 jobs');
  assert.equal(f.el('batchLaunch').disabled, false);
  assert.deepEqual(body.byClass('issue').map(n => n.textContent), ['folder of experiment JZC not found2 configsJZC/a, JZC/b'],
    'only problems of several configs are listed together');
  const rows = body.byClass('batch-row');
  assert.equal(rows.length, 5);
  const boxes = rows.map(row => checkbox(row.children[0]));
  assert.deepEqual(boxes.map(box => 'disabled' in box.attrs), [false, false, false, true, true], 'blocked configs cannot be launched');
  assert.ok(rows[1].textContent.includes('Ready · 1 warning'));
  assert.ok(rows[4].textContent.includes('+1 more'));
  boxes[2].checked = false;
  boxes[2].events.change();
  assert.deepEqual([...b.excluded], ['p/c']);
  assert.equal(f.el('batchLaunch').textContent, 'Launch 2 jobs');
  // Filtering by status keeps the exclusions.
  body.byClass('batch-blocked')[0].events.click();
  assert.equal(b.filter, 'blocked');
  assert.equal(f.el('batchBody').byClass('batch-row').length, 2);
  assert.equal(f.el('batchLaunch').textContent, 'Launch 2 jobs');
});

test('an open question blocks the launch', () => {
  const f = fixture();
  const b = newBatch(f);
  b.result = result();
  b.result.checks[0].status = 'question';
  f.renderBatchChecks();
  assert.equal(f.el('batchLaunch').disabled, true);
});

const job = (id, index, status, extra = {}) => ({ id, name: `UBFC/${id}`, status, created: '2026-11-03T10:00:00', backend: 'local',
  batch: { id: 'B1', name: 'thesis/b1', index, count: 3 }, ...extra });
const entry = (index, status, extra = {}) => ({ path: `p/${index}`, name: `UBFC/${index}`, index, status, job: `j${index}`, attempts: 1, ...extra });

test('batch jobs are nested under their batch; long batches start collapsed', () => {
  const f = fixture();
  f.el('jobFilter').value = 'all';
  f.el('tab-jobs').className = '';
  f.S.batches = [{ id: 'B1', name: 'thesis/b1', created: '2026-11-03T10:00:00', status: 'running', active: true, backend: 'local',
    answers: { policy: 'queue' }, configs: [entry(0, 'succeeded'), entry(1, 'running'), entry(2, 'queued')] }];
  f.S.jobs = [job('j0', 0, 'succeeded'), job('j1', 1, 'running'), job('j2', 2, 'queued'),
    { id: 'solo', name: 'Single run', status: 'succeeded', created: '2026-11-02T09:00:00', backend: 'local' }];
  f.renderJobList();
  const list = f.el('jobList');
  assert.equal(list.children.length, 2, 'one batch row and one single job');
  assert.ok(list.children[0].textContent.includes('batch · 1/3 done'));
  assert.deepEqual(list.children[0].byClass('fold').map(n => n.textContent.slice(0, 6)), ['UBFC/j', 'UBFC/j', 'UBFC/j']);
  assert.equal(f.S.selectedBatch, 'B1', 'the newest entry is selected');
  f.S.batches[0].configs = Array.from({ length: 13 }, (_, i) => entry(i, 'queued'));
  f.renderJobList();
  assert.equal(list.children[0].byClass('fold').length, 0, 'more than 12 jobs: collapsed until opened');
});

test('the batch detail lists its jobs and relaunches the unfinished ones', () => {
  const f = fixture();
  const hour = 3600 * 1000;
  const now = Date.now();
  f.S.batches = [{ id: 'B1', name: 'thesis/b1', created: '2026-11-03T10:00:00', status: 'running', active: true, backend: 'local',
    folder: 'configs/thesis/b1', answers: { policy: 'parallel', wandb: 'existing' }, configs: [
      entry(0, 'succeeded', { started: new Date(now - 3 * hour).toISOString(), finished: new Date(now - 2 * hour).toISOString() }),
      entry(1, 'failed'), entry(2, 'queued'), entry(3, 'queued')] }];
  f.S.jobs = [job('j0', 0, 'succeeded', { summary: { mae_bpm: 2.345 } }), job('j1', 1, 'failed', { progress: { metrics: { FFT: { MAE: { value: 7.5 } } } } })];
  f.S.selectedBatch = 'B1';
  f.updateBatchDetail(true);
  const text = f.el('jobDetail').textContent;
  assert.ok(text.includes('1 of 4 succeeded'));
  assert.ok(text.includes('Run order: several at once'));
  assert.ok(text.includes('W&B: saved API key'));
  assert.ok(text.includes('2.35') && text.includes('7.50'), 'MAE from summary.json or the console metrics');
  const buttons = f.el('bdHead').byTag('button').map(n => n.textContent);
  assert.deepEqual(buttons, ['Stop remaining jobs', 'Relaunch 1 unfinished…']);
  // Two queued jobs of ~1 h, two at a time.
  const progress = f.batchProgress(f.S.batches[0]);
  assert.ok(Math.abs(progress.eta - 3600) < 5, `expected ~1 h left, got ${progress.eta}`);
});
