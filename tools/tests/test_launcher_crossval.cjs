// Run with: node --test tools/tests/test_launcher_crossval.cjs
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

// Minimal DOM, as in test_launcher_sync.cjs: runs the shipped app.js without its startup.
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
  querySelectorAll(selector) {
    const tags = selector.split(',').map(s => s.trim().toUpperCase());
    return this.children.flatMap(c => [...(tags.includes(c.tagName) ? [c] : []), ...c.querySelectorAll(selector)]);
  }
  byClass(name) {
    return this.children.flatMap(c => [...(c.className.split(' ').includes(name) ? [c] : []), ...c.byClass(name)]);
  }
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
  const api = vm.runInContext(`({ S, isActive, cvEnabled, isStructural, groupProgress, renderJobList, foldTable,
    renderValidation, selectGroup, updateGroupDetail, crossValidationOverview })`, context);
  const vhrm = { modes: ['train_and_test'], dataset: ['TRAIN.DATA', ['vHRM']] };
  api.S.fields = [
    { key: 'CROSS_VALIDATION.ENABLED', type: 'bool', default: false, section: 'Cross-validation', when: vhrm },
    { key: 'CROSS_VALIDATION.SCHEME', type: 'enum', default: 'kfold', section: 'Cross-validation',
      when: { ...vhrm, equals: { 'CROSS_VALIDATION.ENABLED': [true] } } },
    { key: 'CROSS_VALIDATION.K', type: 'int', default: 5, section: 'Cross-validation',
      when: { ...vhrm, equals: { 'CROSS_VALIDATION.ENABLED': [true], 'CROSS_VALIDATION.SCHEME': ['kfold'] } } },
  ];
  api.S.fieldMap = Object.fromEntries(api.S.fields.map(f => [f.key, f]));
  api.S.values = { TOOLBOX_MODE: 'train_and_test', 'TRAIN.DATA.DATASET': 'vHRM', 'CROSS_VALIDATION.ENABLED': false,
    'CROSS_VALIDATION.SCHEME': 'kfold', 'CROSS_VALIDATION.K': 5 };
  api.S.boot = { schema: { splits: {} }, platform: 'nt' };
  return { ...api, el: document.getElementById };
}

const fold = (name, status, extra = {}) => ({ name, test: [`p${name}`], valid: ['v'], train: ['a', 'b'], status, job: `job${name}`, attempts: 1, ...extra });

test('fields that depend on other values follow them; such keys re-render the form', () => {
  const f = fixture();
  const k = f.S.fieldMap['CROSS_VALIDATION.K'];
  assert.equal(f.cvEnabled(), false);
  assert.equal(f.isActive(k, f.S.values), false);
  f.S.values['CROSS_VALIDATION.ENABLED'] = true;
  assert.equal(f.cvEnabled(), true);
  assert.equal(f.isActive(k, f.S.values), true);
  f.S.values['CROSS_VALIDATION.SCHEME'] = 'loso';
  assert.equal(f.isActive(k, f.S.values), false, 'K only applies to K balanced groups');
  f.S.values['TRAIN.DATA.DATASET'] = 'PURE';
  assert.equal(f.cvEnabled(), false, 'cross-validation is vHRM only');
  assert.equal(f.isStructural('CROSS_VALIDATION.ENABLED'), true);
  assert.equal(f.isStructural('CROSS_VALIDATION.SCHEME'), true);
  assert.equal(f.isStructural('CROSS_VALIDATION.K'), false);
});

test('the launch button says when a launch starts a cross-validation', () => {
  const f = fixture();
  f.S.validation = { errors: [], warnings: [], info: [], paths: {}, derived: {} };
  f.renderValidation();
  assert.equal(f.el('launchRun').textContent, 'Launch…');
  f.S.values['CROSS_VALIDATION.ENABLED'] = true;
  f.renderValidation();
  assert.equal(f.el('launchRun').textContent, 'Launch cross-validation…');
});

test('group progress counts finished folds and estimates the time left from them', () => {
  const f = fixture();
  const hour = 3600 * 1000;
  const now = Date.now();
  const group = { active: true, folds: [
    fold('1', 'succeeded', { started: new Date(now - 3 * hour).toISOString(), finished: new Date(now - 2 * hour).toISOString() }),
    fold('2', 'failed'),
    fold('3', 'running'),
    fold('4', 'queued'),
  ] };
  f.S.jobs = [{ id: 'job3', status: 'running', started: new Date(now - hour / 2).toISOString(), epochs_total: 10,
    progress: { phase: 'training', epoch: 5 } }];
  const p = f.groupProgress(group);
  assert.equal(p.done, 1);
  assert.equal(p.failed, 1);
  assert.equal(p.running.name, '3');
  assert.equal(p.fraction, (1 + 0.5) / 4);
  assert.equal(Math.round(p.average), 3600);
  assert.ok(Math.abs(p.eta - 1.5 * 3600) < 5, `two folds of ~1 h left minus 30 min spent, got ${p.eta}`);
});

test('fold jobs are nested under their cross-validation in the job list', () => {
  const f = fixture();
  f.el('jobFilter').value = 'all';
  f.el('tab-jobs').className = '';
  const group = { id: 'g1', name: 'CV toy', created: '2026-10-29T10:00:00', status: 'running', active: true, pool: [],
    folds: [fold('1', 'succeeded'), fold('2', 'running')] };
  f.S.groups = [group];
  f.S.jobs = [
    { id: 'job1', name: 'CV toy · fold 1', status: 'succeeded', created: '2026-10-29T10:00:00', backend: 'local', group: { id: 'g1', fold: '1', index: 0, count: 2 } },
    { id: 'job2', name: 'CV toy · fold 2', status: 'running', created: '2026-10-29T10:00:00', backend: 'local', group: { id: 'g1', fold: '2', index: 1, count: 2 } },
    { id: 'solo', name: 'Single run', status: 'succeeded', created: '2026-10-28T09:00:00', backend: 'local' },
  ];
  f.renderJobList();
  const list = f.el('jobList');
  assert.equal(list.children.length, 2, 'one group row and one single job');
  const [groupNode, solo] = list.children;
  assert.ok(groupNode.textContent.includes('1/2 folds done'));
  assert.deepEqual(groupNode.byClass('fold').map(n => n.textContent.slice(0, 6)), ['Fold 1', 'Fold 2']);
  assert.ok(solo.textContent.includes('Single run'));
  assert.equal(f.S.selectedGroup, 'g1', 'the newest entry is selected');

  f.el('jobFilter').value = 'finished';
  f.renderJobList();
  assert.equal(list.children.length, 1, 'an active cross-validation is not finished');
  f.S.collapsedGroups = { g1: true };
  f.el('jobFilter').value = 'all';
  f.renderJobList();
  assert.equal(list.children[0].byClass('fold').length, 0, 'collapsed groups hide their folds');
});

test('group detail shows the folds, their results and the merged summary', () => {
  const f = fixture();
  const group = { id: 'g1', name: 'CV toy', created: '2026-10-29T10:00:00', status: 'running', active: true, scheme: 'kfold',
    backend: 'local', pool: ['p1', 'p2'], root: 'C:/Run/cv', root_host: 'C:/Run/cv', warnings: ['Only 2 participants'],
    merge: { status: 'succeeded', finished: '2026-10-29T11:00:00' }, summary: { modified: 1, overall: { mae_bpm: 3 } },
    folds: [fold('1', 'succeeded', { summary: { mae_bpm: 2.5 }, epoch: 9, epochs_total: 10, attempts: 2 }), fold('2', 'running')] };
  f.S.groups = [group];
  f.S.jobs = [];
  f.S.selectedGroup = 'g1';
  f.S.groupDetail = { id: 'g1', stamp: 1, merge: 'succeeded', group: { ...group, summary: { modified: 1,
    overall: { windows: 12, mae_bpm: 3, rmse_bpm: 3.2, pearson: 0.9 },
    tables: { by_fold: [{ fold: 1, windows: 6, mae_bpm: 2 }], by_participant: [{ participant: 'p1', mae_bpm: 2 }], by_movement: [] } } } };
  f.updateGroupDetail(true);
  const text = f.el('jobDetail').textContent;
  for (const expected of ['CV toy', 'Stop remaining folds', 'Merge results now', '1 of 2 done', 'running fold 2',
    '2.50', '(2 attempts)', '10 / 10', 'Only 2 participants', 'Merged results (all folds)', 'By participant', 'p1', '3.20']) {
    assert.ok(text.includes(expected), `group detail shows ${expected}`);
  }
});

test('results overview of a cross-validation root reads the merged tables', () => {
  const f = fixture();
  f.S.results.selected = 'C:/Run/cv';
  f.S.groups = [{ id: 'g1', name: 'CV toy', root_host: 'C:\\Run\\cv', status: 'succeeded', folds: [fold('1', 'succeeded')], created: '2026-10-29T10:00:00' }];
  const csv = (header, ...rows) => ({ header, rows });
  const parts = f.crossValidationOverview([
    { name: 'cv_summary/overall_summary.csv', content: csv(['windows', 'mae_bpm', 'pearson'], ['12', '3.5', '0.8']) },
    { name: 'cv_summary/by_fold.csv', content: csv(['fold', 'windows', 'mae_bpm'], ['1', '6', '2'], ['2', '6', '5']) },
    { name: 'cv_plan.json', content: { folds: [fold('1'), fold('2')] } },
  ]);
  const text = parts.map(p => p.textContent).join('\n');
  for (const expected of ['Launcher cross-validation: CV toy', 'Open in Jobs', 'All folds', '3.50', 'By fold', '5.00', 'p2']) {
    assert.ok(text.includes(expected), `overview shows ${expected}`);
  }
});

test('fold table lists the participants of every fold', () => {
  const f = fixture();
  const table = f.foldTable([fold('1'), fold('2')]);
  const text = table.textContent;
  assert.ok(text.includes('p1') && text.includes('p2'));
  assert.ok(text.includes('2 participants'));
});
