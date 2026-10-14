"""Launcher recovery state tests without launching real jobs or using credentials."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launcher import configs, jobs, paths, server


class LauncherRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.patcher = patch.object(paths, 'JOBS_DIR', self.folder / 'jobs')
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.manager = jobs.JobManager.__new__(jobs.JobManager)
        self.manager.lock = threading.RLock()
        for name in ('jobs', 'processes', 'parsers', 'offsets', 'secret_env'):
            setattr(self.manager, name, {})
        self.job_dir = paths.JOBS_DIR / 'test-job'
        self.job_dir.mkdir(parents=True)
        checkpoint = self.folder / 'models' / 'toy_resume.pt'
        checkpoint.parent.mkdir()
        checkpoint.touch()
        checkpoint.with_suffix('.json').write_text(json.dumps({'next_epoch': 2, 'epochs_total': 4,
                                                             'training_complete': False}))
        self.job = {
            'id': 'test-job', 'name': 'Test training', 'created': '2026-10-02T10:00:00',
            'status': 'running', 'backend': 'local', 'mode': 'train_and_test',
            'training_recovery': True, 'epochs_total': 4, 'gpu': 0, 'pid': 12345,
            'resume_checkpoint_host': str(checkpoint), 'resume_checkpoint': str(checkpoint),
            'outputs': {'model_dir': str(checkpoint.parent), 'model_dir_host': str(checkpoint.parent)},
            'progress': {'phase': 'training'},
        }
        self.manager.jobs[self.job['id']] = self.job
        (self.job_dir / 'output.log').touch()
        (self.job_dir / 'config.yaml').write_text(configs.dump_yaml({
            'BASE': [''], 'TOOLBOX_MODE': 'train_and_test', 'MODEL': {'NAME': 'FactorizePhys'},
            'TRAIN': {'MODEL_FILE_NAME': 'toy', 'DATA': {'DO_PREPROCESS': True}},
            'VALID': {'DATA': {'DO_PREPROCESS': True}}, 'TEST': {'DATA': {'DO_PREPROCESS': True}},
        }))

    def test_pause_then_exit_releases_gpu_and_resume_uses_snapshot(self):
        result = self.manager.pause('test-job')
        self.assertEqual(result['status'], 'pausing')
        self.assertEqual(self.manager.running_count(), 1)
        self.assertEqual(json.loads((self.job_dir / 'training-control.json').read_text()), {'action': 'pause'})
        (self.job_dir / 'exit.json').write_text(json.dumps({'returncode': 75}))
        with patch.object(jobs.probes, 'pid_alive', return_value=False):
            self.manager._follow(self.job)
        self.assertEqual(self.job['status'], 'paused')
        self.assertEqual(self.manager.running_count(), 0)
        self.assertEqual(self.manager.jobs_on_gpu(0), [])
        prepared = self.manager.resume_config('test-job')
        self.assertEqual(prepared['values']['MODEL.RESUME'], self.job['resume_checkpoint'])
        self.assertEqual(prepared['answers']['outputs'], 'overwrite')
        for prefix in ('TRAIN.DATA', 'VALID.DATA', 'TEST.DATA'):
            self.assertFalse(prepared['values'][prefix + '.DO_PREPROCESS'])
        self.assertEqual(prepared['resume_from'], 'test-job')
        self.assertEqual(self.job['status'], 'paused')  # each attempt keeps its log/state

    def test_failed_and_cancelled_jobs_can_resume_but_legacy_weights_cannot(self):
        for status in ('failed', 'cancelled', 'interrupted'):
            self.job['status'] = status
            self.assertTrue(self.manager.get('test-job')['can_resume'])
        Path(self.job['resume_checkpoint_host']).unlink()
        self.assertFalse(self.manager.get('test-job')['can_resume'])
        with self.assertRaisesRegex(ValueError, 'No committed full-state checkpoint'):
            self.manager.resume_config('test-job')

    def test_restart_preserves_pausing_and_paused_state(self):
        self.manager.pause('test-job')
        reloaded = jobs.JobManager.__new__(jobs.JobManager)
        reloaded.jobs, reloaded.parsers, reloaded.offsets = {}, {}, {}
        reloaded._load()
        self.assertEqual(reloaded.jobs['test-job']['status'], 'pausing')
        self.assertIn('test-job', reloaded.parsers)
        self.job['status'] = 'paused'
        self.manager._save(self.job)
        reloaded._load()
        self.assertEqual(reloaded.jobs['test-job']['status'], 'paused')

    def test_pause_unavailable_after_training_and_duplicate_resume_blocked(self):
        self.job['progress']['phase'] = 'testing'
        with self.assertRaisesRegex(ValueError, 'Only new, running training'):
            self.manager.pause('test-job')
        self.job['status'] = 'paused'
        self.manager.jobs['other'] = {'id': 'other', 'status': 'queued', 'outputs': self.job['outputs']}
        with self.assertRaisesRegex(ValueError, 'Another active attempt'):
            self.manager.resume_config('test-job')

    def test_docker_step_mounts_control_file_and_passes_environment(self):
        steps = self.manager._steps('test-job', 'docker', self.job_dir / 'config.yaml', {})
        argv = steps[-1]['argv']
        self.assertIn(f'{self.job_dir.as_posix()}:{paths.JOB_CONFIG_MOUNT}/test-job-control:ro', argv)
        self.assertIn(f'RPPG_TRAINING_CONTROL={paths.JOB_CONFIG_MOUNT}/test-job-control/training-control.json', argv)

    def test_discovers_actual_motion_augmented_checkpoint_path(self):
        actual = self.folder / 'models' / 'MA-toy_resume.pt'
        actual.touch()
        (self.job_dir / 'output.log').write_text(
            f'[training] Resume checkpoint saved (next epoch 1): {actual}\n'
        )
        self.manager._read_new_output(self.job)
        self.assertEqual(self.job['resume_checkpoint_host'], str(actual.resolve()))
        self.assertEqual(self.job['resume_checkpoint'], str(actual))

    def test_pause_and_resume_http_routes_require_token(self):
        with patch.object(server.jobs, 'JobManager', return_value=self.manager):
            http = server.LauncherServer(('127.0.0.1', 0), set())
        base = f'http://127.0.0.1:{http.server_port}'
        http.allowed_hosts = {f'127.0.0.1:{http.server_port}'}
        thread = threading.Thread(target=http.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        thread.start()
        try:
            headers = {'Content-Type': 'application/json'}
            request = urllib.request.Request(base + '/api/jobs/test-job/pause', data=b'{}', headers=headers)
            with self.assertRaises(urllib.error.HTTPError) as denied:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(denied.exception.code, 403)
            headers['X-Launcher-Token'] = http.token
            request = urllib.request.Request(base + '/api/jobs/test-job/pause', data=b'{}', headers=headers)
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response)['job']['status'], 'pausing')
            self.job['status'] = 'paused'
            request = urllib.request.Request(base + '/api/jobs/test-job/resume', data=b'{}', headers=headers)
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response)['resume_from'], 'test-job')
        finally:
            http.shutdown()
            thread.join(timeout=5)
            http.server_close()


if __name__ == '__main__':
    unittest.main()