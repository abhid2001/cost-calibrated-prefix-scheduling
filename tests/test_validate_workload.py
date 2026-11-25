import datetime, importlib.util, json, pathlib, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validate_workload', ROOT / 'scripts' / 'validate_workload.py')
validate_workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_workload)


class ValidateWorkloadTests(unittest.TestCase):
    def write_rows(self, directory, name, rows):
        (directory / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def valid_directory(self, directory):
        t = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).isoformat()
        request = lambda rid, cached: {'request_id': rid, 'question_id': rid, 'parent_ids': [], 'status': 'completed',
            'ready_at': t, 'dispatched_at': t, 'admitted_at': t, 'first_token_at': t, 'finished_at': t,
            'engine_metadata': {'cached_tokens': cached, 'prefill_gpu_ms': 1.0}}
        self.write_rows(directory, 'trace.jsonl', [{'question_id': 'q1', 'mode': 'direct'}, {'question_id': 'q2', 'mode': 'direct'}])
        self.write_rows(directory, 'requests.jsonl', [request('q1', 2), request('q2', 0)])
        self.write_rows(directory, 'questions.jsonl', [{'question_id': 'q1', 'mode': 'direct', 'status': 'completed'}, {'question_id': 'q2', 'mode': 'direct', 'status': 'completed'}])
        self.write_rows(directory, 'admissions.jsonl', [
            {'actual_id': 'q1', 'admission': 0, 'slot': 0, 'reason': 'oldest', 'ranked_ids': ['q1', 'q2'],
             'candidates': [{'id': 'q1', 'cached_tokens': 2, 'age': 1}, {'id': 'q2', 'cached_tokens': 0, 'age': 2}]},
            {'actual_id': 'q2', 'admission': 1, 'slot': 1, 'reason': 'oldest', 'ranked_ids': ['q2'],
             'candidates': [{'id': 'q2', 'cached_tokens': 0, 'age': 2}]},
        ])
        self.write_rows(directory, 'events.jsonl', [{'sequence': 1, 'event': 'materialized'}, {'sequence': 2, 'event': 'question_arrived'}])

    def test_accepts_complete_fcfs_workload(self):
        with tempfile.TemporaryDirectory() as name:
            directory = pathlib.Path(name); self.valid_directory(directory)
            self.assertEqual(validate_workload.validate(directory, 'fcfs')['errors'], [])

    def test_reports_missing_and_duplicate_admissions(self):
        with tempfile.TemporaryDirectory() as name:
            directory = pathlib.Path(name); self.valid_directory(directory)
            admissions = (directory / 'admissions.jsonl').read_text().splitlines()
            (directory / 'admissions.jsonl').write_text(admissions[0] + '\n')
            self.assertIn('admissions are missing, duplicated, or have unexpected request IDs', validate_workload.validate(directory, 'fcfs')['errors'])
            self.write_rows(directory, 'admissions.jsonl', [json.loads(admissions[0]), json.loads(admissions[0]), json.loads(admissions[1])])
            self.assertIn('admissions are missing, duplicated, or have unexpected request IDs', validate_workload.validate(directory, 'fcfs')['errors'])
