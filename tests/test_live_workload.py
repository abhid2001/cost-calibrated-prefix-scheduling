import pathlib, sys, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from prefix_scheduler.live_workload import materialize, question_result, release


def complete(request, output):
    request.transition('running', 'dispatch')
    request.output = output
    request.transition('completed', 'done')


class LiveWorkloadTests(unittest.TestCase):
    def test_materializes_all_modes_before_arrival(self):
        expected = {'direct': 1, 'cot': 1, 'self_consistency': 3, 'tot': 7}
        for mode, count in expected.items():
            graph = materialize('q-' + mode, 'question', mode, 'arrival', 32)
            self.assertEqual(len(graph.records), count)
            self.assertTrue(all(request.status == 'blocked' for request in graph.records))
            ready, failed = release(graph)
            self.assertFalse(failed)
            self.assertEqual(len(ready), 2 if mode == 'tot' else count)

    def test_tot_releases_and_uses_evaluator_selected_parent(self):
        graph = materialize('q', 'what is 1 + 1?', 'tot', 'arrival', 32)
        roots, _ = release(graph)
        self.assertEqual([request.request_id for request in roots], ['q:r1c1', 'q:r1c2'])
        complete(graph.node('r1c1'), 'first partial')
        complete(graph.node('r1c2'), 'second partial')
        ready, failed = release(graph)
        self.assertFalse(failed)
        self.assertEqual([request.request_id for request in ready], ['q:r1eval'])
        complete(graph.node('r1eval'), 'Selected: 2')
        ready, _ = release(graph)
        self.assertEqual([request.request_id for request in ready], ['q:r2c1', 'q:r2c2'])
        messages, _ = graph.messages_for(graph.node('r2c1'))
        self.assertIn('second partial', messages[0]['content'])
        complete(graph.node('r2c1'), 'third partial')
        complete(graph.node('r2c2'), 'fourth partial')
        ready, _ = release(graph)
        self.assertEqual([request.request_id for request in ready], ['q:r2eval'])
        complete(graph.node('r2eval'), 'Selected: 1')
        ready, _ = release(graph)
        self.assertEqual([request.request_id for request in ready], ['q:final'])
        complete(graph.node('final'), 'Final answer: 2')
        result = question_result(graph)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['answer'], '2')
        self.assertEqual(result['selections'][0]['selected'], 'q:r1c2')

    def test_failed_parent_recursively_prevents_descendant_dispatch(self):
        graph = materialize('q', 'question', 'tot', 'arrival', 32)
        release(graph)
        failed_root = graph.node('r1c1')
        failed_root.transition('running', 'dispatch')
        failed_root.error = 'TimeoutError: timeout'
        failed_root.transition('failed', 'done')
        _, failed = release(graph)
        self.assertEqual({request.request_id for request in failed},
                         {'q:r1eval', 'q:r2c1', 'q:r2c2', 'q:r2eval', 'q:final'})
        self.assertTrue(all(request.status == 'failed' for request in failed))
