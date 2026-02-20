from epitome.test import EpitomeTestCase
from epitome.dataset import EpitomeDataset
from epitome.tuning import tune


class TuningTest(EpitomeTestCase):

    def setUp(self):
        self.dataset = EpitomeDataset(
            targets=['DNase', 'CTCF'],
            cells=['K562', 'HepG2', 'H1', 'A549'],
            data_dir=self.epitome_data_dir,
            assembly=self.epitome_assembly,
        )

    def test_tune_returns_sorted_results(self):
        lr_values = [1e-4, 1e-3]
        results = tune(
            self.dataset,
            lr_values=lr_values,
            max_train_batches=20,
            max_valid_batches=5,
            val_every=10,
            val_batches=5,
            patience=2,
            test_celltypes=['K562'],
            batch_size=32,
        )
        self.assertEqual(len(results), len(lr_values))
        # sorted ascending by best_val_loss
        losses = [r['best_val_loss'] for r in results]
        self.assertEqual(losses, sorted(losses))

    def test_tune_shared_group(self):
        results = tune(
            self.dataset,
            lr_values=[1e-4, 1e-3],
            max_train_batches=20,
            max_valid_batches=5,
            val_every=10,
            val_batches=5,
            patience=2,
            group='test_sweep',
            test_celltypes=['K562'],
            batch_size=32,
        )
        import json
        for r in results:
            groups = [json.loads(l).get('group') for l in open(r['log_path'])]
            self.assertTrue(all(g == 'test_sweep' for g in groups))

    def test_tune_result_fields(self):
        results = tune(
            self.dataset,
            lr_values=[1e-3],
            max_train_batches=20,
            max_valid_batches=5,
            val_every=10,
            val_batches=5,
            patience=2,
            test_celltypes=['K562'],
            batch_size=32,
        )
        r = results[0]
        for key in ('lr', 'best_batch', 'stopped_at', 'best_val_loss', 'run_id', 'log_path'):
            self.assertIn(key, r)
        self.assertEqual(r['lr'], 1e-3)
        self.assertLess(r['best_val_loss'], float('inf'))
