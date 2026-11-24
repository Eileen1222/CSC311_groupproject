"""Independently verify saved grid coverage, splits, and metric calculations."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from train import configs, pick

root, exploration = map(Path, sys.argv[1:])
def read(name):
    return json.loads((root / f'{name}.json').read_text())

rows = [json.loads(line) for line in (root / 'fold_metrics.jsonl').read_text().splitlines()]
groups = {}
for row in rows:
    cfg = row['config']
    groups.setdefault(cfg['config_id'], []).append(row)
expected = {c['config_id']: c for c in configs()}
assert set(groups) == set(expected)
for key, items in groups.items():
    assert {r['fold'] for r in items} == set(range(1, 6)) and len(items) == 5
    assert all(r['config'] == expected[key] for r in items)
    assert all(r['n_train'] + r['n_validation'] == 609 for r in items)
split = pd.read_csv(root / 'split_ids.csv')
proposal_split = pd.read_csv(exploration / 'split_ids.csv').replace({'test': 'internal_test'})
pd.testing.assert_frame_equal(split, proposal_split)
assert split.id.is_unique
folds = pd.read_csv(root / 'fold_ids.csv')
assert folds.id.is_unique and set(folds.id) == set(split.loc[split.split == 'train', 'id'])
validation, test = read('validation'), read('internal_test')
for row in [r for r in rows if r['eligible']] + list(validation.values()) + [test]:
    mat = np.asarray(row['confusion'])
    tp = mat.diagonal()
    actual = mat.sum(axis=1)
    predicted = mat.sum(axis=0)
    np.testing.assert_allclose(row['accuracy'], tp.sum() / mat.sum())
    np.testing.assert_allclose(row['macro_f1'], np.mean(2 * tp / np.maximum(actual + predicted, 1)))
    np.testing.assert_allclose(row['precision'], tp / np.maximum(predicted, 1))
    np.testing.assert_allclose(row['recall'], tp / actual)
    np.testing.assert_allclose(row['row_normalized'], mat / actual[:, None])
selection = read('cv_selection')
summary = []
for key, items in groups.items():
    if not all(r['eligible'] for r in items):
        continue
    row = dict(config=expected[key], eligible=True)
    for measure in ('accuracy', 'macro_f1'):
        values = [r[measure] for r in items]
        row[measure + '_mean'], row[measure + '_sd'] = float(np.mean(values)), float(np.std(values, ddof=1))
    summary.append(row)
for f, leaderboard in selection.items():
    pool = [r for r in summary if r['config']['family'] == f]
    proposal = pick([r for r in pool if r['config']['stage'] == 'proposal'])
    expanded = pick(pool)
    chosen = expanded if expanded['macro_f1_mean'] > proposal['macro_f1_mean'] else proposal
    assert leaderboard == dict(proposal=proposal, expanded=expanded, chosen=chosen)
    assert validation[f]['config'] == chosen['config']
winner = min(validation, key=lambda f: (-validation[f]['macro_f1'], -validation[f]['accuracy'],
                                      validation[f]['artifact_bytes']))
assert read('locked_selection') == validation[winner]['config'] == test['config'] == read('refit')['config']
assert test['rows'] == 132 and read('refit')['rows'] == 738
print(json.dumps(dict(split_respondents=split.split.value_counts().to_dict(),
    fold_validation_respondents=folds.fold.value_counts().to_dict(),
    fold_fits_by_stage=dict(Counter(r['config']['stage'] for r in rows)),
    fold_fits_by_model=dict(Counter(r['config']['family'] for r in rows)),
    softmax_status=dict(Counter(r['config']['stage'] + '/' + r['status']
                        for r in rows if r['config']['family'] == 'softmax')),
    objective_increases=sum(r.get('objective_increases', 0) for r in rows),
    eligible_fits=sum(r['eligible'] for r in rows), winner=winner,
    saved_metrics_and_selection_verified=True, proposal_partition_exact_match=True), indent=2))
