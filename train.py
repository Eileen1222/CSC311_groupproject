"""Reproduce the grouped CSC311 experiments from the released CSV."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from pred import (CLASSES, MULTI, TEXT, model_predict, normalize, numeric_values,
                  options, transform)

FAMILIES = ('knn', 'tree', 'softmax')


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')


def fit_preprocessing(frame, min_df):
    numeric = numeric_values(frame)
    numeric[~np.isfinite(numeric)] = np.nan
    medians = pd.DataFrame(numeric).median().fillna(0).to_numpy()
    filled = np.where(np.isnan(numeric), medians, numeric)
    filled[:, -1] = np.log1p(filled[:, -1])
    std = filled.std(axis=0)
    state = dict(median=medians.tolist(), mean=filled.mean(axis=0).tolist(),
                 std=np.where(std > 0, std, 1).tolist(),
                 options={c: sorted(set().union(*frame[c].map(options))) for c in MULTI},
                 text={})
    for col in TEXT:
        corpus = frame[col].fillna('')
        vec = TfidfVectorizer(min_df=min_df, max_features=100, ngram_range=(1, 1))
        try:
            reference = vec.fit_transform(corpus)
        except ValueError as error:
            if not any(reason in str(error).lower() for reason in
                       ('empty vocabulary', 'no terms remain', 'max_df corresponds')):
                raise
            state['text'][col] = dict(terms=[], idf=[])
            continue
        state['text'][col] = dict(terms=vec.get_feature_names_out().tolist(),
                                  idf=vec.idf_.tolist())
        # Validate deployment TF-IDF against the fitted library implementation.
        reconstructed = transform(frame, {**state, 'text': {
            c: state['text'][col] if c == col else dict(terms=[], idf=[]) for c in TEXT}})
        np.testing.assert_allclose(reconstructed[:, -reference.shape[1]:],
                                   reference.toarray(), atol=1e-12)
    return state


def softmax_fit(x, y, rate, penalty):
    # Contiguous transpose avoids repacking the same array at every update.
    xt = np.ascontiguousarray(x.T)
    w = np.zeros((x.shape[1], len(CLASSES)))
    b = np.zeros(len(CLASSES))
    rows = np.arange(len(x))

    def objective():
        scores = x @ w + b
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        denom = exp.sum(axis=1)
        loss = (np.log(denom) - scores[rows, y]).mean() + penalty * np.square(w).sum() / 2
        return loss, exp / denom[:, None]

    previous, prob = objective()
    history = [[0, float(previous)]]
    stable = 0
    increases = 0
    status = 'iteration_limit'
    for step in range(1, 5001):
        prob[rows, y] -= 1
        w -= rate * (xt @ prob / len(x) + penalty * w)
        b -= rate * prob.mean(axis=0)
        loss, prob = objective()
        if not np.isfinite(loss):
            status = 'nonfinite'
            break
        increases += int(loss > previous + 1e-12)
        relative = abs(loss - previous) / max(abs(previous), np.finfo(float).eps)
        stable = stable + 1 if relative < 1e-7 else 0
        previous = loss
        if step % 100 == 0:
            history.append([step, float(loss)])
        if stable == 20:
            status = 'converged'
            break
    if np.isfinite(loss) and history[-1][0] != step:
        history.append([step, float(loss)])
    return dict(kind='softmax', weights=w, bias=b), dict(
        status=status, updates=step, objective_increases=increases, loss_trace=history)


def fit_model(x, y, config, state):
    kind = config['family']
    if kind == 'softmax':
        return softmax_fit(x, y, config['rate'], config['penalty']), None
    if kind == 'tree':
        estimator = DecisionTreeClassifier(criterion='entropy', max_depth=config['depth'],
                                           min_samples_leaf=config['leaf'], random_state=42)
        estimator.fit(x, y)
        tree = estimator.tree_
        model = dict(kind=kind, left=tree.children_left, right=tree.children_right,
                     feature=tree.feature, threshold=tree.threshold,
                     labels=estimator.classes_[tree.value[:, 0, :].argmax(axis=1)])
        return (model, dict(status='fitted', depth=tree.max_depth, nodes=tree.node_count)), estimator
    text_start = x.shape[1] - sum(len(field['terms']) for field in state['text'].values())
    vectors = x.copy()
    vectors[:, text_start:] *= config['text_weight']
    model = dict(kind=kind, train=vectors, labels=y, k=config['k'],
                 text_start=text_start, text_weight=config['text_weight'])
    return (model, dict(status='stored', rows=len(y))), None


def predict_checked(x, model, estimator):
    prediction = model_predict(x.copy(), model)
    if estimator is not None:
        np.testing.assert_array_equal(prediction, estimator.predict(x))
    return prediction


def export(path, state, model):
    dump(path, dict(preprocessing=state, model={
        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}))


def metrics(y, prediction):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(matrix, (y, prediction), 1)
    tp = matrix.diagonal()
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    return dict(accuracy=float(tp.sum() / matrix.sum()),
                macro_f1=float((2 * tp / np.maximum(actual + predicted, 1)).mean()),
                precision=(tp / np.maximum(predicted, 1)).tolist(),
                recall=(tp / np.maximum(actual, 1)).tolist(), confusion=matrix.tolist(),
                row_normalized=(matrix / np.maximum(actual[:, None], 1)).tolist())


def configs():
    baseline, extension = [], []
    for df in (2, 3, 5):
        baseline.extend(dict(family='knn', df=df, k=k, text_weight=1)
                        for k in (1, 3, 5, 9, 15))
        baseline.extend(dict(family='tree', df=df, depth=d, leaf=l)
                        for d, l in product((3, 5, 8, None), (2, 5, 10)))
        baseline.extend(dict(family='softmax', df=df, rate=r, penalty=p)
                        for r, p in product((.01, .05, .1), (.0001, .001, .01, .1)))
        extension.extend(dict(family='knn', df=df, k=k, text_weight=w)
                         for k, w in product((1, 3, 5, 9, 15), (.5, 2)))
        extension.extend(dict(family='softmax', df=df, rate=r, penalty=p)
                         for r, p in product((.2, .4), (.0001, .001, .01, .1)))
    assert len(baseline) == 87
    return [dict(c, stage=stage, config_id=f'{stage}-{i+1:02d}')
            for stage, grid in [('proposal', baseline), ('extension', extension)]
            for i, c in enumerate(grid)]


def pick(rows):
    def order(row):
        c = row['config']
        if c['family'] == 'knn':
            simplest = (-c['k'],)
        elif c['family'] == 'tree':
            simplest = (float('inf') if c['depth'] is None else c['depth'], -c['leaf'])
        else:
            simplest = (-c['penalty'],)
        return (-row['macro_f1_mean'], -row['accuracy_mean'], *simplest,
                c['stage'] != 'proposal', c['config_id'])
    return min(rows, key=order)


def labels(raw):
    return pd.Categorical(raw.label, categories=CLASSES).codes.astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('output', type=Path, help='New experiment directory')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True)
    raw = pd.read_csv(args.csv, keep_default_na=False)
    if raw.id.eq('').any():
        raise ValueError('Blank respondent ID')
    ids = np.sort(raw.id.unique())
    np.random.default_rng(42).shuffle(ids)
    a, b = int(.7 * len(ids)), int(.15 * len(ids))
    split_ids = np.split(ids, [a, a + b])
    split_names = ['train', 'validation', 'internal_test']
    pd.DataFrame({'id': ids, 'split': np.repeat(split_names, [len(g) for g in split_ids])}).to_csv(
        out / 'split_ids.csv', index=False)
    train, validation, test = [raw.loc[raw.id.isin(group)] for group in split_ids]
    assert [len(r) for r in (train, validation, test)] == [609, 129, 132]
    dump(out / 'environment.json', dict(python=platform.python_version(), numpy=np.__version__,
         pandas=pd.__version__, sklearn=sklearn.__version__, input_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
         class_order=CLASSES, seed=42))
    x, y = normalize(train), labels(train)
    assert np.all(y >= 0)
    folds = list(GroupKFold(5).split(x, y, groups=train.id))
    for tr, va in folds:
        assert not set(train.id.iloc[tr]) & set(train.id.iloc[va])
    pd.concat([train.iloc[va][['id']].drop_duplicates().assign(fold=i)
               for i, (_, va) in enumerate(folds, 1)]).to_csv(out / 'fold_ids.csv', index=False)
    cached = {}
    for i, (tr, va) in enumerate(folds, 1):
        for df in (2, 3, 5):
            start = perf_counter()
            state = fit_preprocessing(x.iloc[tr], df)
            xtr, xva = transform(x.iloc[tr], state), transform(x.iloc[va], state)
            cached[(i, df)] = state, xtr, xva, y[tr], y[va]
            print(f'fold={i} min_df={df} preprocess={perf_counter()-start:.3f}s width={xtr.shape[1]}', flush=True)

    scores = []
    with (out / 'fold_metrics.jsonl').open('w', encoding='utf-8') as log:
        for c in configs():
            fold_scores = []
            for i in range(1, 6):
                state, xtr, xva, ytr, yva = cached[(i, c['df'])]
                start = perf_counter()
                (model, diagnostics), estimator = fit_model(xtr, ytr, c, state)
                fit_time = perf_counter() - start
                start = perf_counter()
                result = dict(config=c, fold=i, n_train=len(ytr), n_validation=len(yva),
                              features=xtr.shape[1], fit_seconds=fit_time, **diagnostics)
                valid = diagnostics['status'] != 'nonfinite' and not diagnostics.get('objective_increases', 0)
                result['eligible'] = valid
                if valid:
                    result.update(metrics(yva, predict_checked(xva, model, estimator)))
                    fold_scores.append(result)
                result['score_seconds'] = perf_counter() - start
                print(json.dumps(result, allow_nan=False), file=log, flush=True)
            row = dict(config=c, eligible=len(fold_scores) == 5)
            if row['eligible']:
                for metric in ('macro_f1', 'accuracy'):
                    vals = [item[metric] for item in fold_scores]
                    row[metric + '_mean'] = float(np.mean(vals))
                    row[metric + '_sd'] = float(np.std(vals, ddof=1))
            scores.append(row)
            print(f'{c["config_id"]} {c["family"]} F1={row.get("macro_f1_mean")}', flush=True)
    pd.json_normalize(scores).to_csv(out / 'cv_summary.csv', index=False)
    champions = {}
    for family in FAMILIES:
        pool = [row for row in scores if row['config']['family'] == family and row['eligible']]
        original = pick([row for row in pool if row['config']['stage'] == 'proposal'])
        expanded = pick(pool)
        # Keep the prespecified representation unless an extension improves mean F1.
        chosen = expanded if expanded['macro_f1_mean'] > original['macro_f1_mean'] else original
        champions[family] = dict(proposal=original, expanded=expanded, chosen=chosen)
    dump(out / 'cv_selection.json', champions)

    xv, yv = normalize(validation), labels(validation)
    assert np.all(yv >= 0)
    heldout = {}
    for family, selection in champions.items():
        c = selection['chosen']['config']
        start = perf_counter()
        state = fit_preprocessing(x, c['df'])
        (model, diagnostics), estimator = fit_model(transform(x, state), y, c, state)
        assert diagnostics['status'] != 'nonfinite' and not diagnostics.get('objective_increases', 0)
        fit_time = perf_counter() - start
        candidate = out / 'candidate.json'
        export(candidate, state, model)
        artifact_bytes = candidate.stat().st_size + Path(__file__).with_name('pred.py').stat().st_size
        candidate.unlink()
        pv = predict_checked(transform(xv, state), model, estimator)
        val = metrics(yv, pv)
        error_rows = np.flatnonzero(yv != pv)
        missing = xv.isna().any(axis=1).to_numpy() | (~np.isfinite(numeric_values(xv))).any(axis=1)
        any_oov = np.zeros(len(xv), dtype=bool)
        from pred import TOKEN
        for col in TEXT:
            vocab = set(state['text'][col]['terms'])
            any_oov |= xv[col].fillna('').map(lambda s: bool(set(TOKEN.findall(s)) - vocab)).to_numpy()
        comparisons = {name: dict(errors=int(mask[error_rows].sum()), rows=int(mask.sum()))
                       for name, mask in [('any_missing', missing), ('any_text_oov', any_oov)]}
        heldout[family] = dict(config=c, **val, diagnostics=diagnostics, fit_seconds=fit_time,
                              artifact_bytes=artifact_bytes, error_strata=comparisons)
    dump(out / 'validation.json', heldout)
    winner = min(heldout, key=lambda f: (-heldout[f]['macro_f1'], -heldout[f]['accuracy'],
                                        heldout[f]['artifact_bytes']))
    c = heldout[winner]['config']
    dump(out / 'locked_selection.json', c)
    start = perf_counter()
    development = pd.concat([train, validation]).sort_index()
    xd = normalize(development)
    state = fit_preprocessing(xd, c['df'])
    (model, diagnostics), estimator = fit_model(transform(xd, state), labels(development), c, state)
    assert diagnostics['status'] != 'nonfinite' and not diagnostics.get('objective_increases', 0)
    export(out / 'model.json', state, model)
    dump(out / 'refit.json', dict(config=c, rows=len(xd), features=transform(xd, state).shape[1],
                                 fit_and_export_seconds=perf_counter() - start, **diagnostics))
    # First and only scoring pass on the internal test. Use the serialized state.
    locked = json.loads((out / 'model.json').read_text())
    xt = transform(normalize(test), locked['preprocessing'])
    pt = predict_checked(xt, locked['model'], estimator)
    yt = labels(test)
    assert np.all(yt >= 0)
    dump(out / 'internal_test.json', dict(config=c, rows=len(test), **metrics(yt, pt)))
    print(f'Final family: {winner}. Internal test accuracy: {(pt == yt).mean():.6f}', flush=True)


if __name__ == '__main__':
    main()
