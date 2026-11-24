"""Check the standalone submission on development and synthetic rows."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pred import CLASSES, FEATURES, QUESTIONS, TEXT, normalize, transform

def main():
    csv, run, folder = map(Path, sys.argv[1:])
    manifest = pd.read_csv(run / 'split_ids.csv')
    ids = manifest.loc[manifest.split == 'validation', 'id']
    raw = pd.read_csv(csv, keep_default_na=False)
    batch = raw.loc[raw.id.isin(ids), list(QUESTIONS)].iloc[:60].copy()
    empty = pd.DataFrame([{column: '' for column in QUESTIONS}])
    unseen = empty.copy()
    unseen.loc[0, [QUESTIONS[4], QUESTIONS[7], QUESTIONS[9], QUESTIONS[11]]] = [
        'zzunknownvenue', 'zzunknownsnack', 'zzunknownsong', 'USD 9']
    combined = pd.concat([batch, empty, unseen], ignore_index=True)
    source = folder / 'pred.py'
    model_path = folder / 'model.json'
    size = source.stat().st_size + model_path.stat().st_size
    assert size <= 10_000_000
    parsed = ast.parse(source.read_text(), feature_version=(3, 10))
    imports = {alias.name.split('.')[0] for node in ast.walk(parsed)
               if isinstance(node, ast.Import) for alias in node.names}
    imports.update(node.module.split('.')[0] for node in ast.walk(parsed)
                   if isinstance(node, ast.ImportFrom))
    assert imports <= {'collections', 'pathlib', 'json', 're', 'numpy', 'pandas'}
    saved = json.loads(model_path.read_text())
    state, m = saved['preprocessing'], saved['model']
    frame = normalize(combined)
    x = transform(frame, state)
    start = x.shape[1] - sum(len(state['text'][c]['terms']) for c in TEXT)
    for col in TEXT:
        info = state['text'][col]
        end = start + len(info['terms'])
        if end > start:
            vec = TfidfVectorizer(vocabulary=info['terms'])
            vec.idf_ = np.asarray(info['idf'])
            np.testing.assert_allclose(x[:, start:end], vec.transform(frame[col].fillna('')).toarray(), atol=1e-12)
        start = end
    expected = []
    for row in x:
        if m['kind'] == 'softmax':
            code = np.argmax(np.array(m['weights']).T @ row + m['bias'])
        elif m['kind'] == 'knn':
            row[m['text_start']:] *= m['text_weight']
            neighbors = np.argsort(np.square(np.asarray(m['train']) - row).sum(axis=1), kind='stable')[:m['k']]
            code = np.bincount(np.asarray(m['labels'])[neighbors], minlength=3).argmax()
        else:
            n = 0
            row = row.astype(np.float32)
            while m['left'][n] != -1:
                n = m['left'][n] if row[m['feature'][n]] <= m['threshold'][n] else m['right'][n]
            code = m['labels'][n]
        expected.append(CLASSES[code])
    script = """
import importlib.abc, json, sys, time
class BlockTrainingLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'torch', 'tensorflow', 'scipy'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, BlockTrainingLibraries())
sys.path.insert(0, sys.argv[1])
start = time.perf_counter()
from pred import predict_all
batch = predict_all('batch.csv')
seconds = time.perf_counter() - start
print(json.dumps(dict(batch=batch, seconds=seconds, reordered=predict_all('reordered.csv'),
                      short=predict_all('short.csv'), empty=predict_all('empty.csv'))))
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        batch.to_csv(path / 'batch.csv', index=False)
        combined[list(reversed(QUESTIONS))].to_csv(path / 'reordered.csv', index=False)
        combined.rename(columns=dict(zip(QUESTIONS, FEATURES))).to_csv(path / 'short.csv', index=False)
        empty.iloc[:0].to_csv(path / 'empty.csv', index=False)
        proc = subprocess.run([sys.executable, '-I', '-c', script, str(folder.resolve())],
                              cwd=path, capture_output=True, text=True, timeout=60, check=True)
    result = json.loads(proc.stdout)
    assert result['batch'] == expected[:60]
    assert result['reordered'] == result['short'] == expected
    assert result['empty'] == []
    print(json.dumps(dict(python=sys.version.split()[0], python_310_syntax=True,
        submission_bytes=size, predictions_in_timed_batch=len(result['batch']),
        cold_start_batch_seconds=result['seconds'], independent_parity_rows=len(expected),
        import_allowlist=sorted(imports), reordered_headers=True, short_headers=True,
        empty_csv=True, missing_and_unknown=True), indent=2))

if __name__ == '__main__':
    main()
