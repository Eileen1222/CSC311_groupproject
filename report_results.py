"""Render the report's numerical results from the saved experiment records."""
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
out = Path(sys.argv[2])
def read(name):
    return json.loads((run / (name + '.json')).read_text())

def percent(v, suffix=False):
    return f'{100*v:.1f}' + (r'\%' if suffix else '')

def setting(c):
    df = fr'd_f={c["df"]}'
    if c['family'] == 'knn':
        return fr'$k={c["k"]},\,{df},\,w={c["text_weight"]}$'
    if c['family'] == 'tree':
        d = c['depth'] if c['depth'] is not None else r'\infty'
        return fr'$d={d},\,\ell={c["leaf"]},\,{df}$'
    return fr'$\eta={c["rate"]:.2f},\,\lambda={c["penalty"]:g},\,{df}$'

selection, validation, test, refit = map(read, ('cv_selection', 'validation', 'internal_test', 'refit'))
winner = test['config']['family']
v = validation[winner]
long_names = dict(knn=r'$k$-nearest neighbours model', tree='entropy decision tree',
                  softmax='L2-regularized softmax model')
short_names = dict(knn=r'$k$NN', tree='Tree', softmax='Softmax')
why = {
    'softmax': 'Combining evidence across questions appears to have helped it classify more consistently than a local vote or a single tree.',
    'knn': 'The result suggests that nearby response profiles retained useful distinctions between sports.',
    'tree': 'The result suggests that a small set of threshold rules captured useful distinctions between sports.',
}
commands = {}
commands['Winner'] = long_names[winner]
commands['SummaryWhy'] = why[winner]
commands['TestAcc'] = f'{100*test["accuracy"]:.2f}' + r'\%'
commands['TestF'] = f'{100*test["macro_f1"]:.2f}' + r'\%'
commands['TestCorrect'] = str(sum(test['confusion'][i][i] for i in range(3)))
commands['WinValAcc'] = percent(v['accuracy'], True)
commands['ExtensionOutcome'] = ' '.join(
    f'{short_names[f]} proposal/expanded best CV macro-F1 was '
    f'{percent(s["proposal"]["macro_f1_mean"], True)}/{percent(s["expanded"]["macro_f1_mean"], True)}.'
    for f, s in selection.items() if f != 'tree'
)
commands['FamilyRows'] = ''
commands['ClassRows'] = ''
commands['MatrixRows'] = ''
for family, val in validation.items():
    cv = selection[family]['chosen']
    cv_fmt = [percent(cv[m + '_mean']) + r' $\pm$ ' + percent(cv[m + '_sd'])
              for m in ('accuracy', 'macro_f1')]
    commands['FamilyRows'] += ' & '.join(
        [short_names[family], setting(val['config']), *cv_fmt,
         percent(val['accuracy']), percent(val['macro_f1'])]) + r'\\'
for name, item in [(f'Validation {short_names[f]}', record) for f, record in validation.items()] + [('Internal test', test)]:
    pr = [percent(value) for pair in zip(item['precision'], item['recall']) for value in pair]
    commands['ClassRows'] += ' & '.join([name, *pr]) + r'\\'
sports = ['Baseball', 'Basketball', 'Soccer']
for i, sport in enumerate(sports):
    row = [f'{n} ({percent(p, True)})' for item in (v, test)
           for n, p in zip(item['confusion'][i], item['row_normalized'][i])]
    commands['MatrixRows'] += ' & '.join([sport, *row]) + r'\\'
runner_name, runner = max(((k, x) for k, x in validation.items() if k != winner),
                          key=lambda item: item[1]['macro_f1'])
commands['SelectionWhy'] = (
    f'{short_names[winner]} led held-out macro-F1 by '
    f'{percent(v["macro_f1"] - runner["macro_f1"])} points over {short_names[runner_name]}.')
soft = validation['softmax']['diagnostics']
status = 'converged' if soft['status'] == 'converged' else 'reached its update cap'
commands['ConvergenceLine'] = (
    f'The softmax champion {status} after {soft["updates"]:,} updates on training, '
    f'with no recorded objective increases.')
if winner == 'softmax':
    final_status = 'converged' if refit['status'] == 'converged' else 'reached the cap'
    commands['ConvergenceLine'] += f' The final refit {final_status} at {refit["updates"]:,} updates.'
def error_sentence(item, split):
    counts = [(item['confusion'][i][j], i, j) for i in range(3) for j in range(3) if i != j]
    count = max(entry[0] for entry in counts)
    winners = [(sports[i], sports[j]) for n, i, j in counts if n == count]
    pairs = ' and '.join(f'{actual} predicted as {predicted}' for actual, predicted in winners)
    return f'{split} most common error{"s" if len(winners)>1 else ""} ({count} {'each' if len(winners)>1 else 'rows'}) involved {pairs}.'
strata = v['error_strata']
missing, oov = strata['any_missing'], strata['any_text_oov']
commands['ErrorLine'] = (
    error_sentence(v, 'Validation') + ' ' + error_sentence(test, 'Internal test') +
    f' Validation errors affected {missing["errors"]}/{missing["rows"]} rows with missing or unparseable answers '
    f'and {oov["errors"]}/{oov["rows"]} rows containing unseen text tokens. These overlapping groups are descriptive.'
)
out.write_text('\n'.join('\\newcommand{\\' + key + '}{' + val + '}'
                         for key, val in commands.items()) + '\n')
