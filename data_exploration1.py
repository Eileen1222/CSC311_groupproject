"""CSC311 proposal: explore ONLY the training split.

Put this script beside csc311_project_training_202609.csv and run it.
Dependencies: pandas, numpy, matplotlib, scikit-learn.
No models are trained here. Outputs are written to exploration_outputs/.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save figures without opening windows.
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

FOLDER = Path(__file__).resolve().parent
OUT = FOLDER / 'exploration_outputs'
OUT.mkdir(exist_ok=True)
DATA = FOLDER / 'csc311_project_training_202609.csv'
# Also accept the filename used when downloading a second copy.
if not DATA.exists():
    DATA = FOLDER / 'csc311_project_training_202609(1).csv'

# 1. Load data and give the long questions readable short names.
df = pd.read_csv(DATA, keep_default_na=False)
NAMES = ['id', 'ease', 'strategy', 'injury', 'individual', 'place',
         'season', 'music', 'food', 'colour', 'song', 'feeling', 'price', 'label']
if len(df.columns) != len(NAMES) or df.columns[0] != 'id' or df.columns[-1] != 'label':
    raise ValueError('Unexpected CSV structure: check the column names/order.')
pd.DataFrame({'short_name': NAMES, 'original_question': df.columns}).to_csv(
    OUT / 'column_dictionary.csv', index=False)
df.columns = NAMES
if df['id'].eq('').any() or df['label'].eq('').any():
    raise ValueError('Respondent ID or label is missing.')
print('Rows and columns:', df.shape)
print('Number of respondents:', df['id'].nunique())

# 2. Keep the SAME grouped split used in our earlier code.
# Only IDs are shuffled: one person's rows cannot enter different splits.
ids = np.sort(df['id'].unique())
rng = np.random.default_rng(42)
rng.shuffle(ids)
n_train, n_val = int(.70 * len(ids)), int(.15 * len(ids))
train_ids = ids[:n_train]
val_ids = ids[n_train:n_train + n_val]
test_ids = ids[n_train + n_val:]
assert not (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids)
            or set(val_ids) & set(test_ids))
manifest = pd.DataFrame({'id': ids, 'split': ['train'] * n_train
                        + ['validation'] * n_val
                        + ['test'] * (len(ids) - n_train - n_val)})
manifest.to_csv(OUT / 'split_ids.csv', index=False)
print('\nData splits:')
for name, group_ids in [('Training', train_ids), ('Validation', val_ids), ('Test', test_ids)]:
    print(f'{name}: {len(group_ids)} respondents, {df.id.isin(group_ids).sum()} rows')
raw = df[df.id.isin(train_ids)].copy()
# Everything below uses training respondents only.

# 3. Normalize explicit missing markers, case and whitespace.
# "None", "no particular type", "neutral" and "nothing" can be meaningful
# answers, so do NOT automatically mark them missing.
MISSING = {'', 'na', 'n/a', 'nan', 'null', '-', '[]'}
def clean(value):
    if pd.isna(value):
        return np.nan
    value = re.sub(r'\s+', ' ', str(value)).strip().lower()
    return np.nan if value in MISSING else value

features = NAMES[1:-1]
train = raw.copy()
for column in features:
    train[column] = train[column].map(clean)
likert = ['strategy', 'injury', 'individual']
multi = ['place', 'season', 'music']
text = ['food', 'colour', 'song', 'feeling']
types = ['numeric'] + ['ordinal'] * 3 + ['multi-select'] * 3 + ['text'] * 4 + ['numeric text']
summary = pd.DataFrame(index=features)
summary['type'] = types
summary['missing_count'] = train[features].isna().sum()
summary['missing_pct'] = 100 * train[features].isna().mean()
summary['unique_nonmissing'] = train[features].nunique()
summary['most_common_pct'] = [100 * train[c].value_counts(normalize=True).iloc[0]
                               if train[c].notna().any() else np.nan for c in features]
summary.to_csv(OUT / 'feature_summary.csv', index_label='feature')
print('\nFeature summary (explicit missing markers normalized):')
print(summary.round(2).to_string())
# Repeated answers across people may be legitimate; report, do not delete.
print('\nExact duplicate training rows:', raw.duplicated().sum())
print('Repeated answer+label rows ignoring ID:', raw.drop(columns='id').duplicated().sum())
print('Repeated training (id, label) pairs:', raw.duplicated(['id', 'label']).sum())

counts = train.label.value_counts().sort_index()
pd.DataFrame({'count': counts, 'percentage': 100 * counts / len(train)}).to_csv(OUT / 'class_balance.csv')
print('\nTraining class counts:\n', counts.to_string())
# Missingness may itself carry class information.
train[features].isna().groupby(train.label).mean().T.mul(100).to_csv(OUT / 'missing_by_class_pct.csv')
for c in features:
    train[c].value_counts(dropna=False).rename_axis('response').reset_index(name='count').to_csv(
        OUT / f'frequencies_{c}.csv', index=False)

# 4. Encode numeric/ordinal features and audit invalid values.
order = {'strongly disagree': 1, 'disagree': 2, 'neutral': 3,
         'agree': 4, 'strongly agree': 5}
numeric = pd.DataFrame(index=train.index)
audit = []
numeric['ease'] = pd.to_numeric(train.ease, errors='coerce')
invalid_ease = train.ease.notna() & (numeric.ease.isna() | ~numeric.ease.between(0, 10))
for idx in train.index[invalid_ease]:
    audit.append({'row': idx, 'feature': 'ease', 'response': raw.loc[idx, 'ease'], 'reason': 'invalid 0-10 rating'})
numeric.loc[invalid_ease, 'ease'] = np.nan
for c in likert:
    numeric[c] = train[c].map(order)
    for idx in train.index[train[c].notna() & numeric[c].isna()]:
        audit.append({'row': idx, 'feature': c, 'response': raw.loc[idx, c], 'reason': 'unknown Likert option'})

# Price policy: bare/$ amounts assumed CAD; explicit foreign currencies left
# unparsed. Use midpoint for a simple range, and the stated amount for a cap.
# Ambiguous multi-scenario answers are left missing rather than guessed.
def parse_price(value):
    if pd.isna(value):
        return np.nan, 'missing'
    s = str(value).replace(',', '')
    if re.search(r'usd|us\$|\bus\b|rmb|cny|yuan|eur|gbp|inr|€|£|¥|₹|rupee|peso', s):
        return np.nan, 'explicit non-CAD currency; no conversion'
    if s.startswith('-') and re.match(r'^-\s*\d', s):
        return np.nan, 'negative price'
    numbers = re.findall(r'\d+(?:\.\d+)?', s)
    if not numbers:
        if re.search(r'\bzero\b|would not pay|wouldn.t pay|not willing to pay', s) or s in {'nothing', 'free'}:
            return 0., 'explicit zero'
        return np.nan, 'no reliable numeric amount'
    if re.search(r'\d\s*[km]\b', s):
        return np.nan, 'abbreviated amount; review manually'
    if len(numbers) == 1:
        return float(numbers[0]), 'single amount or stated cap; assumed CAD unless explicit'
    if len(numbers) == 2 and re.search(r'\d\s*\$?\s*(?:-|–|—|to)\s*\$?\s*\d', s):
        return float(np.mean([float(n) for n in numbers])), 'range midpoint; assumed CAD unless explicit'
    return np.nan, 'ambiguous multiple amounts'

parsed = train.price.map(parse_price)
numeric['price'] = parsed.map(lambda p: p[0])
price_audit = pd.DataFrame({'raw_response': raw.price, 'parsed_price': numeric.price,
                            'parsing_rule': parsed.map(lambda p: p[1])})
price_audit.to_csv(OUT / 'price_parsing_audit.csv', index_label='row')
pd.DataFrame(audit, columns=['row', 'feature', 'response', 'reason']).to_csv(OUT / 'invalid_answers.csv', index=False)
numeric.describe().T.to_csv(OUT / 'numeric_summary.csv')
numeric.groupby(train.label).agg(['mean', 'std', 'median', 'count']).to_csv(OUT / 'numeric_by_class.csv')
numeric.corr(method='spearman').to_csv(OUT / 'numeric_spearman_correlation.csv')
# Flag large prices with IQR, but retain them: expensive seats may be real.
q1, q3 = numeric.price.quantile([.25, .75])
upper = q3 + 1.5 * (q3 - q1)
price_audit.loc[numeric.price > upper].to_csv(OUT / 'high_price_review.csv', index_label='row')
print(f'\nPrice parsed: {numeric.price.notna().sum()}/{len(train)}; IQR upper fence: {upper:.2f}')
print('High prices are flagged for review, not automatically removed.')

# 5. Split multi-select answers into separate binary columns.
blocks = []
for c in multi:
    selections = train[c].fillna('').map(lambda s: {v.strip() for v in s.split(',') if v.strip()})
    options = sorted(set().union(*selections))
    encoded = pd.DataFrame({f'{c}::{option}': selections.map(lambda values: int(option in values))
                            for option in options}, index=train.index)
    encoded.mean().mul(100).rename('selected_pct').to_csv(OUT / f'{c}_option_distribution.csv')
    by_class = encoded.groupby(train.label).mean().T * 100
    by_class.to_csv(OUT / f'{c}_by_class_pct.csv')
    blocks.append(encoded)
    ax = by_class.plot.barh(figsize=(10, max(4, len(options) * .4)))
    ax.set(xlabel='Percentage of training rows in each class', ylabel='', title=f'{c.title()}: selection rates by class')
    plt.tight_layout(); plt.savefig(OUT / f'{c}_by_class.png', dpi=160); plt.close()

# 6. Inspect free-text lengths and train-only TF-IDF representations.
# Each field gets its own vocabulary; unseen words at prediction time are ignored.
# TF-IDF = term frequency * inverse document frequency, followed by L2 row
# normalization. sklearn uses smoothed IDF: log((1+n)/(1+document_frequency))+1.
text_lengths = pd.DataFrame(index=train.index)
vocabularies = {}
for c in text:
    responses = train[c].fillna('')
    text_lengths[c + '_words'] = responses.str.split().str.len()
    vectorizer = TfidfVectorizer(max_features=100, min_df=2, ngram_range=(1, 1))
    try:
        matrix = vectorizer.fit_transform(responses)  # Training ONLY.
    except ValueError as error:
        if 'empty vocabulary' not in str(error) and 'no terms remain' not in str(error).lower():
            raise
        continue
    words = vectorizer.get_feature_names_out()
    names = [f'{c}::tfidf::{word}' for word in words]
    frame = pd.DataFrame(matrix.toarray(), index=train.index, columns=names)
    blocks.append(frame)
    vocabularies[c] = {'terms': words.tolist(), 'idf': vectorizer.idf_.tolist()}
    means = frame.groupby(train.label).mean().T
    means.to_csv(OUT / f'{c}_tfidf_by_class.csv')
    top = {label: means[label].nlargest(10).index.str.split('::').str[-1].tolist() for label in means}
    print(f'\n{c}: most prominent TF-IDF words by class (descriptive, not proven predictive):\n{top}')
text_lengths.describe().T.to_csv(OUT / 'text_length_summary.csv')
text_lengths.groupby(train.label).mean().to_csv(OUT / 'text_length_by_class.csv')

# 7. Demonstrate preprocessing on training only; save learned settings.
# Missing indicators distinguish missing answers from actual zeros.
# Numeric: median imputation; log1p on nonnegative prices reduces skew;
# standardize numeric columns for a possible linear/distance-based model.
# Ordinal coding assumes equal spacing, which can later be compared with one-hot.
missing_flags = train[features].isna().astype(int).add_suffix('_missing')
missing_flags['price_missing'] = numeric.price.isna().astype(int)
missing_flags['ease_missing'] = numeric.ease.isna().astype(int)
for c in likert:
    missing_flags[c + '_missing'] = numeric[c].isna().astype(int)
medians = numeric.median().fillna(0.)
filled = numeric.fillna(medians)
filled['price'] = np.log1p(filled['price'])
means = filled.mean()
stds = filled.std(ddof=0).replace(0., 1.)
scaled = (filled - means) / stds
X = pd.concat([scaled, missing_flags] + blocks, axis=1)
assert np.isfinite(X.to_numpy()).all()
X.to_csv(OUT / 'train_features.csv', index_label='original_row')
train.label.to_csv(OUT / 'train_labels.csv', index_label='original_row')
settings = {'seed': 42, 'likert_order': order, 'numeric_medians': medians.to_dict(),
            'numeric_means_after_price_log1p': means.to_dict(),
            'numeric_stds_after_price_log1p': stds.to_dict(), 'text': vocabularies,
            'feature_columns': X.columns.tolist(),
            'note': 'Training demonstration only. Refit preprocessing inside each CV training fold.'}
(OUT / 'preprocessing_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
X.nunique().rename('unique_values').to_csv(OUT / 'encoded_feature_variation.csv')
print(f'\nEncoded training matrix: {X.shape[0]} rows x {X.shape[1]} features (ID and label excluded).')

# 8. Small set of figures for the proposal; tables above cover every feature.
fig, ax = plt.subplots(figsize=(6, 4))
counts.plot.bar(ax=ax, color=['#4477AA', '#EE6677', '#228833'])
ax.set(xlabel='Sport', ylabel='Training rows', title='Training class balance')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig(OUT / 'class_balance.png', dpi=160); plt.close()
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, c in zip(axes.flat, ['ease'] + likert):
    pd.crosstab(numeric[c], train.label, normalize='columns').mul(100).plot.bar(ax=ax)
    ax.set(xlabel=f'{c} score', ylabel='Percentage of nonmissing answers', title=c.title())
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig(OUT / 'ratings_by_class.png', dpi=160); plt.close()
fig, ax = plt.subplots(figsize=(7, 4))
for label in sorted(train.label.unique()):
    values = np.log1p(numeric.loc[train.label == label, 'price'].dropna())
    ax.hist(values, bins=np.arange(0, 14, .5), alpha=.45, label=label)
ax.set(xlabel='log(1 + parsed ticket price)', ylabel='Training rows', title='Ticket price distribution')
ax.legend(); plt.tight_layout(); plt.savefig(OUT / 'price_distribution.png', dpi=160); plt.close()
print('\nSaved tables and figures to:', OUT)
print('Next: inspect price_parsing_audit.csv and feature_summary.csv, then interpret the plots.')
print('Validation/test answers were not explored. No model was trained.')
