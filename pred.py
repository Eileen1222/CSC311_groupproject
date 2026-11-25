"""Standalone CSC311 inference. Model state is stored beside this file."""
from collections import Counter
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

QUESTIONS = ('How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy.', 'Success in this sport depends heavily on strategy and planning.', 'Players must be comfortable with a high risk of injury.', 'Individual brilliance can outweigh teamwork in this sport.', 'Where might you play this sport?', 'During which seasons would you be most likely to play this sport recreationally in Toronto?', 'What types of music do you most associate with this sport?', 'What foods and drinks do you associate with watching this sport in the place or culture you know best?', 'What colour do you associate with this sport?', 'Which song do you associate with this sport?', 'How do you feel when you watch this sport?', 'How much would you be willing to pay for a decent seat at a professional competition of this sport?')
FEATURES = ('ease', 'strategy', 'injury', 'individual', 'place', 'season', 'music', 'food', 'colour', 'song', 'feeling', 'price')
CLASSES = ('Baseball', 'Basketball', 'Soccer')
NUMERIC = (*FEATURES[:4], 'price')
MULTI = FEATURES[4:7]
TEXT = FEATURES[7:11]
TOKEN = re.compile(r'(?u)\b\w\w+\b')
MISSING = {'', 'na', 'n/a', 'nan', 'null', '-', '[]'}
def clean(value):
    if pd.isna(value):
        return np.nan
    value = re.sub(r'\s+', ' ', str(value)).strip().lower()
    return np.nan if value in MISSING else value
order = {'strongly disagree': 1, 'disagree': 2, 'neutral': 3,
         'agree': 4, 'strongly agree': 5}
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


def read_csv(filename):
    """Accept the survey questions or their short names, in any column order."""
    raw = pd.read_csv(filename, keep_default_na=False)
    return normalize(raw), raw


def normalize(raw):
    raw = raw.rename(columns=lambda c: re.sub(r'\s+', ' ', str(c)).strip())
    columns = {}
    for short, question in zip(FEATURES, QUESTIONS):
        source = question if question in raw else short
        if source not in raw:
            # A missing column is treated as unanswered so prediction still runs.
            columns[short] = pd.Series(np.nan, index=raw.index, dtype=object)
            continue
        columns[short] = raw[source].map(clean)
    return pd.DataFrame(columns, index=raw.index)


def numeric_values(frame):
    ease = pd.to_numeric(frame['ease'], errors='coerce')
    values = [ease.where(ease.between(0, 10))]
    values.extend(frame[c].map(order) for c in NUMERIC[1:-1])
    values.append(frame['price'].map(lambda s: parse_price(s)[0]))
    return np.column_stack(values).astype(float)


def options(response):
    if pd.isna(response):
        return set()
    return {v.strip() for v in response.split(',') if v.strip()}


def transform(frame, state):
    numeric = numeric_values(frame)
    missing = ~np.isfinite(numeric)
    filled = np.where(missing, state['median'], numeric)
    filled[:, -1] = np.log1p(filled[:, -1])
    normalized = (filled - state['mean']) / state['std']
    flags = frame.loc[:, list(FEATURES)].isna().to_numpy()
    for j, col in enumerate(NUMERIC):
        flags[:, FEATURES.index(col)] = missing[:, j]
    blocks = [normalized, flags]
    for col in MULTI:
        choices = state['options'][col]
        encoded = np.zeros((len(frame), len(choices)))
        lookup = {choice: j for j, choice in enumerate(choices)}
        for i, response in enumerate(frame[col]):
            for choice in options(response):
                if choice in lookup:
                    encoded[i, lookup[choice]] = 1
        blocks.append(encoded)
    for col in TEXT:
        vocab = state['text'][col]
        lookup = {word: j for j, word in enumerate(vocab['terms'])}
        encoded = np.zeros((len(frame), len(lookup)))
        for i, response in enumerate(frame[col].fillna('')):
            for word, count in Counter(TOKEN.findall(response)).items():
                if word in lookup:
                    encoded[i, lookup[word]] = count
        encoded *= np.asarray(vocab['idf'])
        norm = np.linalg.norm(encoded, axis=1, keepdims=True)
        np.divide(encoded, norm, out=encoded, where=norm > 0)
        blocks.append(encoded)
    return np.column_stack(blocks)


def knn_predict(x, train, labels, k):
    # Work in batches and use stable training order on exact distance ties.
    output = np.empty(len(x), dtype=int)
    train_norm = np.einsum('ij,ij->i', train, train)
    for start in range(0, len(x), 64):
        batch = x[start:start + 64]
        distance = np.einsum('ij,ij->i', batch, batch)[:, None] + train_norm - 2 * (batch @ train.T)
        neighbors = np.argsort(np.maximum(distance, 0), axis=1, kind='stable')[:, :k]
        votes = np.column_stack([(labels[neighbors] == c).sum(axis=1) for c in range(len(CLASSES))])
        output[start:start + len(batch)] = votes.argmax(axis=1)
    return output


def model_predict(x, model):
    kind = model['kind']
    if kind == 'softmax':
        return (x @ np.asarray(model['weights']) + model['bias']).argmax(axis=1)
    if kind == 'knn':
        x[:, model['text_start']:] *= model['text_weight']
        return knn_predict(x, np.asarray(model['train']), np.asarray(model['labels']), model['k'])
    nodes = np.zeros(len(x), dtype=int)
    x = x.astype(np.float32)
    left, right, features, thresholds, labels = map(np.asarray, (
        model['left'], model['right'], model['feature'], model['threshold'], model['labels']))
    active = np.flatnonzero(left[nodes] != -1)
    while active.size:
        current = nodes[active]
        nodes[active] = np.where(x[active, features[current]] <= thresholds[current],
                                 left[current], right[current])
        active = active[left[nodes[active]] != -1]
    return labels[nodes]


def predict_all(filename):
    """Return one exact sport label per input row, preserving CSV order."""
    frame, _ = read_csv(filename)
    state = json.loads(Path(__file__).with_name('model.json').read_text(encoding='utf-8'))
    predictions = model_predict(transform(frame, state['preprocessing']), state['model'])
    return [CLASSES[i] for i in predictions]
