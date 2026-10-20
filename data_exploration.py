# %% [markdown]
# # CSC311 Project - Data Exploration
#
# Run top to bottom (`python data_exploration.py`) or cell by cell in Cursor.
# Figures go to `figures/`, tables to `tables/`, and the respondent-level
# train/val/test split to `splits/split_ids.csv`.
#
# Everything after the split cell uses the TRAINING split only.

# %%
import re
import sys
from pathlib import Path

import matplotlib

INTERACTIVE = "ipykernel" in sys.modules
if not INTERACTIVE:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = ROOT / "csc311_project_training_202609.csv"
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
SPLIT_DIR = ROOT / "splits"
for d in (FIG_DIR, TAB_DIR, SPLIT_DIR):
    d.mkdir(exist_ok=True)

SEED = 311
CLASSES = ["Baseball", "Basketball", "Soccer"]
CLASS_COLORS = {"Baseball": "#4C72B0", "Basketball": "#DD8452", "Soccer": "#55A868"}
sns.set_theme(style="whitegrid", context="paper")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    if INTERACTIVE:
        plt.show()
    plt.close(fig)


def save_table(df, name):
    df.to_csv(TAB_DIR / f"{name}.csv")
    print(f"\n=== {name} ===")
    print(df.to_string())


# %% [markdown]
# ## 1. Load data and name the features

# %%
raw = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)

SHORT_NAMES = {
    raw.columns[0]: "id",
    raw.columns[1]: "ease",
    raw.columns[2]: "strategy",
    raw.columns[3]: "injury",
    raw.columns[4]: "individual",
    raw.columns[5]: "where",
    raw.columns[6]: "season",
    raw.columns[7]: "music",
    raw.columns[8]: "food",
    raw.columns[9]: "colour",
    raw.columns[10]: "song",
    raw.columns[11]: "feel",
    raw.columns[12]: "pay",
    raw.columns[13]: "label",
}
df = raw.rename(columns=SHORT_NAMES)
df["id"] = df["id"].astype(int)
for c in df.columns:
    if c != "id":
        df[c] = df[c].str.replace("\xa0", " ").str.strip()

NUMERIC = ["ease"]
LIKERT = ["strategy", "injury", "individual"]
MULTI = ["where", "season", "music"]
TEXT = ["food", "colour", "song", "feel"]
NUMERIC_TEXT = ["pay"]
FEATURES = NUMERIC + LIKERT + MULTI + TEXT + NUMERIC_TEXT

feature_types = pd.DataFrame(
    {
        "type": ["numeric (0-10 rating)"]
        + ["Likert (5 levels)"] * 3
        + ["multi-select"] * 3
        + ["free text"] * 4
        + ["free text containing a number"],
        "question": [
            {v: k for k, v in SHORT_NAMES.items()}[f][:90] for f in FEATURES
        ],
    },
    index=FEATURES,
)
save_table(feature_types, "01_feature_types")
print(f"\nrows={len(df)}, respondents={df['id'].nunique()}")
print("rows per respondent:", df.groupby("id").size().value_counts().to_dict())
print("sports per respondent:", df.groupby("id")["label"].nunique().value_counts().to_dict())

# %% [markdown]
# ## 2. Split by respondent BEFORE looking at anything else
#
# Every respondent answered once for each of the three sports, so their rows
# share a personal answering style. Splitting by `id` keeps all three rows of a
# person in the same split; otherwise validation/test scores would be inflated.

# %%
rng = np.random.default_rng(SEED)
ids = rng.permutation(np.sort(df["id"].unique()))
n_train = int(round(0.70 * len(ids)))
n_val = int(round(0.15 * len(ids)))
split_of = {i: "train" for i in ids[:n_train]}
split_of.update({i: "val" for i in ids[n_train : n_train + n_val]})
split_of.update({i: "test" for i in ids[n_train + n_val :]})
df["split"] = df["id"].map(split_of)

pd.DataFrame({"id": list(split_of), "split": list(split_of.values())}).sort_values(
    "id"
).to_csv(SPLIT_DIR / "split_ids.csv", index=False)

split_summary = pd.crosstab(df["split"], df["label"], margins=True)
split_summary["respondents"] = df.groupby("split")["id"].nunique()
split_summary.loc["All", "respondents"] = df["id"].nunique()
save_table(split_summary.loc[["train", "val", "test", "All"]], "02_split_summary")

train = df[df["split"] == "train"].copy().reset_index(drop=True)
print(f"\nExploration uses train only: {len(train)} rows, {train['id'].nunique()} respondents")

# %% [markdown]
# ### Evidence that rows from the same respondent are dependent
# Share of the variance of `ease` that is explained by who answered.

# %%
ease_tmp = pd.to_numeric(train["ease"], errors="coerce")
person_mean = ease_tmp.groupby(train["id"]).transform("mean")
r2_person = 1 - ((ease_tmp - person_mean) ** 2).sum() / ((ease_tmp - ease_tmp.mean()) ** 2).sum()
print(f"R^2 of ease explained by respondent id: {r2_person:.2f}")

# %% [markdown]
# ## 3. Class balance

# %%
fig, ax = plt.subplots(figsize=(4, 2.6))
counts = train["label"].value_counts().reindex(CLASSES)
ax.bar(CLASSES, counts.values, color=[CLASS_COLORS[c] for c in CLASSES])
for i, v in enumerate(counts.values):
    ax.text(i, v + 3, str(v), ha="center")
ax.set_ylabel("rows (train)")
ax.set_title("Class balance in the training split")
save(fig, "03_class_balance")

# %% [markdown]
# ## 4. Missing values
#
# Missing answers take many forms. Normalize them all to NaN before counting.
#   * empty strings / empty multi-select lists
#   * "none", "n/a", "idk", "no idea", "nothing in particular", ...
#   * "#NAME?" (an Excel formula error, i.e. a corrupted answer)
#   * "don't watch", "never watched it" (the respondent does not know the sport)

# %%
MISSING_PATTERNS = [
    r"",
    r"-+",
    r"\?+",
    r"#name\?",
    r"n/?a",
    r"none( in particular| really)?",
    r"nothing( in particular| particular| really| specific)?",
    r"no",
    r"idk",
    r"not sure",
    r"(i )?(have )?no idea",
    r"i (do not|don'?t|dont) know( any songs)?",
    r"no (particular|perticular) \w+",
    r"no \w+ in particular",
    r"no song( comes to mind)?",
    r"no food or drink",
    r"(i )?(don'?t|do not|never) (have )?watch(ed)?( it| this sport)?",
    r"i have never watched it",
    r"never have watched",
    r"remember the name",
]
MISSING_RE = re.compile(r"^(?:" + "|".join(MISSING_PATTERNS) + r")$")


def normalize_missing(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    key = re.sub(r"[.!\s]+$", "", str(value).strip().lower())
    key = key.replace("’", "'")
    return np.nan if MISSING_RE.match(key) else value


# Markers only apply to free text. "No particular type" is a real option of the
# music question, and "none"/"nothing" in pay means a price of 0 (handled in 6).
for c in FEATURES:
    train[c] = train[c].replace("", np.nan)
for c in TEXT:
    train[c] = train[c].map(normalize_missing)

raw_empty = (df.loc[df["split"] == "train", FEATURES] == "").mean()
missing_table = pd.DataFrame(
    {
        "empty string (%)": (100 * raw_empty).round(1),
        "after normalizing markers (%)": (100 * train[FEATURES].isna().mean()).round(1),
    }
)
missing_by_class = (
    train.groupby("label")[FEATURES].apply(lambda g: 100 * g.isna().mean()).T.round(1)
)
missing_by_class.columns = [f"missing % | {c}" for c in missing_by_class.columns]
missing_table = missing_table.join(missing_by_class)
save_table(missing_table, "04_missing_values")

n_missing = train[FEATURES].isna().sum(axis=1)
print("\nrows by number of missing features:", n_missing.value_counts().sort_index().to_dict())
print("completely empty rows (all 12 features missing):",
      train.loc[n_missing == len(FEATURES), ["id", "label"]].values.tolist())
train = train[n_missing < len(FEATURES)].reset_index(drop=True)
print(f"training rows after dropping empty rows: {len(train)}")

# %% [markdown]
# ## 5. Numeric and Likert features

# %%
LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}
train["ease_num"] = pd.to_numeric(train["ease"], errors="coerce")
non_integer_ease = train.loc[train["ease_num"] % 1 != 0, "ease"].dropna().tolist()
print("non-integer ease values:", non_integer_ease)
for c in LIKERT:
    unknown = set(train[c].dropna()) - set(LIKERT_MAP)
    assert not unknown, f"unexpected Likert values in {c}: {unknown}"
    train[f"{c}_num"] = train[c].map(LIKERT_MAP)

fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
ax = axes[0]
bins = np.arange(-0.5, 11.5, 1)
for cls in CLASSES:
    ax.hist(
        train.loc[train["label"] == cls, "ease_num"].dropna(),
        bins=bins,
        histtype="step",
        linewidth=1.8,
        color=CLASS_COLORS[cls],
        label=cls,
    )
ax.set_xlabel("ease (0 = very hard, 10 = very easy)")
ax.set_ylabel("count")
ax.set_title("Ease of starting")
ax.legend(fontsize=7)

for ax, c in zip(axes[1:], LIKERT):
    tab = pd.crosstab(train["label"], train[f"{c}_num"], normalize="index").reindex(CLASSES)
    bottom = np.zeros(len(CLASSES))
    palette = sns.color_palette("RdYlGn", 5)
    for level in range(1, 6):
        vals = tab.get(level, pd.Series(0, index=CLASSES)).values
        ax.barh(CLASSES, vals, left=bottom, color=palette[level - 1], label=str(level))
        bottom += vals
    ax.set_xlim(0, 1)
    ax.set_title(c)
    ax.set_xlabel("share of class")
axes[-1].legend(
    title="1=SD ... 5=SA", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title_fontsize=7
)
save(fig, "05_numeric_likert_by_class")

ORDINAL = ["ease_num"] + [f"{c}_num" for c in LIKERT]

# %% [markdown]
# ## 6. Parse the "willing to pay" free-text answer into a number
#
# Rules (applied in order):
#   1. Answers that rate on a "1 to 10 scale" answered a different question -> NaN.
#   2. Normalize: remove thousands separators, turn "1k" into 1000, spelled-out
#      numbers ("thirty", "forty five") into digits, "couple hundred" into 200.
#   3. If the first numbers form a range ("50-100", "200 to 300"), use the midpoint.
#   4. Otherwise use the first number.
#   5. No number but a clear refusal ("not willing", "nothing", "zero") -> 0.
#   6. Anything else is unparseable -> NaN.

# %%
WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
REFUSAL_RE = re.compile(
    r"\b(none|nothing|zero|unwilling|not willing|no willing|would not pay|wouldn'?t pay|"
    r"won'?t|not really willing|pay me|no i don'?t)\b"
)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\$?\s*(?:-|to|~)\s*\$?\s*(\d+(?:\.\d+)?)")


def words_to_digits(text):
    def repl(m):
        tens, unit = m.group(1), m.group(2)
        return str(WORD_NUMS[tens] + (WORD_NUMS[unit] if unit else 0))

    tens_words = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
    unit_words = "one|two|three|four|five|six|seven|eight|nine"
    text = re.sub(rf"\b({tens_words})(?:[\s-]({unit_words}))?\b", repl, text)
    return re.sub(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                  lambda m: str(WORD_NUMS[m.group(1)]), text)


def parse_pay(value):
    if not isinstance(value, str):
        return np.nan
    s = value.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    if "scale" in s:
        return np.nan
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = re.sub(r"(?<=\d) (?=000\b)", "", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", lambda m: str(float(m.group(1)) * 1000), s)
    s = re.sub(r"\ba couple hundreds?\b|\bcouple hundred\b", "200", s)
    s = words_to_digits(s)
    rng_match = RANGE_RE.search(s)
    first_num = NUM_RE.search(s)
    if rng_match and first_num and rng_match.start() <= first_num.start():
        lo, hi = float(rng_match.group(1)), float(rng_match.group(2))
        return (lo + hi) / 2
    if first_num:
        return float(first_num.group())
    if REFUSAL_RE.search(s):
        return 0.0
    return np.nan


train["pay_num"] = train["pay"].map(parse_pay)
answered = train["pay"].notna()
unparsed = train.loc[answered & train["pay_num"].isna(), "pay"]
print(f"pay answered: {answered.sum()}, parsed: {train['pay_num'].notna().sum()}, "
      f"unparseable: {len(unparsed)}")
print("unparseable examples:", unparsed.str[:70].tolist())

examples = [
    "$1,000 ", "$200-$300", "At most 500 CAD.", "1k", "Thirty dollars",
    "Forty five dollars", "$10 000", "Not willing", "Hundreds of dollars",
    "1 to 10 scale i will choose 2 since im a person that enjoy watching sports online",
]
pay_examples = pd.DataFrame({"raw": examples, "parsed": [parse_pay(e) for e in examples]})
save_table(pay_examples.set_index("raw"), "06_pay_parsing_examples")

# %% [markdown]
# ### Outliers in pay
# The distribution is extremely right-skewed, so outliers are judged on the log
# scale with the 1.5 * IQR rule, and the raw value is capped at the 99th
# percentile of the training split. Zeros are genuine refusals, so only the
# upper fence is used.

# %%
pay = train["pay_num"]
log_pay = np.log1p(pay)
q1, q3 = log_pay.quantile([0.25, 0.75])
iqr = q3 - q1
hi_fence = np.expm1(q3 + 1.5 * iqr)
cap = pay.quantile(0.99)
pay_stats = pd.DataFrame(
    {
        "value": [
            pay.count(), round(100 * pay.isna().mean(), 1),
            pay.median(), pay.mean(), pay.max(), (pay == 0).sum(),
            round(hi_fence, 1), (pay > hi_fence).sum(), cap, (pay > cap).sum(),
        ]
    },
    index=[
        "parsed answers", "missing or unparseable (%)",
        "median", "mean", "max", "answers equal to 0",
        "log-IQR upper fence", "answers above upper fence",
        "99th percentile (cap)", "answers above cap",
    ],
)
save_table(pay_stats, "06_pay_outliers")
train["log_pay"] = np.log1p(pay.clip(upper=cap))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.0))
axes[0].hist(pay.dropna(), bins=50, color="grey")
axes[0].set_title("pay (parsed, raw scale)")
axes[0].set_xlabel("CAD")
axes[1].hist(train["log_pay"].dropna(), bins=30, color="grey")
axes[1].set_title("log(1 + pay), capped at 99th pct")
axes[1].set_xlabel("log(1 + CAD)")
sns.boxplot(data=train, x="label", y="log_pay", order=CLASSES, hue="label",
            palette=CLASS_COLORS, ax=axes[2], legend=False)
axes[2].set_title("log(1 + pay) by class")
axes[2].set_xlabel("")
save(fig, "06_pay_distribution")
ORDINAL.append("log_pay")

# %% [markdown]
# ## 7. Separation of the ordinal/numeric features
#
# For each feature: class means, pooled within-class standard deviation, and the
# largest gap between class means divided by that standard deviation. A ratio
# well below ~0.3 means the classes overlap almost completely.

# %%
rows = {}
for f in ORDINAL:
    g = train.groupby("label")[f]
    means = g.mean().reindex(CLASSES)
    pooled_sd = np.sqrt((g.var() * (g.count() - 1)).sum() / (g.count().sum() - len(CLASSES)))
    rows[f] = {
        **{f"mean | {c}": round(means[c], 2) for c in CLASSES},
        "within-class SD": round(pooled_sd, 2),
        "max gap / SD": round((means.max() - means.min()) / pooled_sd, 2),
    }
separation = pd.DataFrame(rows).T
save_table(separation, "07_ordinal_separation")

corr = train[ORDINAL].corr(method="spearman")
fig, ax = plt.subplots(figsize=(4.6, 3.8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title("Spearman correlation (train)")
save(fig, "07_ordinal_correlation")

# %% [markdown]
# ## 8. Multi-select features -> multi-hot
# Share of each class that ticked each option. Also flag near-constant columns.

# %%
OPTIONS = {}
for c in MULTI:
    opts = sorted({o.strip() for v in train[c].dropna() for o in v.split(",")})
    OPTIONS[c] = opts
    for o in opts:
        train[f"{c}={o}"] = train[c].map(
            lambda v, o=o: np.nan if not isinstance(v, str) else float(o in [x.strip() for x in v.split(",")])
        )
MULTI_COLS = [f"{c}={o}" for c in MULTI for o in OPTIONS[c]]
print("multi-hot columns:", len(MULTI_COLS))

share = train.groupby("label")[MULTI_COLS].mean().reindex(CLASSES).T
share["overall"] = train[MULTI_COLS].mean()
share["max-min gap"] = share[CLASSES].max(axis=1) - share[CLASSES].min(axis=1)
share = share.round(2)
save_table(share, "08_multiselect_share_by_class")

near_constant = share[(share["overall"] > 0.9) | (share["overall"] < 0.05)]
print("\nnear-constant multi-hot columns (overall share > 0.9 or < 0.05):")
print(near_constant[["overall"]].to_string())

fig, ax = plt.subplots(figsize=(5.2, 7.0))
sns.heatmap(share[CLASSES], annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "share of class that selected the option"})
ax.set_title("Multi-select options by class (train)")
ax.set_ylabel("")
save(fig, "08_multiselect_by_class")

# %% [markdown]
# ## 9. Free-text features
#
# Lower-case, drop the anonymization placeholders "[this sport]" and
# "[a competition]", tokenize into words, and count each word at most once per
# answer. For each feature: vocabulary size and the most class-specific words.

# %%
PLACEHOLDER_RE = re.compile(r"\[(this sport|a competition)\]", flags=re.I)


def clean_text(v):
    if not isinstance(v, str):
        return ""
    return PLACEHOLDER_RE.sub(" ", v.lower())


text_summary = {}
top_words = {}
for c in TEXT:
    docs = train[c].map(clean_text)
    vec_all = CountVectorizer(binary=True, stop_words="english", token_pattern=r"(?u)\b[a-z][a-z]+\b")
    X_all = vec_all.fit_transform(docs)
    vec_min3 = CountVectorizer(binary=True, stop_words="english", token_pattern=r"(?u)\b[a-z][a-z]+\b", min_df=3)
    vec_min3.fit(docs)
    lengths = docs.str.split().str.len()
    text_summary[c] = {
        "vocab size": X_all.shape[1],
        "vocab size (min_df=3)": len(vec_min3.vocabulary_),
        "median words per answer": float(lengths[train[c].notna()].median()),
        "distinct raw answers": train[c].dropna().str.lower().nunique(),
    }
    words = pd.DataFrame(X_all.toarray(), columns=vec_all.get_feature_names_out())
    by_class = words.groupby(train["label"].values).sum().reindex(CLASSES).T
    by_class["total"] = by_class.sum(axis=1)
    by_class = by_class[by_class["total"] >= 8]
    by_class["purity"] = by_class[CLASSES].max(axis=1) / by_class["total"]
    by_class["class"] = by_class[CLASSES].idxmax(axis=1)
    top_words[c] = by_class.sort_values("total", ascending=False).head(15)

save_table(pd.DataFrame(text_summary).T, "09_text_summary")
for c in TEXT:
    save_table(top_words[c], f"09_top_words_{c}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, c in zip(axes, TEXT):
    t = top_words[c].head(10)
    sns.heatmap(t[CLASSES].div(t["total"], axis=0), annot=t[CLASSES], fmt="d", cmap="Greens",
                vmin=0, vmax=1, cbar=False, ax=ax, annot_kws={"fontsize": 9})
    ax.set_title(f"{c}: 10 most frequent words", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(labelsize=9)
fig.suptitle("Word counts by class (colour = share of the word's occurrences)", y=1.01)
save(fig, "09_text_words_by_class")

# %% [markdown]
# ## 10. Which feature groups are predictive?
#
# Quick check on the TRAINING split only: 5-fold cross-validation grouped by
# respondent, using one feature group at a time with a lightly regularized
# softmax regression. Chance level is 1/3.

# %%
def group_matrix(name):
    if name in ORDINAL:
        x = train[[name]]
        return x.fillna(x.median()).values
    if name in MULTI:
        cols = [f"{name}={o}" for o in OPTIONS[name]]
        return train[cols].fillna(0).values
    vec = CountVectorizer(binary=True, stop_words="english", token_pattern=r"(?u)\b[a-z][a-z]+\b", min_df=2)
    return vec.fit_transform(train[name].map(clean_text)).toarray()


cv = GroupKFold(n_splits=5)
scores = {}
for name in ORDINAL + MULTI + TEXT:
    X = group_matrix(name)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    s = cross_val_score(model, X, train["label"], groups=train["id"], cv=cv, scoring="accuracy")
    scores[name] = {"n columns": X.shape[1], "CV accuracy": round(s.mean(), 3), "CV std": round(s.std(), 3)}
single_group = pd.DataFrame(scores).T.sort_values("CV accuracy", ascending=False)
save_table(single_group, "10_single_feature_group_accuracy")

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.barh(single_group.index[::-1], single_group["CV accuracy"][::-1],
        xerr=single_group["CV std"][::-1], color="grey")
ax.axvline(1 / 3, color="red", linestyle="--", label="chance (1/3)")
ax.set_xlabel("grouped 5-fold CV accuracy using only this feature")
ax.set_xlim(0, 1)
ax.legend(loc="lower right")
ax.set_title("Predictive power of each feature on its own (train)")
save(fig, "10_single_feature_group_accuracy")

# %% [markdown]
# ## 11. Size of the encoded feature space

# %%
dims = {"ease": 1, "Likert (ordinal)": 3, "pay (log, capped) + missing flag": 2,
        "multi-hot": len(MULTI_COLS)}
for c in TEXT:
    dims[f"bag-of-words {c} (min_df=3)"] = text_summary[c]["vocab size (min_df=3)"]
dims_table = pd.Series(dims, name="columns").to_frame()
dims_table.loc["TOTAL"] = dims_table["columns"].sum()
save_table(dims_table, "11_encoded_dimensions")
print(f"\ntraining rows: {len(train)}; features/rows = {dims_table.loc['TOTAL', 'columns'] / len(train):.2f}")
