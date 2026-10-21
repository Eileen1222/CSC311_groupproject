# 3. Planned Methodology

## Model families and rationale

We will compare k-nearest neighbours (kNN), decision trees, and L2-regularized
softmax regression. They provide
distance-based, nonlinear rule-based, and linear approaches to the same
three-class task. The comparison will establish which assumptions best suit
the survey data rather than choosing a winner in advance.

**kNN.** Similar descriptions may correspond to the same sport. We will store
the training vectors and predict by uniform majority vote among the k closest
vectors using Euclidean distance. This requires no gradient optimizer. Larger
k reduces sensitivity to isolated noisy answers, while overly large k may
underfit. Numeric scaling and vocabulary filtering matter because distances
can be dominated by large scales or many irrelevant text dimensions. Vote ties
will use the fixed class order Baseball, Basketball, Soccer for reproducibility.

**Decision tree.** Venue, season and text indicators may interact with numeric
ratings through nonlinear rules. We will grow a binary classification tree
greedily by maximizing entropy reduction (information gain), using thresholds
between distinct feature values. A node stops splitting when it is pure,
no positive-gain valid split exists, or depth/leaf-size limits are reached.
Leaves predict their majority class; no gradient optimizer is needed. Depth
and minimum leaf size constrain memorization of rare responses. A single tree
also allows interpretable rules and compact prediction code.

**Softmax regression.** A regularized linear classifier can combine many weak
TF-IDF and multi-hot signals without fitting a separate rule for every
combination. It predicts the largest of three linear scores, Wx+b. Training
minimizes mean multiclass cross-entropy plus (lambda/2)||W||_F^2, excluding
the bias from the penalty. We will use full-batch gradient descent, initialize
weights and biases to zero, and stop after at most 5,000 updates or when the
relative training-objective change is below 1e-7 for 20 consecutive updates.
Learning rate and penalty strength will be tuned. Non-finite losses, increases
in the objective, and iteration-limit termination will be logged and checked
before accepting a setting. This model may miss feature interactions captured
by the tree.

## Data representation and validation

We will retain the respondent partition in Data Exploration: seed 42, with
203/43/44 respondents and 609/129/132 rows for training/validation/internal
test. All three answers from a person remain together. The respondent ID is
used only for splitting, and neither ID nor label enters the predictors.

The starting representation is the one described in Data Exploration: median
imputation, log-transformed prices, standardized numeric/ordinal features,
multi-hot option indicators, separate TF-IDF unigrams for the four text fields,
and missingness flags. Each text field has at most 100 terms. Binary indicators
and TF-IDF vectors are not standardized. Prices above the IQR fence are
retained, as in the EDA. Standardization preserves the ordering of numeric
values and therefore does not change the tree's possible threshold partitions.

We will use the same five GroupKFold partitions of the training respondents for
every model and setting. Each fold will fit its own medians, means/standard
deviations, multi-select vocabulary, text vocabulary and IDF using only its
training respondents. Its held-out fold will only be transformed. We will
not cross-validate the precomputed train_features.csv, whose preprocessing
was fitted on the full training partition. The held-out validation and internal
test partitions will not be used to fit preprocessing during tuning.

## Hyperparameter search and model selection

We will tune all three families using exhaustive grid search. In addition to
the model parameters below, every family will compare text minimum document
frequency min_df in {2, 3, 5}, keeping max_features=100 per text field and
unigrams fixed. These are proposed ranges; no tuning outcome is claimed here.

| Model | Model parameters to search | Settings including min_df |
|---|---|---|
| kNN | k in {1, 3, 5, 9, 15}; Euclidean distance; uniform vote | 15 |
| Decision tree | max_depth in {3, 5, 8, None}; min_samples_leaf in {2, 5, 10}; entropy criterion | 36 |
| Softmax regression | learning rate in {0.01, 0.05, 0.1}; lambda in {0.0001, 0.001, 0.01, 0.1} | 36 |

The grid contains 87 settings, or 435 fold fits. The k range spans highly local
to smoother decisions on roughly 487 training rows per fold. Tree settings
span shallow rules to unrestricted depth, with minimum leaf sizes guarding
against tiny terminal groups. Log-spaced penalties cover weak through stronger
shrinkage for 390 starting features and 609 training rows; the learning rates
allow a convergence check across several step sizes. min_df tests rare-word
noise and vocabulary size. If a best setting lies at a boundary or fails to
converge, any revision will be documented and assessed using training CV
before consulting held-out validation outcomes.

Within each family, we will select the setting with the highest mean CV
macro-F1, breaking exact ties by mean accuracy and then a fixed preference
for simpler settings: larger k; smaller depth/larger leaves; or larger lambda.
Remaining ties use a fixed parameter order. We will refit the three family
champions on training and compare them once on the held-out validation set.
Validation macro-F1 selects the final family; accuracy and then smaller
prediction artifacts break exact ties. We will log settings, fold scores,
convergence status and runtimes for every family, rather than tuning only
the eventual winner.

## Evaluation and planned prediction script

We will report accuracy, macro-F1, per-class precision/recall, and confusion
matrices. Accuracy measures the proportion of correctly identified sports.
Macro-F1 is the unweighted mean of the three class F1 scores, where
F1_c = 2TP_c/(2TP_c+FP_c+FN_c), defined as zero for a zero denominator. It
penalizes a classifier that performs well on one sport while neglecting
another. CV means and standard deviations describe consistency across
respondent groups; the standard deviation is not a confidence interval.
Confusion matrices will use true labels as rows and predicted labels as
columns, with counts and row-normalized recalls to reveal systematic errors.

After fixing the model and settings, we will refit on training plus validation
and evaluate the reserved internal test once. We will inspect validation
errors for common confusion patterns and missing/unknown answers. The final
report will compare the tuned families and give one numerical estimate of
test accuracy supported by held-out evidence, without inventing that number
at the proposal stage. We will distinguish the student internal split from
the unavailable instructor/TA test: different respondent populations may
reduce generalization, so student scores cannot guarantee hidden-test accuracy.

For the later prediction submission, we will export the chosen model and
preprocessing parameters and implement inference using only standard Python,
NumPy and pandas. kNN needs distances and voting, a tree needs node traversal,
and softmax needs linear scores. predict_all(filename) will return one exact
sport label per input row in the original order, including rows with missing
answers. We will check agreement with the development model, standalone
execution, unseen words, total submitted size at most 10 MB, and approximately
60 predictions within one minute. This proposal specifies the plan; final
training and prediction implementation remain later project work.

