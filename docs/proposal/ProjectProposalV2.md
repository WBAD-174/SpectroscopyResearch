# Project Proposal (v2): Stability-Aware Region-Level Shape Representation for High-Dimensional Small-Sample Spectral Classification

## 1. Background and Motivation

High-dimensional spectral data (NIR/MIR) typically contain hundreds to thousands of wavelength variables, while the number of labeled samples is small. This small-sample, high-dimensional regime makes models prone to overfitting unstable, wavelength-level signals. Classical approaches — PCA/PLS for dimensionality reduction, and CARS, Lasso, or model-based importance for variable selection — are widely used, and a large family of **interval-based** methods (iPLS, moving-window PLS, iRF, ICO, iVISSA) already exploit the fact that spectra are continuous rather than a bag of independent points.

However, two gaps remain underexplored in the **classification** setting with limited samples:

1. **Most interval methods stop at *selection*.** They identify which continuous regions matter, then feed the *raw wavelengths* inside those regions into a downstream model. They rarely *encode* each region into a compact, physically meaningful descriptor. The step from "selected region" to "low-dimensional shape representation" is where feature selection meets representation learning, and it is comparatively rare in spectral classification.

2. **Stability is usually a means, not a reported outcome.** Spectral classification papers overwhelmingly report only accuracy/F1. Feature-selection *stability* — how consistently the same features/regions are chosen under data perturbation — is rarely measured systematically, even though it is the property most relevant to generalization in small-sample settings.

This project reframes spectral feature engineering from *point-wise wavelength selection* to *stability-aware, region-level shape representation*, and treats feature stability as a first-class evaluation dimension alongside predictive performance.

## 2. Research Question

In high-dimensional, small-sample spectral **classification**, can a compact **region-level shape representation**, built from stable continuous wavelength regions, achieve comparable or better predictive performance than point-wise wavelength selection while producing substantially **more stable and interpretable** features?

## 3. Relation to Prior Work (and how this differs)

This positioning is deliberate because several neighboring ideas already exist, and the contribution must be defined against them:

- **Interval / region selection is not new.** iPLS, moving-window PLS, iRF, ICO, and iVISSA all select continuous spectral intervals. There is also prior work on **geographical-origin classification** of botanicals using interval-based local variable selection on NIR spectra with multi-region, multi-class settings closely resembling tea-origin tasks. *Difference:* this project does not claim novelty in "using regions instead of points." It builds on it.

- **Ensemble / consistency-based importance is not new.** Requiring a feature to be selected by most base selectors is the textbook definition of ensemble feature selection. *Difference:* cross-method agreement is used here only as an internal **stability criterion**, not as the headline contribution.

- **Stability selection is not new.** Repeated resampling to score selection stability is well established in high-dimensional small-sample domains (e.g., genomics). *Difference:* this project applies it at the **region level** and, critically, reports stability as an **evaluation outcome** for spectral classification, where it is usually omitted.

- **What is comparatively underexplored:** the *combination* of (a) region-level aggregation of multi-method importance, (b) stability-based region selection, and (c) compact **shape-descriptor encoding** of each region, evaluated **on classification** with stability as a primary metric. The contribution is integration + a representation-learning framing, not any single component.

## 4. Proposed Approach

**Stage A - Wavelength-level importance under resampling.**
Within each training fold / bootstrap sample, estimate wavelength importance using several methods (PLS coefficients, Random Forest importance, XGBoost importance, Lasso coefficients, CARS selection frequency). Each resampling trial yields one importance profile per method.

**Stage B - Region-level aggregation.**
Divide the spectrum into continuous (optionally overlapping) windows. Aggregate wavelength importance into region-level scores (mean / max / area / smoothness-weighted), converting isolated point signals into continuous region signals.

**Stage C - Stability-based region selection.**
Across all folds/bootstraps and across methods, assign each region a **stability score** (how consistently it ranks as important). Select regions by combining predictive importance and stability, preferring regions that are both informative and reproducible.

**Stage D - Region-level shape representation (the core step).**
Encode each selected region into a small set of **shape descriptors** rather than keeping its raw wavelengths: mean intensity, area under the curve, max/min, local slope, first/second-derivative statistics, peak position/height, and local PCA components. The concatenated descriptors form the final low-dimensional representation.

**Stage E - Downstream classification.**
Train standard classifiers (PLS-DA, SVM, Logistic Regression, Random Forest, XGBoost, small MLP) on the shape representation and compare against baselines.

## 5. Claimed Contributions

1. A **stability-aware, region-level shape representation** pipeline for small-sample spectral classification that bridges interval selection and representation learning.
2. A systematic demonstration that region-level shape representation yields **more stable and interpretable** features than point-wise selection at comparable accuracy.
3. The introduction of **feature-stability metrics as a primary evaluation axis** for spectral classification, not just accuracy.
4. (Stretch) Evidence of **cross-dataset transferability** of the framework beyond a single spectral task.

## 6. Experimental Design

### 6.1 Datasets
- **Primary:** the tea-origin MIR/NIR dataset (already available; multi-class geographical authentication, small-sample/high-dimensional).
- **Secondary (stretch goal):** one public high-dimensional spectral classification dataset with a comparable small-sample profile, to test transferability. Selection criteria: continuous spectrum, classification labels, sample-to-dimension ratio in a similar range.

### 6.2 Protocol (leakage control is non-negotiable)
- Use **nested cross-validation**: inner loop for region selection + descriptor design + hyperparameters; outer loop for unbiased performance.
- **All** preprocessing (standardization, smoothing, derivative, baseline correction) and **all** importance estimation / region selection are fit on training folds only, then applied to held-out data. This is the single most common source of inflated results in spectral papers — guard it explicitly.

### 6.3 Baselines (each under the identical CV protocol)
1. Full-spectrum + classifier (no selection).
2. PCA/SVD features + classifier.
3. PLS-DA components.
4. CARS-selected wavelengths + classifier.
5. RF/XGBoost point-wise importance selection + classifier.
6. An **interval-based baseline** (e.g., iPLS or moving-window PLS) — essential, since "regions" is your space.
7. 1D-CNN on raw spectra (representation-learning reference point).

### 6.4 The two comparisons that carry the paper
- **Comparison 1 (representation):** region-level **shape descriptors** vs. (a) point-wise selected wavelengths and (b) raw wavelengths inside the *same* selected regions. This isolates the value of *encoding* regions as shapes, not just selecting them.
- **Comparison 2 (stability):** stability of selected features/regions for region-level vs. point-wise methods, under data perturbation.

### 6.5 Evaluation metrics
- **Predictive:** accuracy, macro-F1 (per-class F1 given multi-class origin); confusion matrices.
- **Generalization:** train–test gap, cross-validation variance.
- **Stability (primary, often-missing axis):** selected-feature/region consistency across folds and bootstraps, measured with a quantitative selection-stability index (e.g., Kuncheva index or Jaccard-based stability), reported with variance.
- **Compactness:** number of retained features / regions.
- **Interpretability:** whether selected regions map onto physically meaningful absorption bands.

### 6.6 Ablation studies
- With vs. without **region aggregation** (region-level vs. point-wise).
- With vs. without **stability selection** (stability + importance vs. importance only).
- With vs. without **shape descriptors** (descriptors vs. raw wavelengths in the region).
- Aggregation-strategy variants (mean/max/area/smoothness).
- Window size / overlap sensitivity.

### 6.7 Stretch: cross-dataset transferability
Apply the identical pipeline to the secondary dataset; report whether stability/compactness advantages persist. If the second dataset cannot be obtained or aligned in time, **ship the single-dataset study first** — do not block on this.

## 7. Scope and Honest Positioning
The novelty here is **integration + representation framing + stability-as-evaluation**, not a new core algorithm. The realistic target is a domain venue (e.g., Chemometrics and Intelligent Laboratory Systems, Food Control, or a spectroscopy/analytical journal), not a top ML conference. This is appropriate: it is publishable, it establishes a representation-learning direction the author can extend, and it provides concrete substance for graduate-level research conversations.

## 8. Expected Significance
If region-level shape representation matches point-wise accuracy while being markedly more stable, compact, and interpretable, it offers a more reliable alternative for scientific spectral datasets where labels are scarce, wavelengths are highly correlated, and reproducibility matters more than squeezing performance on a single split.