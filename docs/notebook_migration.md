# Migration report: notebooks → package

Source of truth: `livrable1_groupe2_classification_photo_autre.ipynb`,
`Livrable_2_Debruitage_2.ipynb`, `Livrable_3_Captioning.ipynb` (all cells and outputs read).
The notebooks are kept unchanged in `notebooks/` for reference; the package does not use them.

## 1. Pipeline reconstructed from the notebooks

| Step | Notebook | Package |
|---|---|---|
| Inventory of 5 class folders, `Photo` → 1 | L1 §2 | `classification.data.build_inventory` |
| Exclusion of `Painting/painting_02662.jpg`, unreadable images dropped | L1 §2 | `known_invalid_files` + `build_image_cache` |
| PIL draft decode, RGB, 180×180 bilinear, JPEG q95 cache (MD5 of path) | L1 §2 | `load_and_resize`, `build_image_cache` |
| Optional stratified sub-sample (`DATA_FRACTION`) | L1 §2 | `subsample` / `classification.data_fraction` |
| Stratified 70/15/15 split on source class, seed 42, disjointness asserts | L1 §4 | `split_dataset`, `assert_disjoint` |
| tf.data pipeline, shuffle train only, `Rescaling` in model | L1 §5 | `make_dataset`, `build_classifier` |
| Balanced class weights on train | L1 §6 | `compute_class_weights` |
| Reference / regularised / 4-block CNN, Adam 3e-4, clipnorm 1 | L1 §7-8 | `build_classifier(variant)` |
| ReduceLROnPlateau + EarlyStopping (start epoch 15 for the 4-block CNN) | L1 §8 | `_fit`, `candidates[].early_stopping_start_epoch` |
| Model chosen on val F1 macro, threshold grid 0.10-0.90 on val | L1 §10 | `select_threshold` |
| Per-source-class error analysis, single test evaluation | L1 §12-14 | `source_class_analysis`, end of `train_classifier` |
| Save `.keras` + JSON parameters | L1 §16 | `classifier.keras`, `classifier_metadata.json` |
| L1 `Photo` folder (4 000 sampled) as training photos; L2 dataset halved into val/test | L2 §2 | `prepare_denoising_data`, `split_eval_paths` |
| Centre crop 128×128; near-duplicate removal | L2 §2 | `read_photo`, `near_duplicate_mask` |
| 3 noises × 3 levels; fixed seeds 1000/2000 for val/test | L2 §3 | `noise.py`, `fixed_noisy_sets` |
| Mean / gaussian / median 3×3 baselines, PSNR + SSIM | L2 §4 | `filters.py` |
| A0-A3 autoencoders | L2 §5 | `denoising.models` (parameter counts identical) |
| On-the-fly noisy pairs, random flip, random intensity, mixed noise | L2 §6, §10 | `training_dataset` |
| Architecture comparison (gaussian, BCE, lr 1e-3) | L2 §8 | `compare_architectures` / `tounum denoise compare` |
| Final A2 + MSE + lr 3e-3 + mixed noise, test on 9 cases, clean-image check | L2 §9-12 | `train_denoiser` |
| `debruiter`: reflect padding to a multiple of 4 | L2 §13 | `Denoiser.denoise` |
| COCO captions grouped per image, shuffle seed 42, test/val/train cut | L3 §2 | `load_captions`, `split_image_ids` |
| Cleaning, `<start>/<end>`, 5 000-word tokenizer, 99th-percentile length, post padding | L3 §3 | `captioning.text` |
| Frozen InceptionV3 / InceptionResNetV2, 64×C float16 feature cache | L3 §4 | `captioning.features` |
| Encoder, Bahdanau attention, GRU/LSTM decoder, initial state from image | L3 §5 (`captioning_modele.py`) | `captioning.model` (parameter counts identical) |
| Teacher forcing, masked loss, early stopping on val BLEU-4, best weights restored | L3 §6 | `fit_captioner` |
| Greedy vs sampling comparison; greedy retained | L3 §10.2 | `validation.greedy/sample` in metrics, `captioning.decoding` |
| Single test evaluation; save weights + JSON config | L3 §11, §13 | end of `train_captioner` |

The hyper-parameter sweeps of L3 §9 (one parameter at a time on 1 000 images, ~30 models) are
not automated: their outcome is the default `captioning.model` configuration (architecture B,
dropout 0.5), and any variant can be trained by editing the configuration.

## 2. Corrections (behaviour changes, and why)

| # | Issue in the notebook | Correction | Effect on results |
|---|---|---|---|
| 1 | **Train/serve skew (L1).** `predict_image` decoded with `tf.io.decode_image` and resized with `tf.image.resize`, whereas training images were resized by PIL (draft decoding), JPEG-encoded and decoded by TensorFlow. | `preprocess_for_model` reproduces the training path exactly (PIL resize → JPEG q95 → `tf.io.decode_jpeg`); a test asserts pixel equality. | Predictions on new images now see the same pixels as in training; evaluation metrics are unchanged (they were computed on the cache). |
| 2 | **Artifacts overwritten (L3).** The final model was saved under the same file names as the 20 000-image model, which had to be reloaded after a kernel restart; configurations were then retyped by hand. | Every run writes to a new `artifacts/<task>/<run>/` folder with `config_used.json`; overwriting is refused. | None on metrics; runs become traceable and reproducible end to end. |
| 3 | **Vocabulary fitted on 20 000 images while the final model trains on 40 000 (L3).** | `prepare_text` fits the vocabulary and the max length on the training images actually used. | May change word indices / max length slightly vs. the notebook; consistent with "fit on train only". |
| 4 | **JPEG-only decoding for captioning (L3).** `tf.image.decode_jpeg` fails on PNG scans in the end-to-end pipeline. | `tf.io.decode_image(..., expand_animations=False)`, identical on JPEG. | None on COCO (JPEG). |
| 5 | **Legacy Keras API (L3).** `keras.preprocessing.text.Tokenizer` is not part of Keras 3. | `Vocabulary`, a plain-Python re-implementation; a test checks index-for-index equality with the legacy tokenizer. | None. |
| 6 | **`BahdanauAttention` had no `get_config` (L3).** | Added. | None. |
| 7 | **Hard-coded, user-specific paths** (`/home/ylies/...`, `Path.home()/projet/...`, `../Livrable1`). | All paths in the YAML configuration, resolved relative to it; `tounum check-data`. | None. |
| 8 | **Feature extraction for both CNNs** even though one is used. | Only the configured CNN is extracted. | Faster; none on metrics. |

Not changed on purpose (documented as limitations): denoising loss and learning rate were tuned
on gaussian noise only and then reused for the mixed model; the L3 test set had been looked at
once before the final choice (stated in the notebook); single training run per configuration.

## 3. Reported results (from the notebook outputs, not re-run)

| Module | Metric (test) | Value |
|---|---|---|
| Classification | F1 macro / ROC-AUC / PR-AUC / accuracy | 0.9129 / 0.9783 / 0.9367 / 0.9367 |
| Classification | Threshold (chosen on validation) | 0.78 |
| Denoising (medium noise) | PSNR gaussian / salt & pepper / speckle | 28.7 / 32.4 / 28.2 dB |
| Denoising | Gain vs best filter | +3.4 / +3.9 / +3.8 dB |
| Captioning | BLEU-1 / BLEU-4 | 0.705 / 0.267 |

## 4. Verification performed during the refactor

* Parameter counts of all architectures equal those printed by the notebooks
  (classifier 8 023 361 / 4 353 601; denoisers 29 507 / 298 755 / 158 659 / 75 651;
  captioners 6 603 658 / 7 259 018 / 9 325 962).
* Vocabulary indices identical to the Keras tokenizer.
* End-to-end training, saving and reloading of the three modules on synthetic data.
* Full training on the real datasets: **not performed** (data and ImageNet weights unavailable
  in the refactoring environment).
