# Original research notebooks

These are the three notebooks of the original group project, kept **for reference** with their
executed outputs (figures, training logs and the results reported in the main README).
The package in `src/tounum/` reimplements them; it does **not** need them to run.

| Notebook | Module | Package |
|---|---|---|
| `01_classification_photo_other.ipynb` (Livrable 1) | photo / other CNN classifier | `tounum.classification` |
| `02_denoising_autoencoder.ipynb` (Livrable 2) | denoising autoencoders | `tounum.denoising` |
| `03_image_captioning.ipynb` (Livrable 3) | captioning by transfer learning | `tounum.captioning` |

They were run by the team on a local GPU machine; the hard-coded paths they contain
(e.g. `/home/<user>/projet/...`) are those of that machine. See
[`../docs/notebook_migration.md`](../docs/notebook_migration.md) for the mapping to the package
and the corrections applied.
