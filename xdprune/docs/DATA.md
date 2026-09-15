# Datasets and Domain Views

The study uses three independently licensed traffic datasets. Dataset content
is not redistributed in this repository.

| Study domain | Dataset | Purpose | Classes in the retained YAML |
|---|---|---|---:|
| GEN | MIO-TCD | Initial general-traffic study | 11 |
| SNOW | ACDC | Initial adverse/snow study | 8 |
| GEN2 | BDD100K clear view | Secondary matched-dataset study | 6 |
| NGN2 | BDD100K rainy, snowy, and foggy view | Secondary matched-dataset study | 6 |

Official dataset pages:

- [MIO-TCD](https://tcd.miovision.com/challenge/dataset.html)
- [ACDC](https://acdc.vision.ee.ethz.ch/)
- [BDD100K](https://bdd-data.berkeley.edu/)

## Expected layouts

Each YAML under `data_views/` is intentionally relative to its
own directory. Populate the corresponding `images/` and `labels/` directories
locally using YOLO-format labels:

```text
<view>/
├── dataset.yaml (or the supplied domain-specific YAML)
├── images/
│   ├── train/
│   ├── val/
│   └── test/       # only where defined by the source study
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The BDD100K study uses accessible training and validation annotations only;
the held-out test split is not used in the controlled pruning analysis.
The frozen BDD100K views contain 37,091/5,299 GEN2 train/validation images
and 10,727/1,504 NGN2 train/validation images. Their validation sets contain
61,603 and 17,733 annotated instances, respectively.

## Preparation and validation

The MIO-TCD and ACDC preparation utility accepts explicit source paths:

```bash
python scripts/prepare_detection_datasets.py --dataset both \
  --mio-root /path/to/mio --acdc-root /path/to/acdc \
  --output-root data_views --seed 42
```

Do not commit generated images, labels, caches, or provider annotations. The
repository `.gitignore` excludes them. Review the generated manifests before
training because source-dataset releases and local extraction layouts can
differ.
