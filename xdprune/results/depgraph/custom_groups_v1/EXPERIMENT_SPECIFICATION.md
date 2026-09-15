# Custom Group Physical-Validation Specification V1

Status: frozen before canonical custom-family physical results were generated;
completed with 18/18 GEN/SNOW runs passing.

## Inputs

- GEN checkpoint: `results/baselines/b_gen_mio_yolo26n_s42_v1/weights/best.pt`
- SNOW checkpoint: `results/baselines/b_snow_acdc_yolo26n_s42_v1/weights/best.pt`
- requested logical-width fraction: `1/8 = 0.125`
- physical-validation input sizes: `64` and `640`

## Non-attention C3k2 logical group

For hidden width `c` and selected logical indices `J`, V1 removes:

- `J` and `c + J` from the outer `cv1` output and BatchNorm;
- the mapped copy of `J` from each of the `2 + n` outer concatenation segments;
- `J` from the input and output sides required by a residual Bottleneck; and
- the corresponding input branches and output of a nested C3k block.

The outer C3k2 input and output channel counts remain unchanged. Internal
bottleneck expansion widths that are not dependency-coupled to the logical
outer width are not additionally pruned in V1.

## Importance

One L1 vector is calculated for every physical tensor slice coupled to the
logical width. Each vector is divided by its own mean, and the normalized
vectors are averaged to obtain one score per logical channel. The lowest
`c / 8` logical scores are selected independently for each frozen checkpoint.

## Head-aware C2PSA and attention-C3k2 groups

Stock attention uses two heads, a 64-channel value dimension per head and a
32-channel query/key dimension per head. One logical attention unit couples two
value/embedding channels with their corresponding query and key positions. V1
selects four units independently within each head, removing eight units and 16
embedding channels overall. After pruning, both heads remain present with a
56-channel value dimension and a 28-channel query/key dimension.

The rule updates the outer paired split and concatenation, qkv input and packed
output, attention projection, depthwise positional convolution, feed-forward
residual output, `head_dim`, `key_dim` and attention scale. The attention-C3k2
rule also updates the preceding residual Bottleneck and every reused outer
concatenation segment.

## Exclusions

- no detection-head pruning;
- no accuracy evaluation, BatchNorm update or fine-tuning;
- no cumulative pruning; and
- no edits to the sealed generic catalogue or T1/T2 V2 results.
