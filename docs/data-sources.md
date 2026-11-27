# Data sources

Every model in Admetus is trained on a publicly published dataset. Nothing here is
redistributed: `admetus data fetch` downloads each file at use time from Harvard
Dataverse, where Therapeutics Data Commons publishes them.

## Licensing

The datasets are licensed by their publishers, not by this project, and Admetus'
Apache-2.0 licence does not extend to them. Therapeutics Data Commons records
them as CC BY 4.0, with most pages phrasing it as "Not Specified. CC BY 4.0". If
you redistribute a dataset, or publish a model trained on one, check the licence
and the primary reference on that dataset's own page rather than relying on this
summary.

Per-dataset primary references are listed on each dataset's page at
<https://tdcommons.ai>. They are not reproduced here, because citing a paper
accurately matters more than citing it conveniently, and the upstream pages are
the authoritative list.

## Citing Therapeutics Data Commons

- Huang et al., *Therapeutics Data Commons: Machine Learning Datasets and Tasks
  for Drug Discovery and Development*, NeurIPS 2021.
  <https://arxiv.org/abs/2102.09548>
- Huang et al., *Artificial intelligence foundation for therapeutic science*,
  Nature Chemical Biology, 2022.
  <https://doi.org/10.1038/s41589-022-01131-2>

## Integrity

Each dataset's SHA-256 is pinned in `src/admetus/data/endpoints.yaml`, and a file
that does not match is refused rather than adopted. Being plain about what that
is worth: the publisher does not provide checksums, so the pinned values are the
ones this project observed first. It remains trust on first use. What it buys is
that the trust decision is frozen at a known point and tracked in version
control, instead of drifting silently on every fetch, so a model's recorded
provenance stays true. Taking a revised dataset is deliberate:
`admetus data fetch --accept-upstream-change`.

## Datasets

The table lists the Harvard Dataverse file id each dataset is fetched by, and the
row count after normalization. Normalized counts differ from the raw files
because Admetus filters to one species where a dataset covers several, selects a
single assay column from a wide file, and collapses duplicate structures.

| Endpoint | Dataset | Dataverse id | Task | Rows | Official group |
|---|---|---|---|---|---|
| `caco2` | Caco2_Wang | 4259569 | regression | 906 | yes |
| `pampa` | PAMPA_NCATS | 6695858 | classification | 2,034 | - |
| `hia` | HIA_Hou | 4259591 | classification | 578 | yes |
| `pgp` | Pgp_Broccatelli | 4259597 | classification | 1,212 | yes |
| `bioavailability` | Bioavailability_Ma | 4259567 | classification | 640 | yes |
| `lipophilicity` | Lipophilicity_AstraZeneca | 4259595 | regression | 4,200 | yes |
| `solubility` | Solubility_AqSolDB | 4259610 | regression | 9,982 | yes |
| `bbb` | BBB_Martins | 4259566 | classification | 1,965 | yes |
| `ppbr` | PPBR_AZ | 6413140 | regression | 1,614 | yes |
| `vdss` | VDss_Lombardo | 4267387 | regression | 1,111 | yes |
| `cyp2c9_veith` | CYP2C9_Veith | 4259577 | classification | 12,092 | yes |
| `cyp2d6_veith` | CYP2D6_Veith | 4259580 | classification | 13,130 | yes |
| `cyp3a4_veith` | CYP3A4_Veith | 4259582 | classification | 12,328 | yes |
| `cyp1a2_veith` | CYP1A2_Veith | 4259573 | classification | 12,579 | - |
| `cyp2c19_veith` | CYP2C19_Veith | 4259576 | classification | 12,665 | - |
| `cyp2c9_substrate` | CYP2C9_Substrate_CarbonMangels | 4259584 | classification | 666 | yes |
| `cyp2d6_substrate` | CYP2D6_Substrate_CarbonMangels | 4259578 | classification | 663 | yes |
| `cyp3a4_substrate` | CYP3A4_Substrate_CarbonMangels | 4259581 | classification | 667 | yes |
| `half_life` | Half_Life_Obach | 4266799 | regression | 665 | yes |
| `clearance_hepatocyte` | Clearance_Hepatocyte_AZ | 4266187 | regression | 1,020 | yes |
| `clearance_microsome` | Clearance_Microsome_AZ | 4266186 | regression | 1,102 | yes |
| `ld50` | LD50_Zhu | 4267146 | regression | 7,342 | yes |
| `herg` | hERG | 4259588 | classification | 645 | yes |
| `herg_karim` | hERG_Karim | 6822246 | classification | 13,445 | - |
| `ames` | AMES | 4259564 | classification | 7,255 | yes |
| `dili` | DILI | 4259585 | classification | 475 | yes |
| `carcinogens` | Carcinogens_Lagunin | 4259570 | classification | 276 | - |
| `skin_reaction` | Skin Reaction | 4259609 | classification | 404 | - |
| `tox21_nr_ar` | Tox21 (NR-AR) | 4259612 | classification | 7,265 | - |

Each is fetched from `https://dataverse.harvard.edu/api/access/datafile/<id>`.

## Normalization applied

| Endpoint | What differs from the raw file |
|---|---|
| `ppbr` | The published file covers several species; only *Homo sapiens* rows are used |
| `tox21_nr_ar` | The file is wide, one column per assay; the NR-AR column is selected and rows without that label dropped |
| `solubility` | Salts are kept rather than reduced to their largest fragment, because the measured solubility belongs to the salt |
| all | Duplicate structures are collapsed. Regression takes the group median; a classification group whose labels disagree is dropped, since a contradictory label teaches nothing |

The exact counts before and after are recorded in each model's `bundle.json`
under `provenance.dataset`, so what a model was trained on can be reconstructed.

## Notes on specific datasets

- **`herg` and `herg_karim`** are different curations of the same assay, with
  substantial compound overlap and different label definitions. Treating their
  two predictions as independent evidence would be wrong.
- **The five CYP inhibition panels** (`cyp1a2_veith`, `cyp2c9_veith`,
  `cyp2c19_veith`, `cyp2d6_veith`, `cyp3a4_veith`) derive from the same screening
  library, so their compound sets, errors and domain flags are correlated.
- **`tox21_nr_ar`** is the androgen receptor assay alone, not general toxicity.
  The other eleven assays in that file can be added as further registry entries
  without code changes.
- **`ld50`** contains inorganics, metals and mixtures that RDKit cannot
  featurize. Those rows are reported and excluded, never substituted.

