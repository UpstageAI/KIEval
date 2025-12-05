## KIEval: Evaluation Metric for Document Key Information Extraction

KIEval is an application-centric evaluation metric for **Document Key Information Extraction (KIE)**. 

This repository contains the **official implementation** used in the paper  
**["KIEval: Evaluation Metric for Document Key Information Extraction"](https://arxiv.org/abs/2503.05488)**, ICDAR 2025.

---

## Key characteristics

- **Multi-level evaluation**:  
  - **Non-group level** – entity-wise performance for standalone keys (e.g., `store_name`, `date`).  
  - **Group level** – performance over **sets of related entities** (e.g., per-line item on a receipt).  
- **Structure-aware matching**: Uses strategies such as the Hungarian algorithm to align predicted and gold groups, handling **variable numbers of entities and groups**.
- **CORD pipeline provided**: This repository ships with an end-to-end evaluation script for the **CORD** dataset via HuggingFace `datasets`.

---

## Evaluation flow
- Model Output2CSV conversion stage
  - Given the model output file and the target dataset's ontology information, convert the model output into CSV files
- Evaluation stage
  - KIEval then assesses the model's performance based on the saved pred and gt CSV files

---
## Repository structure

- **`run_eval.py`**:  
  High-level script to evaluate a model on the CORD dataset using KIEval.  
  - Loads the dataset via `datasets.load_dataset`.  
  - Automatically builds an ontology from CORD ground-truth.  
  - Converts ground truth and predictions into CSV format.  
  - Runs the KIEval metric and writes a markdown summary.

- **`eval_utils.py`**:  
  - `load_ontology(...)`: Derives ontology keys and grouping information from the dataset (currently supports **CORD**).  
  - `parse_arguments(...)`: CLI options for `run_eval.py`.  
  - `set_up_savefolder(...)`: Creates the folder structure for evaluation artifacts.

- **`KIEval/kieval.py`**:  
  Core implementation of the **KIEval metric**:
  - Confusion-matrix construction at non-group and group levels.  
  - Public function `kieval(...)` that operates on CSV files.

- **`KIEval/utils.py`**:  
  Utility functions for:
  - Parsing CSVs into grouped and non-grouped entities.  
  - Computing TP / FP / FN.  
  - Performing Hungarian matching between groups.  
  - Aggregating metrics and rendering markdown tables.  

---

## Installation

The recommended way to reproduce the environment is:
```bash
git clone <THIS_REPO_URL>
cd KIEval

# This repository was tested with python=3.8
pip install -r requirements.txt
```

---

## Quick start: evaluating a model on CORD

The provided script `run_eval.py` implements a full evaluation pipeline for the **CORD** dataset, using data loaded via HuggingFace `datasets`.

### 1. Prepare CORD ground truth

The script expects a dataset accessible through `datasets.load_dataset(...)`.  
For the public CORD dataset on HuggingFace, use:

```bash
python -c "from datasets import load_dataset; load_dataset('naver-clova-ix/cord-v2')"
```

`run_eval.py` will later call:

```python
from datasets import load_dataset
dataset = load_dataset(args.data_path_or_name)
```

and use the split `dataset['test']`, where each sample has a JSON-formatted `ground_truth` field with a `gt_parse` structure.

### 2. Prepare model predictions

You are responsible for running your own KIE model on the CORD **test** split and writing its predictions to disk.

For reference, we provide a sample output from **QwenVL_72B** on CORD in `KIEval/QwenVL_72B_Sample/QwenVL_72B_output.tar.gz`.

You can extract and examine this to understand the expected format.

### 3. Run the evaluator

From the repository root:

```bash
python run_eval.py \
  --model_output_dir /path/to/model_outputs \
  --data_path_or_name naver-clova-ix/cord-v2 \
  --save_dir /path/to/eval_results
```

This will:

- Build an ontology from the dataset (all splits) and derive grouped ontology information.  
- Convert each document’s ground truth and prediction into a CSV representation suitable for the metric.  
- Run KIEval and write a summary markdown file:
  - **`/path/to/eval_results/result.md`** – aggregate KIEval scores.
- Also create intermediate folders:
  - **`/path/to/eval_results/gt/`** – CSVs of gold annotations.  
  - **`/path/to/eval_results/pred/`** – CSVs of predictions.

You can open `result.md` to check KIEval Entity F1 and KIEval Aligned scores (as discussed in the paper).

---

## Python API

In addition to the CORD pipeline, you can call the KIEval metric directly from Python if you already have:
- gold CSVs
- pred CSVs
- ontology file -- list of ontology keys to include in KIEval evaluation

### Core metric function

The main entry point is `KIEval.kieval.kieval`:

```python
from glob import glob

from KIEval.kieval import kieval

# Paths to CSVs (one gold and one prediction file per document)
gold_csv_files = sorted(glob("/path/to/gt/*.csv"))
pred_csv_files = sorted(glob("/path/to/pred/*.csv"))
with open("/path/to/ontology_file.json") as f:
    ontology_keys = json.load(f)

result, _, _ = kieval(
    gold_csv_files=gold_csv_files,
    pred_csv_files=pred_csv_files,
    shortlist=list(ontology_keys),      # ontology keys to include in scoring
    empty_token="<empty>",
    grouping_strategy="hungarian",       # or "max_em_score"
    value_delimiter="||",               # delimiter for multi-valued fields
    entity_type_delimiter="\t",         # delimiter between entity key and value in CSV
    split_merged_value=True,
    include_empty_token_in_score_calculation=False,
    print_score=True,
)

print(result)
```

### Expected CSV format (high level)

The CSV format is shared by gold and prediction files:

- Files are logically split into **blocks** separated by blank lines.  
- **Group blocks**:
  - First line: integer group index (e.g., `0`, `1`, ...).  
  - Second line: **ontology keys** separated by `entity_type_delimiter` (default: tab).  
  - Following lines: values for each entity key, again separated by `entity_type_delimiter`.  
- **Non-group blocks**:
  - Lines that do not start with a digit belong to non-group entities.  
  - Each line is `key<delimiter>value`.  
- Only keys listed in `shortlist` are scored; others are filtered out.

For CORD, these CSVs are generated automatically by `run_eval.py` and `fill_entity_for_cord(...)`, so you do not normally need to construct them by hand unless you are integrating a new dataset.

---

## Extending KIEval to new datasets

To apply KIEval to another KIE dataset, you typically need to:

- **Define an ontology** of fields and groups for the dataset.  
- **Write a converter** that transforms both ground-truth annotations and model predictions into the CSV format used by `KIEval.utils.read_non_group_and_group_entities`.  
- **Call the Python API** (`kieval(...)`) with:
  - Your list of ontology keys (`shortlist`).  
  - Appropriate delimiters and `empty_token`.  
  - A grouping strategy (`"hungarian"` is generally recommended).

---

## Citation

If you use KIEval or this codebase in your research or product, please cite:

```bibtex
@inproceedings{khang2025kieval,
  title={KIEval: Evaluation metric for document key information extraction},
  author={Khang, Minsoo and Jung, Sang Chul and Park, Sungrae and Hong, Teakgyu},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={270--286},
  year={2025},
  organization={Springer}
}
```

---

## License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this code for both academic and commercial purposes.


---

### Contributors
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/mckhang">
        <img src="https://avatars.githubusercontent.com/u/37468544?v=4" width="80px;" alt="mckhang"/>
        <br />
        <sub><b>Khang, Minsoo</b></sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/tkdcjf159">
        <img src="https://avatars.githubusercontent.com/u/29270584?v=4" width="80px;" alt="Sang Chul Jung"/>
        <br />
        <sub><b>Jung, Sang Chul</b></sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/sungrae-park">
        <img src="https://avatars.githubusercontent.com/u/79886061?v=4" width="80px;" alt="Sungrae Park"/>
        <br />
        <sub><b>Park, Sungrae</b></sub>
      </a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/tghong">
        <img src="https://avatars.githubusercontent.com/u/17827160?v=4" width="80px;" alt="tghong"/>
        <br />
        <sub><b>Hong, Teakgyu</b></sub>
      </a>
      <br />
    </td>
  </tr>
</table>


