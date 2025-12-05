import argparse
import os
import json
from datasets import load_dataset

def load_ontology(data_path_or_name):
    """
    Build ontology information for a supported dataset.

    This helper currently supports:
      * CORD:  a hierarchical ontology extracted from the nested ``gt_parse`` structure.

    Parameters
    ----------
    data_path_or_name: str
        The HuggingFace ``datasets`` identifier or local path for the dataset.

    Returns
    -------
    ontology: list[str]
        List of ontology keys.
    ontology_info: dict or None
        For hierarchical datasets (e.g., CORD), a mapping from top-level group
        name to a tuple of ``(group_index, [ontology_keys])``.
    """
    dataset_name = data_path_or_name.lower()

    # -------------------------------------------------------------------------
    # Hierarchical ontology datasets (e.g., CORD)
    # -------------------------------------------------------------------------
    if "cord" in dataset_name:

        def extract_cord_ontology_keys(obj, prefix="", depth=0):
            """
            Recursively traverse a CORD ``gt_parse`` object and collect ontology keys.

            Each key encodes the path within the nested structure. For example:
                category.field
                category.field_subfield
            """
            keys = set()

            if not isinstance(obj, dict):
                return keys

            for field_name, field_value in obj.items():
                if depth == 0:
                    current_key = f"{prefix}.{field_name}"
                else:
                    current_key = f"{prefix}_{field_name}"

                if isinstance(field_value, dict):
                    # Recurse into nested dictionaries.
                    keys |= extract_cord_ontology_keys(field_value, current_key, depth + 1)
                elif isinstance(field_value, str):
                    # Leaf string fields become ontology entries.
                    keys.add(current_key)

            return keys

        # Load the full dataset so that the ontology is derived from *all* splits.
        dataset = load_dataset(data_path_or_name)
        ontology_keys = set()

        for split_name in dataset.keys():
            for sample in dataset[split_name]:
                ground_truth = json.loads(sample["ground_truth"])["gt_parse"]

                for category_name, category_values in ground_truth.items():
                    if isinstance(category_values, list):
                        # Category has multiple entities.
                        for entity in category_values:
                            ontology_keys |= extract_cord_ontology_keys(entity, category_name)
                    elif isinstance(category_values, dict):
                        # Category has a single entity.
                        ontology_keys |= extract_cord_ontology_keys(category_values, category_name)
                    else:
                        raise ValueError(
                            f"Unexpected type for values in category '{category_name}': "
                            f"{type(category_values)}"
                        )

        # Sort ontology keys for determinism and readability.
        ontology = sorted(ontology_keys)

        # Group ontology keys by their top-level category (text before the first '.').
        ontology_info = {}
        group_index = 0

        for ontology_key in ontology:
            group_name = ontology_key.split(".")[0]

            if group_name not in ontology_info:
                # First time we see this group: initialize its entry.
                ontology_info[group_name] = (group_index, [ontology_key])
                group_index += 1
            else:
                # Subsequent keys for this group are appended to the existing list.
                ontology_info[group_name][1].append(ontology_key)

        return ontology, ontology_info

    # -------------------------------------------------------------------------
    # Unsupported datasets
    # -------------------------------------------------------------------------
    raise ValueError(f"Unsupported dataset for ontology extraction: {data_path_or_name}")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the following attributes:

        * model_output_dir: directory containing model prediction files
        * data_path_or_name: dataset identifier or local path
        * save_dir: directory where evaluation outputs will be written
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Key Information Extraction (KIE) model outputs against "
            "ground-truth annotations using the UpScore metric."
        )
    )

    parser.add_argument(
        "--model_output_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing model prediction files in JSON "
            "format (one file per document)."
        ),
    )
    parser.add_argument(
        "--data_path_or_name",
        type=str,
        required=True,
        help=(
            "HuggingFace datasets identifier or local path for the evaluation "
            "dataset (for example: 'naver-clova-ix/cord-v2')."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help=(
            "Directory where evaluation artifacts will be saved, including "
            "intermediate CSV files and markdown summaries."
        ),
    )

    return parser.parse_args()

def set_up_savefolder(save_dir):
    # create save_dir if not exists
    os.makedirs(save_dir, exist_ok=True)

    # create gt_output_dir
    gt_output_dir = os.path.join(save_dir, "gt")
    os.makedirs(gt_output_dir, exist_ok=True)

    # create pred_output_dir
    pred_output_dir = os.path.join(save_dir, "pred")
    os.makedirs(pred_output_dir, exist_ok=True)

    return gt_output_dir, pred_output_dir