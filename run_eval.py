import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from KIEval.kieval import kieval
from eval_utils import load_ontology, parse_arguments, set_up_savefolder


def fill_entity_for_cord(values, category, ontology_keys, empty_token, value_delimiter, is_pred=False):
    entity = {ont: empty_token for ont in ontology_keys}
    for ent, value in values.items():
        if value is None:
            continue
        if isinstance(value, dict):
            # sub entity
            for k, v in value.items():
                if is_pred:
                    ont = k
                else:
                    ont = f"{category}.{ent}_{k}"
                # multiple sub entity
                if isinstance(v, list):
                    for v_ in v:
                        if ont in entity:

                            if entity[ont] == empty_token:
                                entity[ont] = v_
                            else:
                                entity[ont] += value_delimiter + v_
                        else:
                            raise ValueError(f"ont {ont} is not in entity")

                else:
                    assert isinstance(v, str), f"{v} is not str"
                    # drop if ont is not in entity
                    if ont in entity:
                        entity[ont] = v

        elif isinstance(value, list):
            # multiple entity
            for val in value:
                if isinstance(val, str):
                    if is_pred:
                        ont = ent
                    else:
                        ont = f"{category}.{ent}"
                    # drop if ont is not in entity
                    if ont in entity:
                        if entity[ont] == empty_token:
                            entity[ont] = val
                        else:
                            entity[ont] += value_delimiter + val
                elif isinstance(val, dict):
                    # multiple sub entity
                    for k, v in val.items():
                        assert isinstance(v, str), f"{v} is not str"
                        if is_pred:
                            ont = k
                        else:
                            ont = f"{category}.{ent}_{k}"
                        # drop if ont is not in entity
                        if ont in entity:
                            if entity[ont] == empty_token:
                                entity[ont] = v
                            else:
                                entity[ont] += value_delimiter + v
                else:
                    raise ValueError("wrong type", type(val))
        elif isinstance(value, str):
            # entity
            if is_pred:
                ont = ent
            else:
                ont = f"{category}.{ent}"
            # drop if ont is not in entity
            if ont in entity:
                entity[ont] = value
        else:
            raise ValueError("wrong type", type(value))
    return entity

def write_csv_for_cord(output_dir, index, sample, ontology, empty_token, is_pred=False, delimiter="\t", value_delimiter="||"):
    csv_list = []
    for category, values in sample.items():
        try:
            group_index, ontology_keys = ontology[category]
        except Exception:
            continue

        # write group index
        csv_list.append(str(group_index))
        # write ontology keys
        csv_list.append(ontology_keys)

        if isinstance(values, list):
            # multiple group entities
            for val in values:
                entity = fill_entity_for_cord(val, category, ontology_keys, empty_token, value_delimiter, is_pred=is_pred)
                csv_list.append(entity.values())
        elif isinstance(values, dict):
            # single group entity
            entity = fill_entity_for_cord(values, category, ontology_keys, empty_token, value_delimiter, is_pred=is_pred)
            csv_list.append(entity.values())
        else:
            # drop
            print("wrong type", type(values))

        csv_list.append([])

    output_path = os.path.join(output_dir, f"{index}.csv")
    # Open a CSV file to write into
    with open(output_path, 'w', newline='\n') as file:
        writer = csv.writer(file, delimiter=delimiter)  # specify delimiter here
        for row in csv_list:
            # Write the fields into the CSV file
            writer.writerow(row)

def infer(model_output_dir, save_dir, dataset, ontology_keys, ontology_info, empty_token="<empty>"):
    # set up save folder
    gt_output_dir, pred_output_dir = set_up_savefolder(save_dir)

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        if isinstance(sample['ground_truth'], dict):
            ground_truth = sample["ground_truth"]
        else:
            ground_truth = json.loads(sample["ground_truth"])
        ground_truth = ground_truth["gt_parse"]
        '''
        {
            "menu": {
                "nm": "-TICKET CP",
                "num": "901016",
                "cnt": "2",
                "price": "60.000",
                "itemsubtotal": "60.000"
            },
            "sub_total": {
                "subtotal_price": "60.000",
                "discount_price": "-60.000",
                "tax_price": "5.455"
            },
            "total": {
                "total_price": "60.000",
                "creditcardprice": "60.000",
                "menuqty_cnt": "2.00"
            }
        }
        '''

        if "cord" in args.data_path_or_name.lower():
            write_csv_for_cord(gt_output_dir, idx, ground_truth, ontology_info, empty_token, is_pred=False)
        else:
            raise ValueError("Unsupported dataset")

        # Load model output result
        str_idx = str(idx).zfill(3)
        model_output_file = os.path.join(model_output_dir, f"{str_idx}.json")
        with open(model_output_file, "r") as file:
            model_output_raw = json.load(file)
        # Sanity refining and grounding
        if "cord" in args.data_path_or_name.lower():
            model_output = {}
            for cat_key in model_output_raw.keys():
                if cat_key == 'subtotal':
                    model_output['sub_total'] = model_output_raw[cat_key]
                else:
                    model_output[cat_key] = model_output_raw[cat_key]
        else:
            raise ValueError("Unsupported dataset")

        if "cord" in args.data_path_or_name.lower():
            write_csv_for_cord(pred_output_dir, idx, model_output, ontology_info, empty_token, is_pred=True)
        else:
            raise ValueError("Unsupported dataset")

    # get csv file path
    gold_csv_files = sorted(glob.glob(os.path.join(gt_output_dir, "*.csv")))
    pred_csv_files = sorted(glob.glob(os.path.join(pred_output_dir, "*.csv")))

    # calculate KIEval
    result, _, _ = kieval(
        gold_csv_files,
        pred_csv_files,
        ontology_keys,
        empty_token="<empty>",
        grouping_strategy="hungarian",
        value_delimiter="||",
        entity_type_delimiter="\t",
        split_merged_value=True,
        include_empty_token_in_score_calculation=False,
    )
    output_file = os.path.join(save_dir, "result.md")
    with open(output_file, "w") as file:
        file.write(result)

if __name__ == "__main__":
    # Parse command-line arguments describing where inputs live and
    # where evaluation artifacts should be written.
    args = parse_arguments()

    # load ontology from train dataset
    dataset = load_dataset(args.data_path_or_name)

    # next, load ontology from dataset
    ontology, ontology_info = load_ontology(args.data_path_or_name)

    # Evaluating model output
    infer(args.model_output_dir, args.save_dir, dataset['test'], ontology, ontology_info)
