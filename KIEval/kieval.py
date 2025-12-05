import argparse
import copy
import glob
import os
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import yaml

from .utils import (
    defaultdict_to_dict,
    get_f1_recall_precision_accuracy_kieval,
    get_TP_FP_FN,
    print_kieval,
    read_non_group_and_group_entities,
    sort_indices_based_on_scores,
)


def get_non_group_confusion_matrix(
    gold: Dict[str, Any],
    pred: Dict[str, Any],
    confusion_matrix: Dict[str, Any],
    empty_token: str,
    value_delimiter: str,
    split_merged_value: bool,
    include_empty_token_in_score_calculation: bool,
) -> Tuple[Dict[str, Any], bool]:
    """
    Calculates the confusion matrix at non_group for given 'gold' and 'predicted' dictionaries.
    The confusion matrix here includes True Positives (TP), False Positives (FP), and False Negatives (FN)
    for each key in the union of keys from both gold and predicted dictionaries.

    Args:
        gold (Dict[str, Any]): A dictionary with true values.
        pred (Dict[str, Any]): A dictionary with predicted values.
        confusion_matrix (Dict[str, Any]): A dictionary to store the confusion matrix data.
        empty_token (str): A string representing the empty token symbol.
        value_delimiter (str): A string representing the delimiter for values.
        split_merged_value (bool): A flag indicating if the merged values should be split.
        include_empty_token_in_score_calculation (bool): A flag indicating if the empty token should be included in the score calculation.

    Returns:
        The first element is the confusion matrix dictionary updated with the current key's TP, FP, and FN quantities.
        The second element is a boolean representing whether a perfect match has been achieved for all keys.

    """
    exact_match_only = []
    if not gold and not pred:
        return confusion_matrix, exact_match_only

    union_keys = set(gold.keys()).union(set(pred.keys()))

    for key in list(union_keys):
        TP, FP, FN = 0, 0, 0
        if key in gold:
            if split_merged_value:
                if key not in pred:
                    pred_values = []
                else:
                    pred_values = [
                        v
                        for merged_value in pred[key]
                        for v in merged_value.split(value_delimiter)
                        if v != empty_token or include_empty_token_in_score_calculation
                    ]
                gold_values = [
                    v
                    for merged_value in gold[key]
                    for v in merged_value.split(value_delimiter)
                    if v != empty_token or include_empty_token_in_score_calculation
                ]
            else:
                if key not in pred:
                    pred_values = []
                else:
                    pred_values = [
                        merged_value
                        for merged_value in pred[key]
                        if merged_value != empty_token or include_empty_token_in_score_calculation
                    ]
                gold_values = [
                    merged_value
                    for merged_value in gold[key]
                    if merged_value != empty_token or include_empty_token_in_score_calculation
                ]
            TP, FP, FN = get_TP_FP_FN(gold_values, pred_values)
            wrong_match = min(FP, FN)
            no_match = max(FP, FN) - wrong_match

            if FP < 0 or FN < 0:
                raise ValueError("False Positive (FP) and False Negative (FN) should not be less than zero.")
            confusion_matrix[key]["TP"] += TP
            confusion_matrix[key]["FP"] += FP
            confusion_matrix[key]["FN"] += FN
            confusion_matrix[key]["wrong_match"] += wrong_match
            confusion_matrix[key]["no_match"] += no_match
            confusion_matrix[key]["label"] += len(gold_values)
        else:
            # key not in gold. it is a false positive
            pred_values = [v for v in pred[key] if v != empty_token or include_empty_token_in_score_calculation]
            FP = len(pred_values)
            confusion_matrix[key]["FP"] += FP
            confusion_matrix[key]["no_match"] += FP

        if TP > 0 and FP + FN == 0:
            exact_match_only.append(True)
        else:
            exact_match_only.append(False)

    return confusion_matrix, exact_match_only


def get_group_confusion_matrix(
    gold: List[Dict[str, Any]],
    pred: List[Dict[str, Any]],
    confusion_matrix: Dict[str, Any],
    doc_name: str,
    empty_token: str,
    value_delimiter: str,
    grouping_strategy: str,
    split_merged_value: bool,
    include_empty_token_in_score_calculation: bool,
) -> Tuple[Dict[str, Any], bool]:
    """
    Calculates the confusion matrix at group level for given 'gold' and 'predicted' dictionaries.
    The confusion matrix here includes True Positives (TP), False Positives (FP), and False Negatives (FN)

    Args:
        gold: A list of dictionaries containing the true values.
        pred: A list of dictionaries containing the predicted values.
        confusion_matrix: A dictionary that stores the scores for True Positives(TP), False Positives(FP), and False Negatives(FN).
        doc_name: Name of the document. Used for keying the confusion matrix
        empty_token: A string that represents a missing or empty prediction.
        value_delimiter: A string that is used to separate the different values in a non_group prediction or gold annotation.
        grouping_strategy: A string used to define the strategy for grouping entities.
        split_merged_value: A boolean that if True, the function will split merged values.
        include_empty_token_in_score_calculation (bool): A flag indicating if the empty token should be included in the score calculation.

    Returns:
        confusion_matrix: A dictionary that stores the scores for True Positives(TP), False Positives(FP), and False Negatives(FN).
    """

    exact_match_only = []
    if not gold and not pred:
        return confusion_matrix, exact_match_only

    union_keys = set(gold.keys()).union(set(pred.keys()))
    for group_id in list(union_keys):
        TP, FP, FN, label = 0, 0, 0, 0

        if group_id not in gold:
            # count wrong prediction. group_id not in gold
            for p_idx, p_kv in enumerate(pred[group_id]):
                # grouping count wrong prediction, no label(=gold)
                confusion_matrix[f"{group_id}.grouping"]["FP"] += 1
                confusion_matrix[f"{group_id}.grouping"]["no_match"] += 1
                for p_k, p_v in p_kv.items():
                    if split_merged_value:
                        pred_v = [
                            pv
                            for pv in p_v.split(value_delimiter)
                            if pv != empty_token or include_empty_token_in_score_calculation
                        ]
                        FP += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += len(pred_v)
                    else:
                        if p_v != empty_token or include_empty_token_in_score_calculation:
                            FP += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += 1
        else:
            # build score matrix -> to track which group in gold best-matches with which group in pred
            exact_match_score_matrix = defaultdict(lambda: defaultdict(int))

            # greedy search for max exact match score
            for g_idx, g_kv in enumerate(gold[group_id]):
                for p_idx, p_kv in enumerate(pred[group_id]):
                    exact_match_score_matrix[g_idx][p_idx] += 0
                    for p_k, p_v in p_kv.items():
                        # exist pred_key in gold
                        if p_k in g_kv:
                            if split_merged_value:
                                # split merged value
                                gold_v = [
                                    gv
                                    for gv in g_kv[p_k].split(value_delimiter)
                                    if gv != empty_token or include_empty_token_in_score_calculation
                                ]
                                pred_v = [
                                    pv
                                    for pv in p_v.split(value_delimiter)
                                    if pv != empty_token or include_empty_token_in_score_calculation
                                ]
                                # count exact match
                                for v in pred_v:
                                    if v in gold_v:
                                        exact_match_score_matrix[g_idx][p_idx] += 1
                                        gold_v.remove(v)
                            else:
                                # count exact match
                                if p_v == g_kv[p_k] and (
                                    p_v != empty_token or include_empty_token_in_score_calculation
                                ):
                                    exact_match_score_matrix[g_idx][p_idx] += 1

            # sort score matrix by grouping_strategy algorithm
            sorted_key_value_pairs = sort_indices_based_on_scores(exact_match_score_matrix, strategy=grouping_strategy)

            remove_g_idx = []
            remove_p_idx = []
            # find max exact match score
            for (g_idx, p_idx), max_em_score in sorted_key_value_pairs:
                if g_idx not in remove_g_idx and p_idx not in remove_p_idx:
                    remove_g_idx.append(g_idx)
                    remove_p_idx.append(p_idx)

                    TP += max_em_score
                    gold_v = gold[group_id][g_idx]
                    pred_v = pred[group_id][p_idx]

                    gold_v = {
                        k: v for k, v in gold_v.items() if v != empty_token or include_empty_token_in_score_calculation
                    }
                    pred_v = {
                        k: v for k, v in pred_v.items() if v != empty_token or include_empty_token_in_score_calculation
                    }

                    ##################################################
                    # Grouping-level evaluation
                    ##################################################
                    if gold_v == pred_v:
                        confusion_matrix[f"{group_id}.grouping"]["TP"] += 1
                    else:
                        confusion_matrix[f"{group_id}.grouping"]["FP"] += 1
                        confusion_matrix[f"{group_id}.grouping"]["FN"] += 1
                        confusion_matrix[f"{group_id}.grouping"]["wrong_match"] += 1
                    # count label(=gold)
                    confusion_matrix[f"{group_id}.grouping"]["label"] += 1


                    ##################################################
                    # Entity-level evaluation for entities in a group
                    # Three cases need to be handled:
                    # 1. both the pred & gold contains the entity-key
                    # 2. only the pred contains the entity-key -- False Positive
                    # 3. only the gold contains the entity-key -- False Negative
                    ##################################################
                    for p_k, p_v in pred_v.items():
                        # case 1: both the pred & gold contains the entity-key
                        if p_k in gold_v:
                            if split_merged_value:
                                gold_list = gold_v[p_k].split(value_delimiter)
                                pred_list = p_v.split(value_delimiter)
                                group_TP, group_FP, group_FN = get_TP_FP_FN(gold_list, pred_list)
                                group_wrong_match = min(group_FP, group_FN)
                                group_no_match = max(group_FP, group_FN) - group_wrong_match
                                FP += group_FP
                                FN += group_FN
                                label += len(gold_list)
                                confusion_matrix[f"{p_k}.{group_id}.group"]["TP"] += group_TP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += group_FP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FN"] += group_FN
                                confusion_matrix[f"{p_k}.{group_id}.group"]["wrong_match"] += group_wrong_match
                                confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += group_no_match
                                confusion_matrix[f"{p_k}.{group_id}.group"]["label"] += len(gold_list)
                            else:
                                if p_v == gold_v[p_k]:
                                    confusion_matrix[f"{p_k}.{group_id}.group"]["TP"] += 1
                                else:
                                    FP += 1
                                    FN += 1
                                    confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                                    confusion_matrix[f"{p_k}.{group_id}.group"]["FN"] += 1
                                    confusion_matrix[f"{p_k}.{group_id}.group"]["wrong_match"] += 1
                                label += 1
                                confusion_matrix[f"{p_k}.{group_id}.group"]["label"] += 1
                        else:
                            # case 2: only the pred contains the entity-key -- False Positive
                            if split_merged_value:
                                group_FP = len(p_v.split(value_delimiter))
                                FP += group_FP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += group_FP
                                confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += group_FP
                            else:
                                FP += 1
                                confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                                confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += 1

                    # case 3: only the gold contains the entity-key -- False Negative
                    missing_keys = set(gold_v.keys()) - set(pred_v.keys())
                    for m_k in missing_keys:
                        if split_merged_value:
                            group_FN = len(gold_v[m_k].split(value_delimiter))
                            FN += group_FN
                            label += group_FN
                            confusion_matrix[f"{m_k}.{group_id}.group"]["FN"] += group_FN
                            confusion_matrix[f"{m_k}.{group_id}.group"]["no_match"] += group_FN
                            confusion_matrix[f"{m_k}.{group_id}.group"]["label"] += group_FN
                        else:
                            FN += 1
                            label += 1
                            confusion_matrix[f"{m_k}.{group_id}.group"]["FN"] += 1
                            confusion_matrix[f"{m_k}.{group_id}.group"]["no_match"] += 1
                            confusion_matrix[f"{m_k}.{group_id}.group"]["label"] += 1

            # count wrong prediction
            not_removed_p_idx = []
            not_removed_g_idx = []
            for p_idx, p_kv in enumerate(pred[group_id]):
                if p_idx in remove_p_idx: # Skip processed group entries
                    continue
                not_removed_p_idx.append(p_idx)
                # grouping count wrong prediction
                confusion_matrix[f"{group_id}.grouping"]["FP"] += 1
                confusion_matrix[f"{group_id}.grouping"]["no_match"] += 1
                for p_k, p_v in p_kv.items():
                    if split_merged_value:
                        pred_v = [
                            pv
                            for pv in p_v.split(value_delimiter)
                            if pv != empty_token or include_empty_token_in_score_calculation
                        ]
                        FP += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += len(pred_v)
                        confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += len(pred_v)
                    else:
                        if p_v != empty_token or include_empty_token_in_score_calculation:
                            FP += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["FP"] += 1
                            confusion_matrix[f"{p_k}.{group_id}.group"]["no_match"] += 1

            # count miss prediction
            for g_idx, g_kv in enumerate(gold[group_id]):
                if g_idx in remove_g_idx:
                    continue
                not_removed_g_idx.append(g_idx)
                # grouping count miss prediction and label(=gold)
                confusion_matrix[f"{group_id}.grouping"]["FN"] += 1
                confusion_matrix[f"{group_id}.grouping"]["no_match"] += 1
                confusion_matrix[f"{group_id}.grouping"]["label"] += 1
                for g_k, g_v in g_kv.items():
                    if split_merged_value:
                        gold_v = [
                            gv
                            for gv in g_v.split(value_delimiter)
                            if gv != empty_token or include_empty_token_in_score_calculation
                        ]
                        FN += len(gold_v)
                        label += len(gold_v)
                        confusion_matrix[f"{g_k}.{group_id}.group"]["FN"] += len(gold_v)
                        confusion_matrix[f"{g_k}.{group_id}.group"]["no_match"] += len(gold_v)
                        confusion_matrix[f"{g_k}.{group_id}.group"]["label"] += len(gold_v)
                    else:
                        if g_v != empty_token or include_empty_token_in_score_calculation:
                            FN += 1
                            label += 1
                            confusion_matrix[f"{g_k}.{group_id}.group"]["FN"] += 1
                            confusion_matrix[f"{g_k}.{group_id}.group"]["no_match"] += 1
                            confusion_matrix[f"{g_k}.{group_id}.group"]["label"] += 1

        if FP < 0 or FN < 0:
            raise ValueError("False Positive (FP) and False Negative (FN) should not be less than zero.")

        if TP > 0 and FN + FP == 0:
            exact_match_only.append(True)
        else:
            exact_match_only.append(False)

    return confusion_matrix, exact_match_only


def get_kieval(confusion_matrix: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Computes a suite of evaluation metrics (precision, recall, micro_f1, macro_f1, accuracy)
    for the given confusion matrix, as well as the overall total.

    Args:
        confusion_matrix (Dict[str, Any]): A dictionary representing the confusion matrix,
        where each key corresponds to a specific classification category and the value is
        another dictionary with counts of 'True Positives', 'False Positives', 'False Negatives',
        and total labels.

    Returns:
        entity_kieval (Dict[str, Dict[str, int]]): The first dictionary of this tuple where each key represents
        a specific classification category and corresponding value is a dictionary with calculated scores of precision,
        recall, micro_f1, macro_f1, and accuracy.
        total_kieval (Dict[str, Dict[str, int]]): The second dictionary of this tuple where the key represents the type
        of total score (total.non_group or total.group) and the corresponding value is a dictionary of calculated scores
        (precision, recall, micro_f1, macro_f1, accuracy).
    """
    entity_kieval = defaultdict(lambda: defaultdict(int))
    total_kieval = defaultdict(lambda: defaultdict(int))

    # Initialize metric accumulators for different entity types
    metric_keys = ["TP", "FN", "FP", "wrong_match", "no_match", "label"]

    total_kieval_metrics = {key: 0 for key in metric_keys}
    non_group_metrics = {key: 0 for key in metric_keys}
    group_metrics = {key: 0 for key in metric_keys}
    grouping_metrics = {key: 0 for key in metric_keys}

    # Calculate metric score by key
    for key, conf_mat in confusion_matrix.items():
        TP, FN, FP, wrong_match, no_match, label = (
            conf_mat["TP"],
            conf_mat["FN"],
            conf_mat["FP"],
            conf_mat["wrong_match"],
            conf_mat["no_match"],
            conf_mat["label"],
        )
        if key.endswith(".group"): # This is for group-level metric
            group_metrics["TP"] += TP
            group_metrics["FN"] += FN
            group_metrics["FP"] += FP
            group_metrics["wrong_match"] += wrong_match
            group_metrics["no_match"] += no_match
            group_metrics["label"] += label
        elif key.endswith(".grouping"): # This is for entity-level for entities in a group
            grouping_metrics["TP"] += TP
            grouping_metrics["FN"] += FN
            grouping_metrics["FP"] += FP
            grouping_metrics["wrong_match"] += wrong_match
            grouping_metrics["no_match"] += no_match
            grouping_metrics["label"] += label
        else: # This is for non-group entity-level metric
            non_group_metrics["TP"] += TP
            non_group_metrics["FN"] += FN
            non_group_metrics["FP"] += FP
            non_group_metrics["wrong_match"] += wrong_match
            non_group_metrics["no_match"] += no_match
            non_group_metrics["label"] += label

        entity_kieval = get_f1_recall_precision_accuracy_kieval(
            metrics=entity_kieval,
            key=key,
            label=label,
            TP=TP,
            FN=FN,
            FP=FP,
            wrong_match=wrong_match,
            no_match=no_match,
        )

    # Calculate the overall metric score
    group_f1_list = [entity_kieval[k]["kieval_entity_f1"] for k, v in entity_kieval.items() if "group_id" in v]
    non_group_f1_list = [entity_kieval[k]["kieval_entity_f1"] for k, v in entity_kieval.items() if not k.endswith(".group")]
    grouping_f1_list = [
        entity_kieval[k]["kieval_entity_f1"] for k, v in entity_kieval.items() if not k.endswith(".grouping")
    ]
    total_kieval_f1_list = []

    if non_group_metrics["label"] > 0:
        total_kieval = get_f1_recall_precision_accuracy_kieval(
            metrics=total_kieval,
            key="total.non_group",
            label=non_group_metrics["label"],
            TP=non_group_metrics["TP"],
            FN=non_group_metrics["FN"],
            FP=non_group_metrics["FP"],
            wrong_match=non_group_metrics["wrong_match"],
            no_match=non_group_metrics["no_match"],
            f1_list=non_group_f1_list,
        )
        total_kieval_metrics["label"] += non_group_metrics["label"]
        total_kieval_metrics["TP"] += non_group_metrics["TP"]
        total_kieval_metrics["FN"] += non_group_metrics["FN"]
        total_kieval_metrics["FP"] += non_group_metrics["FP"]
        total_kieval_metrics["wrong_match"] += non_group_metrics["wrong_match"]
        total_kieval_metrics["no_match"] += non_group_metrics["no_match"]
        total_kieval_f1_list += non_group_f1_list

    if group_metrics["label"] > 0:
        # group entity score
        total_kieval = get_f1_recall_precision_accuracy_kieval(
            metrics=total_kieval,
            key="total.group",
            label=group_metrics["label"],
            TP=group_metrics["TP"],
            FN=group_metrics["FN"],
            FP=group_metrics["FP"],
            wrong_match=group_metrics["wrong_match"],
            no_match=group_metrics["no_match"],
            f1_list=group_f1_list,
        )
        # total grouping score
        total_kieval = get_f1_recall_precision_accuracy_kieval(
            metrics=total_kieval,
            key="total.grouping",
            label=grouping_metrics["label"],
            TP=grouping_metrics["TP"],
            FN=grouping_metrics["FN"],
            FP=grouping_metrics["FP"],
            wrong_match=grouping_metrics["wrong_match"],
            no_match=grouping_metrics["no_match"],
            f1_list=grouping_f1_list,
        )
        total_kieval_metrics["label"] += group_metrics["label"]
        total_kieval_metrics["TP"] += group_metrics["TP"]
        total_kieval_metrics["FN"] += group_metrics["FN"]
        total_kieval_metrics["FP"] += group_metrics["FP"]
        total_kieval_metrics["wrong_match"] += group_metrics["wrong_match"]
        total_kieval_metrics["no_match"] += group_metrics["no_match"]
        total_kieval_f1_list += group_f1_list

    # total entity score (non_group + group)
    total_kieval = get_f1_recall_precision_accuracy_kieval(
        metrics=total_kieval,
        key="total.kieval",
        label=total_kieval_metrics["label"],
        TP=total_kieval_metrics["TP"],
        FN=total_kieval_metrics["FN"],
        FP=total_kieval_metrics["FP"],
        wrong_match=total_kieval_metrics["wrong_match"],
        no_match=total_kieval_metrics["no_match"],
        f1_list=total_kieval_f1_list,
    )

    # check key in total kieval
    for key in ["total.non_group", "total.group", "total.kieval", "total.grouping"]:
        if key not in total_kieval:
            total_kieval[key] = None

    return entity_kieval, total_kieval


def kieval(
    gold_csv_files: List[str],
    pred_csv_files: List[str],
    shortlist: List[str],
    empty_token: str,
    grouping_strategy: str,
    value_delimiter: str,
    entity_type_delimiter: str,
    split_merged_value: bool,
    include_empty_token_in_score_calculation: bool,
    print_score: bool = True,
) -> Tuple[str, Dict, Dict]:
    """
    Calculates the performance of model's predictions at three levels: non_group-level, group-level, and document-level.

    Args:
        gold_csv_files (List[str]): A list of file paths for the gold standard (actual) entities in CSV format.
        pred_csv_files (List[str]): A list of file paths for the predicted entities in CSV format.
        shortlist (List[str]): A list of used keys for calculating kieval.
        empty_token (str): A string representing the empty token symbol.
        grouping_strategy (str): A string representing the strategy for grouping entities.
        value_delimiter (str): A string representing the delimiter for values.
        entity_type_delimiter (str): A string representing the delimiter for entity types.
        split_merged_value (bool): A flag indicating if the merged values should be split.
        include_empty_token_in_score_calculation (bool): A flag indicating if the empty token should be included in the score calculation.
        print_score (bool): A flag indicating whether to print the scores. Default is True.

    Returns:
        markdown_text (str): A markdown formatted text representing tables with the calculated metrics and scores
        at non_group-level, group-level, and document-level.
        entity_kieval (Dict[str, Dict[str, int]]): The first dictionary of this tuple where each key represents
        a specific classification category and corresponding value is a dictionary with calculated scores of precision,
        recall, micro_f1, macro_f1, and accuracy.
        total_kieval (Dict[str, Dict[str, int]]): The second dictionary of this tuple where the key represents the type
        of total score (total.non_group or total.group) and the corresponding value is a dictionary of calculated scores
        (precision, recall, micro_f1, macro_f1, accuracy).
    """

    confusion_matrix = defaultdict(lambda: defaultdict(int))

    document_exact_match = defaultdict(int)
    for gold_csv, pred_csv in zip(gold_csv_files, pred_csv_files):
        # sanity check
        assert os.path.basename(gold_csv) == os.path.basename(pred_csv)

        filename = os.path.basename(gold_csv)
        doc_name, extention = os.path.splitext(filename)

        ###################################################
        # First load all non-group and group entities from 
        # Gold and Pred CSV files
        ###################################################

        # Load non_group_entities, group_entities
        gold_non_group_entities, gold_group_entities = read_non_group_and_group_entities(
            gold_csv, entity_type_delimiter, empty_token, include_empty_token_in_score_calculation, shortlist
        )
        pred_non_group_entities, pred_group_entities = read_non_group_and_group_entities(
            pred_csv, entity_type_delimiter, empty_token, include_empty_token_in_score_calculation, shortlist
        )


        ###################################################
        # Next, calculate the confusion matrix at group and
        # non-group level
        ###################################################

        # find group level entity confusion metric
        confusion_matrix, group_match = get_group_confusion_matrix(
            gold_group_entities,
            pred_group_entities,
            confusion_matrix,
            doc_name,
            empty_token,
            value_delimiter,
            grouping_strategy,
            split_merged_value,
            include_empty_token_in_score_calculation,
        )

        # find non_group level entity confusion metric
        confusion_matrix, non_group_match = get_non_group_confusion_matrix(
            gold_non_group_entities,
            pred_non_group_entities,
            confusion_matrix,
            empty_token,
            value_delimiter,
            split_merged_value,
            include_empty_token_in_score_calculation,
        )

        if all(group_match) and all(non_group_match):
            if len(group_match) + len(non_group_match) > 0:
                document_exact_match[doc_name] = 1
            else:
                print(f"empty document : {doc_name}")
        else:
            document_exact_match[doc_name] = 0

    entity_kieval, total_kieval = get_kieval(confusion_matrix)
    for k, v in document_exact_match.items():
        entity_kieval[f"{k}.document"]["exact_match"] = v

    total_kieval["total.document"]["exact_match"] = sum(document_exact_match.values())
    total_kieval["total.document"]["num_of_document"] = len(document_exact_match)
    total_kieval["total.document"]["accuracy"] = sum(document_exact_match.values()) / len(document_exact_match)

    markdown_text = print_kieval(entity_kieval, total_kieval, print_score=print_score)

    # Convert defaultdict to dict
    entity_kieval = defaultdict_to_dict(entity_kieval)
    total_kieval = defaultdict_to_dict(total_kieval)

    return markdown_text, entity_kieval, total_kieval
