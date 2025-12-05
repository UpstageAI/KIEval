import copy
import operator
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import yaml
from rich.console import Console
from rich.markdown import Markdown
from scipy.optimize import linear_sum_assignment


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def normalized_edit_distance(s1, s2):
    levenshtein_dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0  # To avoid division by zero
    else:
        return levenshtein_dist / max_len


def get_entities(group, non_group, split_merged_value, empty_token, value_delimiter, include_empty_token_in_score_calculation):
    return_entities = defaultdict(list)

    for group_id, entities in group.items():
        for entity in entities:
            for k, v in entity.items():
                if split_merged_value:
                    return_entities[k] += [
                        v_
                        for v_ in v.split(value_delimiter)
                        if v_ != empty_token or include_empty_token_in_score_calculation
                    ]
                else:
                    if v != empty_token or include_empty_token_in_score_calculation:
                        return_entities[k] += [v]

    for k, entity in non_group.items():
        for v in entity:
            if split_merged_value:
                return_entities[k] += [
                    v_
                    for v_ in v.split(value_delimiter)
                    if v_ != empty_token or include_empty_token_in_score_calculation
                ]
            else:
                if v != empty_token or include_empty_token_in_score_calculation:
                    return_entities[k] += [v]
    return return_entities

def read_non_group_and_group_entities(
    csv_path: str,
    entity_type_delimiter: str,
    empty_token: str,
    include_empty_token_in_score_calculation: bool,
    shortlist: List[str],
) -> Tuple[Dict[str, List], List]:
    """Dividing its content into two distinct sections: a group and a non-group, returning these as separate dictionaries.
    Args:
        csv_path (str): csv file path
        entity_type_delimiter (str): entity_type_delimiter
        empty_token (str): empty token
        include_empty_token_in_score_calculation (bool): include empty token in score calculation
        shortlist (List[str]): A list of used keys for calculating kieval.

    Returns:
        non_group_entities (Dict[str, List]): Dict mapping entity entity-key to list of entity values
        group_entities (List of Dict[str, Any]): group level entities
    """
    csv_text = open(csv_path, "r", encoding="utf-8").read()
    csv_text_splited = csv_text.split("\n\n")

    group_entities = defaultdict(list)
    groups = [
        (int(chunk[0]), "\n".join(chunk.split("\n")[1:])) for chunk in csv_text_splited if chunk and chunk[0].isdigit()
    ]
    non_group_entities = defaultdict(list)
    non_groups = ["\n".join(chunk.split("\n")[1:]) for chunk in csv_text_splited if chunk and not chunk[0].isdigit()]

    if groups:
        for group_idx, group in groups:
            lines = group.split("\n")
            group_keys = lines[0].split(entity_type_delimiter) # First line of the group in csv represent the entity-keys of the group

            group_value_list = [line.split(entity_type_delimiter) for line in lines[1:] if line != ""]
            unused_keys_index = sorted([i for i, k in enumerate(group_keys) if k not in shortlist], reverse=True)
            
            # remove key that is not in the shortlist
            for idx in unused_keys_index:
                del group_keys[idx]

            for group_values in group_value_list:
                # remove value that corresponds to the key that is not in the shortlist
                for idx in unused_keys_index:
                    del group_values[idx]
                group_item = dict(zip(group_keys, group_values))
                
                if group_item:
                    # only add group_item if it is not empty
                    group_entities[group_idx].append(group_item)

    else:
        group_entities = defaultdict(list)

    if non_groups:
        for non_group in non_groups:
            for line in non_group.split("\n"):
                if line == "":
                    continue

                # entity_type_delimiter must not be in the entity key name.
                key_value = line.split(entity_type_delimiter, 1)

                key, value = key_value
                if value == empty_token and not include_empty_token_in_score_calculation:
                    continue
                if key in shortlist:
                    non_group_entities[key].append(value)

    return non_group_entities, group_entities


def get_TP_FP_FN(gold: List, pred: List) -> Tuple[int, int, int]:
    """
    The get_TP_FP_FN function calculates the number of true positives (TP), false positives (FP), and false negatives (FN) in prediction results.
    Args:
        gold (List[int]): a list of integers representing the gold standard (true values)
        pred (List[int]): A list of integers representing the predicted values


    Returns:
        The function returns a tuple of integers (Tuple[int, int, int]),
        showing the numbers of True Positives, False Positives,
        and False Negatives respectively.
        True Positive (TP): The values that exist in the gold standard (true labels) and are also correctly predicted in the predictions (pred).
        False Positive (FP): The values that only appear in the predictions (pred) but are not present in the gold standard (true labels).
        False Negative (FN): The values that are present in the gold standard (true labels) but are missing from the predictions (pred), meaning they were not predicted.
    """

    TP, FP, FN = 0, 0, 0
    tmp_gold_values = copy.deepcopy(gold)
    for p in pred:
        if p in tmp_gold_values:
            TP += 1
            tmp_gold_values.remove(p)

    FP = max(len(pred) - TP, 0)
    FN = len(tmp_gold_values)

    del tmp_gold_values
    return TP, FP, FN


def sort_indices_based_on_scores(exact_match_score_matrix: Dict[str, Any], strategy: str = "hungarian") -> List:
    """
    The sort_indices_based_on_scores function computes a sorted list of tuples, each contains a pair of indices and their corresponding score.
    The sorting is executed based on a provided strategy, which can be either 'max_em_score' or 'hungarian'.

    the strategy is 'max_em_score', this function sort indices pairs according to their exact match scores in descending order.
    the strategy is 'hungarian', it applies Hungarian method (using linear_sum_assignment function) on a 2D array constructed from exact match scores.

    Args:
        exact_match_score_matrix (Dict[str, Any]): A dictionary containing exact match scores, where each key corresponds to a pair of indices (gold index, prediction index), and each value is the corresponding exact match score.
        strategy (str): A string representing the strategy to be used for sorting the index pairs. By default, it is set to "hungarian".
    """
    if strategy == "max_em_score":
        key_value_pairs = []
        for gold_idx, inner_dict in exact_match_score_matrix.items():
            for pred_idx, em_score in inner_dict.items():
                key_value_pairs.append(((gold_idx, pred_idx), em_score))
        if key_value_pairs:
            sorted_key_value_pairs = sorted(key_value_pairs, key=operator.itemgetter(1), reverse=True)
        else:
            sorted_key_value_pairs = []

    elif strategy == "hungarian":
        sorted_key_value_pairs = []
        em_score_2d_array = np.array(
            [list(inner_dict.values()) for gold_idx, inner_dict in exact_match_score_matrix.items()]
        )
        if em_score_2d_array.size > 0:
            gold_idxs, pred_idxs = linear_sum_assignment(-em_score_2d_array)
            for g_idx, p_idx in zip(gold_idxs.tolist(), pred_idxs.tolist()):
                sorted_key_value_pairs.append(((g_idx, p_idx), exact_match_score_matrix[g_idx][p_idx]))

    else:
        raise ValueError("Invalid strategy. The strategy should be either 'max_em_score' or 'hungarian'.")

    return sorted_key_value_pairs


def get_f1_recall_precision_accuracy_kieval(
    metrics: Dict[str, Any],
    key: str,
    label: int,
    TP: int = 0,
    FN: int = 0,
    FP: int = 0,
    TN: int = 0,
    wrong_match: int = 0,
    no_match: int = 0,
    f1_list: List[int] = None,
) -> Dict[str, Any]:
    """
    Computes a suite of evaluation metrics (precision, recall, micro-f1, macro-f1, accuracy)
    given the values of True Positives (TP), False Negatives (FN), False Positives (FP), True Negatives (TN)
    and optionally a list of f1 scores.

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = TP + TN / (FN + TP + FP + TN)
    kieval_aligned = TP + TN / (no_match + TP + wrong_match + TN)

    Args:
        metrics (Dict[str, Any]): A dictionary to store the metrics for a specific classification category.
        key (str): A string representing the specific classification category.
        label (int): The ground truth label for the key.
        TP (int): The count of True Positives.
        FN (int): The count of False Negatives.
        FP (int): The count of False Positives.
        TN (int): The count of True Negatives. Its default value is 0.
        wrong_match (int): The count of wrong match.
        no_match (int): The count of no match.
        f1_list (List[int]): An optional list of f1 scores used to calculate the macro f1 score. Its default value is None.

    Returns:
        A type of metric (precision, recall, kieval_entity_f1, macro_f1, etc.)
        and the corresponding value is the calculated score.
    """
    precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if TP + FP + FN + TN != 0 else 0.0
    kieval_aligned = (TP + TN) / (TP + no_match + wrong_match + TN) if TP + no_match + wrong_match + TN != 0 else 0.0
    kieval_entity_f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0
    if f1_list is not None:
        macro_f1 = sum(f1_list) / len(f1_list) if len(f1_list) != 0 else 0.0
    else:
        macro_f1 = kieval_entity_f1

    if key.endswith(".group"):
        # {ontology_key}.{group_id}.group
        split_key = key.split(".")
        if len(split_key) > 3:
            key = ".".join(split_key[:-2])
            metrics[key]["group_id"] = split_key[-2]
    metrics[key]["precision"] = precision
    metrics[key]["recall"] = recall
    metrics[key]["kieval_entity_f1"] = kieval_entity_f1
    metrics[key]["macro_f1"] = macro_f1
    metrics[key]["label"] = label
    metrics[key]["accuracy"] = accuracy
    metrics[key]["kieval_aligned"] = kieval_aligned
    metrics[key]["exact_match"] = TP
    metrics[key]["wrong_match"] = wrong_match
    metrics[key]["no_match"] = no_match
    return metrics


def get_markdown_text(title: str, metric_list: List[str], kieval_dict: Dict[str, Any], first_header: str) -> str:
    """
    Generates a Markdown formatted table text with the provided title, metrics, scores and first header.

    Args:
        title (str): A string representing the title of the markdown table.
        metric_list (List[str]): A list of strings where each string represents a type of metric to be included in the table.
        kieval_dict (Dict[str, Any]): A dictionary where each key represents a classification category and
        the corresponding value is another dictionary with the metrics and scores.
        first_header (str): A string representing the first header of the table.

    Returns:
        A markdown formatted text representing a table with the provided metrics and scores.
    """

    markdown_text = title
    markdown_header = [first_header] + metric_list
    markdown_text += "|" + "|".join(markdown_header) + "|\n"
    markdown_text += "|" + "|".join(["--" for _ in markdown_header]) + "|\n"

    for key, scores in kieval_dict.items():
        if scores is None:
            continue
        markdown_text += (
            f"|{key}|"
            + "|".join(
                [
                    f"{scores[metric]:.4f}" if isinstance(scores[metric], float) else f"{scores[metric]}"
                    for metric in metric_list
                ]
            )
            + "|\n"
        )

    return markdown_text + "\n"


def print_formatted_kieval_results(
    entity_kieval: Dict[str, Any], total_kieval: Dict[str, Any], category: str, print_score: bool = True
) -> str:
    """
    The print_formatted_kieval_results function prints a structured result (in markdown format)
    which includes various scores at both entity and total levels.

    Args:
        entity_kieval (Dict[str, Any]): A dictionary that represents the scores at each entity level.
        Each key represents a specific classification category and the corresponding value is another dictionary with the metrics and scores.
        total_kieval (Dict[str, Any]): A dictionary that represents the total scores.
        Each key represents a type of total score (eg. total.non_group or total.group) and
        the corresponding value is another dictionary of calculated scores (precision, recall, micro_f1, macro_f1, accuracy, etc.).
        category (str): A string representing the level at which to print scores (e.g., "document").
        print_score (bool): A boolean value indicating whether to print the scores. Default is True.

    Returns:
        A concatenation of markdown formatted text representing tables with the provided metrics and scores,
        as generated by the get_markdown_text function.
    """

    console = Console()

    if total_kieval:
        if category == "document":
            total_metric_list = ["accuracy", "exact_match", "num_of_document"]
        else:
            total_metric_list = [
                "kieval_entity_f1",
                "macro_f1",
                "recall",
                "precision",
                "accuracy",
                "kieval_aligned",
                "label",
                "wrong_match",
                "no_match",
                "exact_match",
            ]

        total_kieval = OrderedDict(sorted(total_kieval.items()))
        title = f"# `{category} total`\n"

        total_markdown_text = get_markdown_text(title, total_metric_list, total_kieval, first_header="")
        markdown = Markdown(total_markdown_text)
        if print_score:
            console.print(markdown)
    else:
        total_markdown_text = ""

    # detailed score
    if entity_kieval:
        if category == "document":
            entity_metric_list = ["exact_match"]
        elif category == "grouping":
            entity_metric_list = [
                "kieval_entity_f1",
                "recall",
                "precision",
                "accuracy",
                "kieval_aligned",
                "label",
                "wrong_match",
                "no_match",
                "exact_match",
            ]
        elif category == "group":
            entity_metric_list = [
                "group_id",
                "kieval_entity_f1",
                "recall",
                "precision",
                "accuracy",
                "kieval_aligned",
                "label",
                "wrong_match",
                "no_match",
                "exact_match",
            ]
        elif category == "non_group":
            entity_metric_list = [
                "kieval_entity_f1",
                "recall",
                "precision",
                "accuracy",
                "kieval_aligned",
                "label",
                "wrong_match",
                "no_match",
                "exact_match",
            ]
        else:
            raise ValueError("Invalid category. The category should be either 'document', 'group' or 'non_group'.")

        first_header_dict = {
            "non_group": "ontology key",
            "group": "ontology key",
            "grouping": "grouping_id",
            "document": "document name",
        }
        entity_kieval = OrderedDict(sorted(entity_kieval.items()))
        title = f"# `{category} detailed summary `\n"
        entity_markdown_text = get_markdown_text(
            title,
            entity_metric_list,
            entity_kieval,
            first_header=first_header_dict[category],
        )
        markdown = Markdown(entity_markdown_text)
        if print_score:
            console.print(markdown)
    else:
        entity_markdown_text = ""

    return total_markdown_text + entity_markdown_text


def print_kieval(
    entity_kieval_dict: Dict[str, Any], total_kieval_dict: Dict[str, Any], print_score: bool = True
) -> str:
    """
    The print_kieval function prepares and prints markdown-based score cards for different categories: 'no_group', 'group', 'total', 'grouping' and 'document'.
    The function separates both `entity_kieval_dict` and `total_kieval_dict` dictionaries into these entity level categories based on their keys.
    After separation, it proceeds to call the `print_formatted_kieval_results` function for each entity level category to produce markdown texts.
    All three resulting markdown texts are then concatenated and returned.

    Args:
        entity_kieval_dict: A dictionary containing the kieval of each entity.
            Each key is a string specifying the entity, and each value is a number representing the kieval of that entity.
        total_kieval_dict: A dictionary with the total kieval of each entity type.
            Each key is a string specifying the entity type (e.g. 'non_group', 'group', 'document'),
            and the value is the total kieval for that type of entity.
        print_score: A boolean value indicating whether to print the scores. Default is True.

    Returns:
        A formatted string with markdown-based score cards, including the kieval for each entity level.
        It specifically separates the markdown text into sections for 'non_group', 'group', and 'document' levels
        and prints them in that order.
    """

    document_entity_kieval = {k: v for k, v in entity_kieval_dict.items() if k.endswith(".document")}
    group_entity_kieval = {k: v for k, v in entity_kieval_dict.items() if "group_id" in v}
    grouping_kieval = {k: v for k, v in entity_kieval_dict.items() if k.endswith(".grouping")}
    non_group_entity_kieval = {
        k: v
        for k, v in entity_kieval_dict.items()
        if "group_id" not in v and not k.endswith(".document") and not k.endswith(".grouping")
    }

    document_total_kieval = {k: v for k, v in total_kieval_dict.items() if k.endswith(".document")}
    group_total_kieval = {k: v for k, v in total_kieval_dict.items() if k.endswith(".group")}
    grouping_total_kieval = {k: v for k, v in total_kieval_dict.items() if k.endswith(".grouping")}
    non_group_total_kieval = {k: v for k, v in total_kieval_dict.items() if k.endswith(".non_group")}

    total_kieval = {k: v for k, v in total_kieval_dict.items() if k.endswith(".kieval")}

    kieval_markdown_text = print_formatted_kieval_results(
        None, total_kieval, category="kieval", print_score=print_score
    )

    return kieval_markdown_text


def defaultdict_to_dict(d: Union[dict, defaultdict]):
    """Convert defaultdict to dict."""
    if isinstance(d, dict) or isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d
