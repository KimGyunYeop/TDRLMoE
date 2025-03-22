# utils_qa.py
import collections
import json
import logging
import os
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
):
    """
    Hugging Face 공식 예제에 있는 post-processing 함수. 
    (start_logits, end_logits) => 실제 context 상의 문자열로 변환
    """
    all_start_logits, all_end_logits = predictions

    # example과 feature 매핑
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

    # 각 example을 순회하며 후보(span)을 찾아내기
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            # CLS(= no_answer) 위치 가정
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            start_indexes = np.argsort(start_logits)[-1:-n_best_size - 1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size - 1:-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or (end_index - start_index + 1) > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    prelim_predictions.append(
                        {
                            "offsets": (start_char, end_char),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # null prediction도 후보로 추가 (version_2_with_negative 일 때)
        if version_2_with_negative and min_null_prediction is not None:
            prelim_predictions.append(min_null_prediction)

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # 답 substring 복원
        for pred in predictions:
            offsets = pred["offsets"]
            pred["text"] = context[offsets[0]: offsets[1]]

        if not predictions:
            predictions.insert(0, {"text": "", "score": 0.0})

        # 가장 상위의 text를 정답으로
        best_pred = predictions[0]["text"]
        all_predictions[example["id"]] = best_pred

        # nbest 저장
        all_nbest_json[example["id"]] = [
            {
                "text": pred["text"],
                "score": float(pred["score"]),
                "start_logit": float(pred["start_logit"]),
                "end_logit": float(pred["end_logit"]),
            }
            for pred in predictions
        ]

    # 필요시 JSON 파일로 저장
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        prediction_file = os.path.join(output_dir, f"{prefix}_predictions.json" if prefix else "predictions.json")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False))

    return all_predictions, all_nbest_json
