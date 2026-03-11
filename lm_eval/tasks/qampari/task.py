"""
QAMPARI task: loads dataset from the official zip
(https://aggreg-qa.s3.amazonaws.com/qampari.zip) because the HF dataset has schema issues
and dataset scripts are no longer supported.
"""
import json
import os
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict
from lm_eval.api.task import ConfigurableTask
from lm_eval import utils

QAMPARI_URL = "https://aggreg-qa.s3.amazonaws.com/qampari.zip"


def _get_cache_dir():
    cache = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.expanduser("~/.cache/huggingface/datasets"),
    )
    data_dir = Path(cache) / "qampari_official"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_and_extract():
    data_dir = _get_cache_dir()
    zip_path = data_dir / "qampari.zip"
    # Look for jsonl files in data_dir or one level down (zip may extract into a subdir)
    def _find_jsonl(name):
        for base in (data_dir,):
            p = base / name
            if p.exists():
                return p
            for sub in base.iterdir():
                if sub.is_dir():
                    q = sub / name
                    if q.exists():
                        return q
        return None

    if _find_jsonl("dev_data.jsonl") is None:
        try:
            import urllib.request
            urllib.request.urlretrieve(QAMPARI_URL, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download QAMPARI from {QAMPARI_URL}. "
                f"Download the zip manually and extract it into {data_dir}. Error: {e}"
            ) from e
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
    return data_dir


def _find_jsonl_paths(data_dir):
    """Return dict of split_name -> Path for each jsonl file (data_dir or one level down)."""
    mapping = [
        ("train", "train_data.jsonl"),
        ("validation", "dev_data.jsonl"),
        ("test", "test_data.jsonl"),
    ]
    found = {}
    for split_name, filename in mapping:
        for base in (data_dir,):
            p = base / filename
            if p.exists():
                found[split_name] = p
                break
            for sub in base.iterdir():
                if sub.is_dir():
                    q = sub / filename
                    if q.exists():
                        found[split_name] = q
                        break
            if split_name in found:
                break
    return found


def _parse_jsonl(path):
    question_texts = []
    answers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            question_texts.append(d.get("question_text", ""))
            ans_list = d.get("answer_list", [])
            if ans_list and isinstance(ans_list[0], dict):
                first = ans_list[0].get("answer_text", "")
            elif ans_list:
                first = str(ans_list[0])
            else:
                first = ""
            answers.append(first)
    return Dataset.from_dict({"question_text": question_texts, "answer": answers})


def _load_qampari_dataset():
    """Build DatasetDict from official QAMPARI zip."""
    data_dir = _download_and_extract()
    paths = _find_jsonl_paths(data_dir)
    if not paths:
        raise FileNotFoundError(
            f"No QAMPARI jsonl files (train_data.jsonl, dev_data.jsonl, test_data.jsonl) found in "
            f"{data_dir} or its subdirectories.\n"
            f"Download the dataset manually:\n"
            f"  1. Get {QAMPARI_URL}\n"
            f"  2. Extract the zip so that {data_dir} contains dev_data.jsonl and test_data.jsonl "
            f"(or a subfolder containing them)."
        )
    return DatasetDict(
        {split_name: _parse_jsonl(p) for split_name, p in paths.items()}
    )


class QampariTask(ConfigurableTask):
    """QAMPARI generate_until task; dataset loaded from official zip in download()."""

    def __init__(self):
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qampari.yaml")
        config = utils.load_yaml_config(yaml_path, mode="full")
        config.pop("class", None)
        super().__init__(config=config)

    def download(self, dataset_kwargs=None):
        self.dataset = _load_qampari_dataset()
