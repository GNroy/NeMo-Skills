# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random
import re

def format_entry(entry, detailed_thinking="off"):
    if detailed_thinking == "off":
        input_template = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\ndetailed thinking {detailed_thinking}" + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n" + f"{entry['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise NotImplementedError
    output_template = "{solution}<|eot_id|>\n"
    entry = {"input": input_template,
             "output": output_template.format(solution=entry["solution"]),
             "expected_answer": entry["answer"]}
    return entry


def write_data_to_file(output_file, data, detailed_thinking):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry, detailed_thinking), fout)
            fout.write("\n")


def save_data(detailed_thinking):
    dataset = load_dataset("GAIR/LIMO")["train"]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"train_detailed_thinking{detailed_thinking.capitalize()}.jsonl"
    write_data_to_file(output_file, dataset, detailed_thinking)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detailed_thinking",
        default="off",
        choices=("on", "off"),
        help="Set to on to enable detailed thinking.",
    )
    args = parser.parse_args()

    save_data(args.detailed_thinking)
