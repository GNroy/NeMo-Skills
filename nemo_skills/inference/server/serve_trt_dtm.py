# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import asyncio
import copy
import logging
import re
import sys
import time
import uuid
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
import torch
from fastapi import FastAPI, HTTPException
from mpi4py import MPI
from pydantic import BaseModel
from tensorrt_llm.runtime.model_runner_cpp import ExternalDraftTokensConfig, ModelRunnerCpp
from transformers import AutoTokenizer

app = FastAPI(title="TensorRT-LLM Server")


# keeping it here to make this file self-contained. This is duplicated from model.py
def trim_after_stop_phrases(text: str, stop_phrases: List[str]) -> str:
    """Removes everything after the last stop token."""
    if not stop_phrases:
        return text
    # Escape all special characters in stop phrases
    escaped_stop_phrases = [re.escape(sp) for sp in stop_phrases]
    return re.split("|".join(escaped_stop_phrases), text, maxsplit=1)[0]


def parse_input(input_texts: str, tokenizer):
    batch_input_ids = [
        tokenizer.encode(
            input_text,
            add_special_tokens=False,
        )
        for input_text in input_texts
    ]
    batch_input_ids = [torch.tensor(x, dtype=torch.int32) for x in batch_input_ids]
    input_lengths = [x.size(0) for x in batch_input_ids]

    return batch_input_ids, input_lengths


def get_output(output_ids, input_length, max_output_len, tokenizer, eos_token) -> tuple[str, list[str], int]:
    """Returns detokenized text and the number of tokens."""
    output_begin = input_length
    output_end = input_length + max_output_len
    outputs = output_ids[output_begin:output_end]
    eos_ids = (outputs == eos_token).nonzero(as_tuple=True)[-1]
    if len(eos_ids) > 0:
        outputs = outputs[: eos_ids[0]]
    outputs = outputs.tolist()
    return tokenizer.decode(outputs), tokenizer.convert_ids_to_tokens(outputs), len(outputs)


def load_tokenizer(tokenizer_dir: str):

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        legacy=False,
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    return tokenizer, pad_id, end_id


class TensorRTLLM:
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        max_batch_size: Optional[int] = None,
        max_input_len: Optional[int] = None,
        max_output_len: Optional[int] = None,
        max_beam_width: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        kv_cache_free_gpu_memory_fraction: Optional[float] = None,
        disable_chunked_context: bool = False,
    ):
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(tokenizer_dir=target_model_path)

        runner_kwargs = dict(
            rank=tensorrt_llm.mpi_rank(),
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=max_beam_width,
            enable_chunked_context=not disable_chunked_context,
            kv_cache_enable_block_reuse=True,
        )

        self.target_runner = ModelRunnerCpp.from_dir(
            engine_dir=target_model_path,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction * 0.9,
            **runner_kwargs,
        )
        self.draft_runner = ModelRunnerCpp.from_dir(
            engine_dir=draft_model_path,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction * 0.1,
            **runner_kwargs,
        )

        self.timeout = timeout_seconds

        self.active_generations = {}
        self.executor = ThreadPoolExecutor(max_workers=1024)

    def get_output(
        self,
        batch_input_ids,
        input_lengths,
        max_output_token,
        top_k,
        top_p,
        top_p_min,
        temperature,
        repetition_penalty,
        random_seed,
        stop_words_list,
        top_logprobs,
    ):
        if top_p_min == 0.0:
            top_p_min = None
        # TODO: pass as param
        draft_len = 10

        num_tokens_to_check = 20
        start_time = time.time()
        prefix = batch_input_ids
        input_batch_size = len(batch_input_ids)  # Note as `BS`
        beam_width = 1  # Note as `BW`

        # Repack the output like the output of function `generate`
        input_len = [len(p) for p in batch_input_ids]
        max_seq_len = [i + max_output_token for i in input_len]
        outputs = {}
        outputs["output_ids"] = torch.full([1, 1, max(max_seq_len)], self.end_id, dtype=torch.int32)
        for bi in range(input_batch_size):
            outputs["output_ids"][bi, :, : input_len[bi]] = batch_input_ids[bi]
        outputs["sequence_lengths"] = torch.full([input_batch_size, beam_width], 0, dtype=torch.int32)

        n_draft_token = [0 for _ in range(input_batch_size)]
        n_accept_token = [0 for _ in range(input_batch_size)]
        n_iteration = 0

        while True:
            n_iteration += 1
            print("HERE")
            draft = self.draft_runner.generate(
                prefix,
                max_new_tokens=draft_len,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                top_p_min=top_p_min,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                output_log_probs=bool(top_logprobs is not None),
                input_lengths=input_lengths,
                return_dict=True,
                output_sequence_lengths=True,
                streaming=False,
            )
            print("THERE")
            prefix_len = [len(prefix[i]) for i in range(input_batch_size)]
            # draft["output_ids"].shape -> [BS, BW, maxSL]
            # draft["sequence_lengths"].shape -> [BS, BW]
            # draft["generation_logits"].shape -> [BS, BW, draft_len, vocab_size]
            d_ids = [[self.end_id]] * input_batch_size
            d_seq_len = draft["sequence_lengths"][:, 0].tolist()
            d_len = [d_seq_len[bi] - prefix_len[bi] for bi in range(input_batch_size)]
            for bi in range(input_batch_size):
                l, r = prefix_len[bi], d_seq_len[bi]
                if l >= r:  # No useful draft tokens
                    continue
                d_ids[bi] = draft["output_ids"][bi, 0, l:r].tolist()
            print("HERE2")
            target = self.target_runner.generate(
                batch_input_ids=prefix,
                draft_tokens_list=d_ids,
                max_new_tokens=draft_len + 1,
                end_id=self.end_id,
                pad_id=self.pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                top_p_min=top_p_min,
                repetition_penalty=repetition_penalty,
                random_seed=random_seed,
                output_log_probs=bool(top_logprobs is not None),
                input_lengths=input_lengths,
                return_dict=True,
                output_sequence_lengths=True,
                streaming=False,
            )
            print("THERE2")
            t_ids = [None] * input_batch_size
            t_seq_ids = [None] * input_batch_size
            t_seq_len = target["sequence_lengths"][:, 0].tolist()
            t_len = [t_seq_len[bi] - prefix_len[bi] for bi in range(input_batch_size)]

            # Update output and tokens for next iteration
            for bi in range(input_batch_size):
                gbi = 0
                l = prefix_len[bi]
                r = min(t_seq_len[bi], max_seq_len[gbi])
                t_ids[bi] = target["output_ids"][bi, 0, l:r].tolist()
                t_seq_ids[bi] = target["output_ids"][bi, 0, :r]
                outputs["output_ids"][gbi, 0, l:r] = torch.IntTensor(t_ids[bi])
                outputs["sequence_lengths"][gbi, 0] = r
                n_draft_token[gbi] += d_len[bi]
                length = min(d_len[bi], t_len[bi], max_seq_len[gbi] - prefix_len[bi])
                res = [d_ids[bi][i] == t_ids[bi][i] for i in range(length)]
                n_accept_token[gbi] += ((~torch.BoolTensor(res)).cumsum(axis=-1) < 1).sum()

            # Evaluate stop criteria and prepare inputs for next iteration
            prefix_next = []
            batch_slot_next = []
            for bi in range(input_batch_size):
                gbi = 0
                # Stop due to output length
                if len(t_seq_ids[bi]) >= max_seq_len[gbi]:
                    continue  # No need to update for the stopped requests
                # Stop due to the same output. Normally target should return 1 more token.
                # if (d_ids is not None and np.array_equal(d_ids[bi], t_ids[bi])):
                #     continue
                # Stop due to no change (hit early stopping)
                if np.array_equal(t_seq_ids[bi].cpu().numpy(), prefix[bi].cpu().numpy()):
                    continue
                # Stop due to end words
                if self.end_id in t_seq_ids[bi][prefix_len[bi] :]:
                    continue

                seq_length = outputs['sequence_lengths']
                generation_suffix = outputs['output_ids'][0, 0, seq_length[0] - num_tokens_to_check : seq_length[0]]
                out_string = get_output(generation_suffix, 0, num_tokens_to_check, self.tokenizer, self.end_id)[0]
                matching_stop_word = None
                for stop_word in stop_words_list:
                    if stop_word in out_string:
                        matching_stop_word = stop_word
                        break

                if matching_stop_word is not None:
                    break

                if self.timeout:
                    current_time = time.time() - start_time
                    if current_time >= self.timeout:
                        break

                prefix_next.append(t_seq_ids[bi])
                batch_slot_next.append(gbi)
            prefix = prefix_next
            if len(prefix) == 0:  # Leave while loop if no request remained
                break

        print(f"Acceptance ratio: {n_accept_token[0] / n_draft_token[0] * 100 :6.2f}%")

        out_string, out_tokens, num_generated_tokens = get_output(
            outputs['output_ids'][0, 0], input_lengths[0], outputs['sequence_lengths'][0], self.tokenizer, self.end_id
        )
        # TODO: the number of tokens is not exact, because we might trim the output a bit,
        #       but close enough for practical purposes
        for stop_word in stop_words_list:
            if stop_word in out_string:
                matching_stop_word = stop_word
                break
        if matching_stop_word is not None:
            out_string = trim_after_stop_phrases(out_string, stop_words_list)
            # adding it back, since we only need to remove what's *after* the stop phrase
            out_string += matching_stop_word
        else:
            # trtllm removes end id if it was the stop reason
            # this is a hack to add it back, but we are going to include it even when
            # it was not generated by the model e.g. if we stopped due to max tokens
            out_string += self.tokenizer.decode(self.end_id)

        generation_time = int(round(time.time() - start_time))

        result = {
            'generation': out_string,
            'num_generated_tokens': num_generated_tokens,
            'generation_time': generation_time,
            'draft_acceptance_ratio': n_accept_token[0] / n_draft_token[0] * 100.0,
        }

        return result

    @torch.no_grad()
    def start_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        generation_id = str(uuid.uuid4())
        batch_input_ids, input_lengths = parse_input([data["prompt"]], self.tokenizer)

        future = self.executor.submit(
            self.get_output,
            batch_input_ids,
            input_lengths,
            data["max_new_tokens"],
            data["top_k"],
            data["top_p"],
            data["top_p_min"],
            data["temperature"],
            data["repetition_penalty"],
            data["random_seed"],
            data["stop_words_list"],
            data["top_logprobs"],
        )

        self.active_generations[generation_id] = future

        return generation_id

    def get_generation(self, generation_id: str) -> Dict[str, Any]:
        if generation_id not in self.active_generations:
            raise HTTPException(status_code=404, detail="Generation not found")

        future = self.active_generations[generation_id]

        if future.done():
            result = future.result()
            # Clean up completed generation
            del self.active_generations[generation_id]
            return result
        else:
            return None

    def cancel_request(self, request_id):
        self.runner.session.cancel_request(request_id)

    def cancel_generation(self, generation_id: str) -> Dict[str, Any]:
        if generation_id not in self.active_generations:
            raise HTTPException(status_code=404, detail="Generation not found")

        future = self.active_generations[generation_id]
        future.cancel()

        # Clean up canceled generation
        del self.active_generations[generation_id]

        return {"status": "canceled"}


class GenerationRequest(BaseModel):
    prompt: str
    tokens_to_generate: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: float = 1.0
    top_p_min: float = 0.0
    repetition_penalty: float = 1.2
    random_seed: int = 0
    stop_words_list: Optional[List[str]] = None
    top_logprobs: Optional[int] = None


class GenerationResponse(BaseModel):
    generation: Optional[str] = None
    num_generated_tokens: Optional[int] = None
    generation_time: Optional[int] = None
    tokens: Optional[list[str]] = None
    logprobs: Optional[list[float]] = None


class GenerationResponseAsync(BaseModel):
    generation_id: str


class CancelGenerationResponse(BaseModel):
    status: str


class GetGenerationRequest(BaseModel):
    generation_id: str


class MPIWrapper:
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        max_batch_size: Optional[int] = None,
        max_input_len: Optional[int] = None,
        max_output_len: Optional[int] = None,
        max_beam_width: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        kv_cache_free_gpu_memory_fraction: Optional[float] = None,
        disable_chunked_context: bool = False,
    ):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.model = TensorRTLLM(
            target_model_path=target_model_path,
            draft_model_path=draft_model_path,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=max_beam_width,
            kv_cache_free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
            timeout_seconds=timeout_seconds,
            disable_chunked_context=disable_chunked_context,
        )

        self.app = None
        if self.rank == 0:
            self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="TensorRT-LLM Service")

        @app.put("/generate", response_model=GenerationResponse)
        async def generate(request: GenerationRequest):
            data = {
                "prompt": request.prompt,
                "max_new_tokens": request.tokens_to_generate,
                "temperature": request.temperature,
                "top_k": None if request.top_k == 0 else request.top_k,
                "top_p": request.top_p,
                "top_p_min": request.top_p_min,
                "repetition_penalty": request.repetition_penalty,
                "random_seed": request.random_seed,
                "stop_words_list": request.stop_words_list,
                "top_logprobs": request.top_logprobs,
            }

            self.comm.Barrier()
            data = self.comm.bcast(data, root=0)

            generation_id = self.model.start_generation(data)

            while True:
                output = self.model.get_generation(generation_id)
                if output is not None:
                    return output
                await asyncio.sleep(0.1)

        @app.put("/generate_async", response_model=GenerationResponseAsync)
        async def generate_async(request: GenerationRequest):
            data = {
                "prompt": request.prompt,
                "max_new_tokens": request.tokens_to_generate,
                "temperature": request.temperature,
                "top_k": None if request.top_k == 0 else request.top_k,
                "top_p": request.top_p,
                "top_p_min": request.top_p_min,
                "repetition_penalty": request.repetition_penalty,
                "random_seed": request.random_seed,
                "stop_words_list": request.stop_words_list,
                "top_logprobs": request.top_logprobs,
            }

            self.comm.Barrier()
            data = self.comm.bcast(data, root=0)

            generation_id = self.model.start_generation(data)
            return {'generation_id': generation_id}

        @app.put("/get_generation", response_model=GenerationResponse)
        async def get_generation(request: GetGenerationRequest):
            generation_id = request.generation_id

            output = self.model.get_generation(generation_id)
            if output is not None:
                return output
            return {'generation': None}

        @app.put("/cancel_generation", response_model=CancelGenerationResponse)
        async def cancel_generation(request: GetGenerationRequest):
            generation_id = request.generation_id
            return self.model.cancel_generation(generation_id)

        return app

    def worker_loop(self):
        """Worker loop for non-rank-0 processes"""
        while True:
            self.comm.Barrier()
            data = None
            data = self.comm.bcast(data, root=0)
            if data is None:
                continue
            self.model.start_generation(data)

    def run(self, host: str = "0.0.0.0", port: int = 5000):
        if self.rank == 0:
            import uvicorn

            uvicorn.run(self.app, host=host, port=port, ws_max_queue=1500)
        else:
            self.worker_loop()


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--draft_model_path", required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--max_batch_size", type=int, default=None, help="Maximum batch size")
    parser.add_argument("--max_input_len", type=int, default=None, help="Maximum input length")
    parser.add_argument("--max_output_len", type=int, default=None, help="Maximum output length")
    parser.add_argument("--max_beam_width", type=int, default=None, help="Maximum beam width")
    parser.add_argument(
        "--timeout_seconds", type=int, default=None, help="No session should take longer than the timeout"
    )
    parser.add_argument(
        "--kv_cache_free_gpu_memory_fraction", type=float, default=0.9, help="Free GPU memory fraction for cache"
    )
    parser.add_argument("--disable_chunked_context", action="store_true", help="Disable chunked context")
    args = parser.parse_args()

    wrapper = MPIWrapper(
        target_model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_beam_width=args.max_beam_width,
        timeout_seconds=args.timeout_seconds,
        kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
        disable_chunked_context=args.disable_chunked_context,
    )
    wrapper.run(host=args.host, port=args.port)


if __name__ == "__main__":

    class LogFilter(logging.Filter):
        def filter(self, record):
            filter_strings = (
                "PUT /generate HTTP/1.1",
                "PUT /get_generation HTTP/1.1",
                "PUT /generate_async HTTP/1.1",
                "PUT /cancel_generation HTTP/1.1",
            )
            return all(filter_string not in record.getMessage() for filter_string in filter_strings)

    logging.getLogger('uvicorn.access').addFilter(LogFilter())
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    main()
