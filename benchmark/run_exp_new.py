"""
To run with real mode:
python run_exp.py --backend dm --suite a10g --breakdown  --mode real
with synthetic mode:
python run_exp.py --backend dm --suite a10g --breakdown  --mode synthetic
default to synthetic mode.
"""

import argparse
import asyncio
import csv
from itertools import chain
import json
import re
import numpy as np
import os
import sys
import time
import random
from tqdm import tqdm
from typing import List, Tuple
from math import ceil
from collections import defaultdict, deque

import aiohttp

# from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import Request, dummy_prompt

# sys.path.append("../bench_lora")

def reward(response_time):
    response_ub = 6
    # assume degree 2 polynomial
    a = - 1 / response_ub ** 2
    if response_time < response_ub:
        return a * response_time ** 2 + 1
    else:
        return 0

def attainment_func(response_time):
    response_ub = 6
    # 1 for < slo, 0 for >= slo
    if response_time < response_ub:
        return 1
    else:
        return 0


GB = 1024**3

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []
vllm_packed_adapter_dir_to_url_map = {}


REMAINING_PREFILLS_PER_SERVER = defaultdict(dict)
REMAINING_DECODES_PER_SERVER = defaultdict(dict)
PREFILL_LOCK = asyncio.Lock()
DECODE_LOCK = asyncio.Lock()


def get_peak_mem(server):
    url = server + "/get_peak_mem"
    response = requests.post(url)
    return response.json()["peak_mem"]


async def send_request(
    backend: str,
    server: str,
    req_id: str,
    model_dir: str,
    adapter_dir: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    output_file: str,
    debug: bool,
    arrival_time: float
) -> None:
    request_start_time = time.time()
    scheduling_latency = max(0.0, request_start_time - arrival_time)
    headers = {"Content-Type": "application/json"}
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        url = server + "/generate"
    elif backend == "vllm-packed":
        url = vllm_packed_adapter_dir_to_url_map[adapter_dir] + "/generate"
    else:
        url = server + "/generate_stream"

    if backend in ["dm", "system", "baseline", "contiguous"]:
        data = {
            "model_dir": model_dir,
            "lora_dir": adapter_dir,
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_len,
                # 'temperature': 0.1,
            },
        }
    elif backend in ["lightllm"]:
        data = {
            "inputs": prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_len,
                # 'temperature': 0.1,
            },
        }
    elif backend in ["vllm", "vllm-packed"]:
        data = {
            "prompt": prompt,
            "max_tokens": output_len,
            "ignore_eos": True,
        }
    elif backend in ["toppings"]:
        adapter_id = int(adapter_dir.split("-")[-1])
        data = {
            "inputs": prompt,
            "parameters": {
                "frequency_penalty": 1,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
            "lora_id": adapter_id
        }
    # with open(f"fine_{output_file}", "a") as f:
    #     f.write(f"sent {req_id}, {server}\n")
    first_token_latency = None
    server_queuing_time, server_execution_time = None, None
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    if first_token_latency is None:
                        first_token_latency = time.time() - request_start_time
                        if backend == "toppings":
                            rank = int(re.search(r"rank-(\d+)", adapter_dir).group(1))
                            await PREFILL_LOCK.acquire()
                            REMAINING_PREFILLS_PER_SERVER[server][rank] -= prompt_len
                            PREFILL_LOCK.release()
                    
                    if backend == "toppings":
                        rank = int(re.search(r"rank-(\d+)", adapter_dir).group(1))
                        await DECODE_LOCK.acquire()
                        REMAINING_DECODES_PER_SERVER[server][rank] -= 1
                        DECODE_LOCK.release()
                    try:
                        chunk_str = chunk.decode("utf-8")
                        if 'queue_time' in chunk_str:
                            # print(chunk_str)
                            json_str = chunk_str.split("data:", 1)[1]
                            data_obj = json.loads(json_str)
                            server_queuing_time = data_obj['token'].get('queue_time')
                            server_execution_time = data_obj['token'].get('prefill_time')
                    except Exception as e:
                        print(f"Error decoding chunk: {e}")
                        chunk_str = ""
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            # output = json.loads(output)
            # print(output)

            if '"finished": -1' not in output:
                break
            else:
                first_token_latency = None
                break
            # #     print(output)
            # #     print(json.loads(output))
            # break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    tbt = (
        (request_latency - first_token_latency) / (output_len - 1)
        if output_len > 1
        else 0
    )
    log_line = (
        f"req_id {req_id} {server} adapter_dir {adapter_dir} prompt_len {prompt_len} output_len {output_len} "
        f"request_latency {request_latency:.4f} s, client_scheduling_latency {scheduling_latency:.4f} s, first_token_latency {first_token_latency:.4f} s, tbt {tbt:.4f} s"
    )
    
    if server_queuing_time is not None and server_execution_time is not None:
        log_line += f", server_prefill_queuing_time {server_queuing_time:.4f} s, server_prefill_execution_time {server_execution_time:.4f} s"

    print(log_line)
    with open(f"fine_{output_file}", "a") as f:
        f.write(log_line + "\n")
    REQUEST_LATENCY.append(
        (prompt_len, output_len, request_latency, first_token_latency, tbt, scheduling_latency)
    )
    return (prompt_len, output_len, request_latency, first_token_latency, tbt, scheduling_latency)


async def calc_cost_toppings(req, server, operating_points):
    time_to_process = 0.0
    
    for rank in [8, 16, 32, 64, 128]:
        await PREFILL_LOCK.acquire()
        await DECODE_LOCK.acquire()
        remaining_prefills = REMAINING_PREFILLS_PER_SERVER[server].get(rank, 0)
        # remaining_decodes = REMAINING_DECODES_PER_SERVER[server].get(rank, 0)
        # jitter_factor = 1 + random.uniform(-0.25, 0.25) 
        # jittered_decodes = max(0, int(remaining_decodes * jitter_factor))
        # total_tokens_rank = remaining_prefills + jittered_decodes
        avg_decode = 70
        total_tokens_rank = remaining_prefills + avg_decode
        PREFILL_LOCK.release()
        DECODE_LOCK.release()
        time_for_rank = total_tokens_rank / operating_points[rank]
        time_to_process += time_for_rank
        
    return time_to_process

async def benchmark_toppings(
    backend: str,
    input_requests: List[Tuple[str, str, str, int, int]],
    output,
    servers,
    debug=False,
) -> None:
    start = time.time()
    tasks: List[asyncio.Task] = []
    for req in input_requests:
        arrival_time = start + req.req_time
        sleep_time = arrival_time - time.time()
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        if debug:
            print(
                f"{req.req_id} {req.req_time:.5f} wait {start + req.req_time - time.time():.5f} "
                f"{req.adapter_dir}"
            )
        # print(req)
        
        # operating_points = {
        #     8: 5500,
        #     16: 5400,
        #     32: 5250,
        #     64: 5000,
        #     128: 4500,
        # }  # operating point, fn of max rank, ND96asrv 8xA100 80GB
        
        operating_points = {8: 5950, 16: 5150, 32: 4700, 64: 5050, 128: 4650} # prod 7b tp1, NC24ads
        
        #! todo, select server with toppings algo
        chosen_server = servers[0]
        min_cost = np.inf

        for server in servers:
            cost = await calc_cost_toppings(req, server, operating_points)
            if cost < min_cost:
                min_cost = cost
                chosen_server = server

        assert chosen_server is not None and chosen_server in servers, f"Chosen server {chosen_server} is invalid."

        await PREFILL_LOCK.acquire()
        rank = int(re.search(r"rank-(\d+)", req.adapter_dir).group(1))
        if rank not in REMAINING_PREFILLS_PER_SERVER[chosen_server]:
            REMAINING_PREFILLS_PER_SERVER[chosen_server][rank] = 0
        REMAINING_PREFILLS_PER_SERVER[chosen_server][rank] += req.prompt_len
        PREFILL_LOCK.release()
            
        await DECODE_LOCK.acquire()
        rank = int(re.search(r"rank-(\d+)", req.adapter_dir).group(1))
        if rank not in REMAINING_DECODES_PER_SERVER[chosen_server]:
            REMAINING_DECODES_PER_SERVER[chosen_server][rank] = 0
        REMAINING_DECODES_PER_SERVER[chosen_server][rank] += (req.output_len - 1)
        DECODE_LOCK.release()
        
        task = asyncio.create_task(
            send_request(
                backend,
                chosen_server,
                req.req_id,
                req.model_dir,
                req.adapter_dir,
                req.prompt,
                req.prompt_len,
                req.output_len,
                output,
                debug,
                arrival_time
            )
        )
        tasks.append(task)
    latency = await asyncio.gather(*tasks)
    return latency


def get_adapter_dirs(num_adapters, adapter_dirs, backend=None):
    ret = []
    num_iter = num_adapters // len(adapter_dirs) + 1

    if backend == "vllm-packed":
        num_iter = num_adapters // len(adapter_dirs)

    for i in range(num_iter):
        for adapter_dir in adapter_dirs:
            ret.append(adapter_dir + f"-{i}")
    print(ret)
    return ret


def get_res_stats(
    per_req_latency, benchmark_time, backend, warmup_time=0, warmup_num=0
):
    # get throughput
    num_abort = len([i for i in per_req_latency if i[3] is None])
    per_req_latency = [i for i in per_req_latency if i[3] is not None]
    throughput = len(per_req_latency) / benchmark_time
    # print(per_req_latency)
    # if backend == "dm":
    #     peak_mem = get_peak_mem(server)
    #     print(f"GPU peak memory (GB):", [[f"{x / GB:.2f}" for x in tpg] for tpg in peak_mem])
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Non warmup time: {benchmark_time - warmup_time:.2f} s")
    print(f"Total requests: {len(per_req_latency)}")
    print(f"Non warmup requests: {len(per_req_latency) - warmup_num}")
    print(f"Aborted Request: {num_abort}")
    print(f"Throughput: {throughput:.2f} requests/s")

    strip_throughput = (len(per_req_latency) - warmup_num) / max(
        1e-9, benchmark_time - warmup_time
    )
    print(f"Throughput strip: {strip_throughput:.2f} requests/s")

    # Exclude first warmup_num requests from further statistics
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    if warmup_num >= len(per_req_latency):
        raise ValueError(
            "Warmup number exceeds number of requests whose latency was logged"
        )
    else:
        stats_base = per_req_latency[warmup_num:]

    # each entry in stats_base: (prompt_len, output_len, request_latency, first_token_latency, tbt, scheduling_delay)
    e2e_latencies = [latency for _, _, latency, _, _, _ in stats_base]
    per_token_latencies = [
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency, _, _, _ in stats_base
        if (prompt_len + output_len) > 0
    ]
    per_output_token_latencies = [
        latency / output_len
        for _, output_len, latency, _, _, _ in stats_base
        if output_len > 0
    ]
    first_token_latencies = [latency for _, _, _, latency, _, _ in stats_base]
    tbts = [latency for _, _, _, _, latency, _ in stats_base]
    num_abort = len([i for i in stats_base if i[3] is None])
    abort_satisfaction = [0] * num_abort
    satisfactions = [
        reward(latency) for _, _, _, latency, _, _ in stats_base
    ] + abort_satisfaction
    attainments = [
        attainment_func(latency) for _, _, _, latency, _, _ in stats_base
    ] + abort_satisfaction

    scheduling_delays = [delay[5] for delay in stats_base if len(delay) > 5]
    
    if len(scheduling_delays) == 0:
        scheduling_delays = [0.0] * len(stats_base)

    metrics = {
        "e2e": e2e_latencies,
        "per_token": per_token_latencies,
        "per_output_token": per_output_token_latencies,
        "scheduling": scheduling_delays,
        "first_token": first_token_latencies,
        "tbt": tbts,
        "satisfaction": satisfactions,
        "attainment": attainments,
    }

    stats = {}
    for name, values in metrics.items():
        stats[name] = {
            "avg": np.mean(values),
            **{f"p{p}": np.percentile(values, p) for p in percentiles},
        }

    # dump results
    if backend == "dm":
        # TODO
        # single_gpu_peak_mem = peak_mem
        single_gpu_peak_mem = 0
    else:
        single_gpu_peak_mem = 0

    result = {
        "total_time": benchmark_time,
        "gpu_peak_mem": single_gpu_peak_mem,
        "num_abort": num_abort,
        "throughput": throughput,
        "strip_throughput": strip_throughput,
        "stats": stats,
        "backend": backend,
    }
    res = {"result": result}

    return res


def read_requests(trace_file):
    requests = []
    adapter_dirs = set()
    with open(trace_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            elements = line.split(",")
            requests.append(
                Request(
                    req_id=int(elements[0]),
                    model_dir=elements[1],
                    adapter_dir=elements[2],
                    prompt=dummy_prompt(int(elements[3])),
                    prompt_len=int(elements[3]),
                    output_len=int(elements[4]),
                    req_time=float(elements[5]),
                )
            )
            # requests.append((int(elements[0]),elements[1],elements[2],int(elements[3]),int(elements[4]),float(elements[5])))
            adapter_dirs.add(elements[2])
    requests.sort(key=lambda r: r.req_time)
    return list(adapter_dirs), requests


def run_exp(
    backend,
    servers,
    trace_file,
    output,
    debug=False,
    warmup_time: int = 60,
    warmup_num: int = 600,
):
    # first generate your data using real_trace/clean_chat_data.py
    # base_model = BASE_MODEL[model_setting]
    # adapter_dirs = LORA_DIR[model_setting]
    adapter_dirs, requests = read_requests(trace_file=trace_file)
    # print(requests)
    avg_prompt_len = np.mean([req.prompt_len for req in requests])
    avg_output_len = np.mean([req.output_len for req in requests])
    avg_len = np.mean([req.prompt_len + req.output_len for req in requests])
    print(
        "num_adapters",
        len(adapter_dirs),
        "num_requests",
        len(requests),
        "avg_len:",
        avg_len,
        "avg_prompt_len:",
        avg_prompt_len,
        "avg_output_len:",
        avg_output_len,
    )

    if debug:
        print("num requests:", len(requests))
        for req in requests[:4]:
            print(req)
    if backend == "baseline" or backend == "toppings":
        # benchmark
        random.seed(42)
        shuffled = adapter_dirs.copy()
        random.shuffle(shuffled)

        # Split shuffled list into n parts as evenly as possible
        k, m = divmod(len(shuffled), len(servers))
        # server_map = {adapter:server_name for adapter in shuffled[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i, server_name in enumerate(servers)}
        server_map = {}
        for i, server_name in enumerate(servers):
            start = i * k + min(i, m)
            end = (i + 1) * k + min(i + 1, m)
            for adapter in shuffled[start:end]:
                server_map[adapter] = server_name
        print(server_map)
        
        adapter_allocations = defaultdict(list)
        for adapter, server in server_map.items():
            adapter_allocations[server].append(adapter)
            
        # with open("server_map.csv", "w") as f:
        #     f.write("server,adapters\n")
        #     for server, adapters in adapter_allocations.items():
        #         f.write(f"{server},{' '.join(adapters)}\n")
                
        with open("server_map.json", "w") as f:
            json.dump(server_map, f, indent=4)
        
        if backend == "toppings":
            benchmark_start_time = time.time()
            per_req_latency = asyncio.run(
                benchmark_toppings(backend, requests, output, servers, debug)
            )
            benchmark_end_time = time.time()
            benchmark_time = benchmark_end_time - benchmark_start_time

    res = get_res_stats(
        per_req_latency,
        benchmark_time,
        backend,
        warmup_time=warmup_time,
        warmup_num=warmup_num,
    )

    with open(output, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["system", "baseline", "contiguous", "toppings"],
    )

    # parser.add_argument("--model-setting", type=str, default="S1")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--append", action="store_true")

    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-lora-swap", action="store_true")
    parser.add_argument("--no-lora-copy", action="store_true")
    parser.add_argument("--trace-file-path", required=True)

    parser.add_argument("--servers", "-s", type=str, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--warmup-time",
        type=int,
        default=60,
        help="Number of seconds considered as warmup, excluded from stats",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=600,
        help="Number of requests considered as warmup, excluded from stats (rps * warmup-time)",
    )
    args = parser.parse_args()

    assert not args.no_lora_copy or args.no_lora_compute
    assert not (args.debug and args.breakdown)

    # set output file name
    if args.output is None:
        args.output = f"all_results_{args.mode}_" + args.backend + ".jsonl"
    if args.no_lora_swap and args.no_lora_compute and args.no_lora_copy:
        args.output = "no_lora_compute_swap_copy_results.jsonl"
    elif args.no_lora_swap and args.no_lora_compute:
        args.output = "no_lora_compute_swap_results.jsonl"
    elif args.no_lora_swap:
        args.output = "no_lora_swap_results.jsonl"
    elif args.no_lora_compute:
        args.output = "no_lora_compute_results.jsonl"
    if args.debug or args.breakdown:
        args.output = "debug_" + args.output

    # suites = get_all_suites(mode=args.mode, debug=args.debug, suite=args.suite, breakdown=args.breakdown)

    if not args.append:
        os.system(f"rm {args.output}")
        os.system(f"rm allocation_log.txt")
        os.system(f"rm fine_{args.output}")
        results = []
    else:
        with open(args.output, "r") as f:
            lines = f.readlines()
        results = [json.loads(line)["config"] for line in lines]

    # for config in tqdm(suites, desc="suites"):
    #     if to_dict(config) not in results:
    stats = run_exp(
        args.backend,
        args.servers,
        args.trace_file_path,
        args.output,
        args.debug,
        args.warmup_time,
        args.warmup_requests,
    )