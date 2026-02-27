# !python -m pip install numpy icecream scikit-learn pprint


import numpy as np
import random
import heapq
from itertools import chain
import re
import bisect
import json
from pprint import pprint
from math import ceil
from functools import total_ordering
from tqdm.auto import tqdm
from collections import defaultdict, deque
from sklearn.linear_model import LinearRegression
from icecream import ic
import os
import time


def reset_allocation_log():
    with open("./allocation_log.txt", "w") as f:
        f.write("")


reset_allocation_log()

rps_list = [56, 60, 64, 68]

for rps in tqdm(rps_list):
    TRACE_FILE = f"/mnt/azureml/cr/j/a38c81c135ab437bbf7444755a941e24/exe/wd/S-LoRA/benchmarks/trace.csv"
    # servers = [
    #     "http://10.0.0.1:8000",
    #     "http://10.0.0.2:8000",
    #     "http://10.0.0.3:8000",
    #     "http://10.0.0.4:8000",
    #     "http://10.0.0.5:8000",
    #     "http://10.0.0.6:8000",
    #     "http://10.0.0.7:8000",
    #     "http://10.0.0.8:8000",
    # ]
    # servers = ["http://10.0.0.1:8000", "http://10.0.0.2:8000", "http://10.0.0.3:8000", "http://10.0.0.4:8000"]
    servers = ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
    server_map_folder = (
        f"server_maps/{TRACE_FILE.split('/')[-1].replace('.csv', '_server_maps')}"
    )
    server_map_folder += f"_{len(servers)}_servers"
    os.makedirs(server_map_folder, exist_ok=True)
    os.makedirs(f"{server_map_folder}/system", exist_ok=True)
    os.makedirs(f"{server_map_folder}/baseline", exist_ok=True)
    os.makedirs(f"{server_map_folder}/contiguous", exist_ok=True)

    backend = "system"
    step = 60

    @total_ordering
    class Request:
        def __init__(
            self,
            req_id,
            model_dir,
            adapter_dir,
            prompt,
            prompt_len,
            output_len,
            req_time,
        ):
            self.req_id = req_id
            self.model_dir = model_dir
            self.adapter_dir = adapter_dir
            self.prompt = prompt
            self.prompt_len = prompt_len
            self.output_len = output_len
            self.req_time = req_time

        def __repr__(self):
            return (
                f"req_id={self.req_id}, "
                f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, "
                f"prompt_len={self.prompt_len}, output_len={self.output_len}, "
                f"req_time={self.req_time}"
            )

        def __eq__(self, other):
            return self.req_id == other.req_id

        def __lt__(self, other):
            return self.req_time < other.req_time

    def dummy_prompt(prompt_len):
        return "Hello " * prompt_len

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

    adapter_dirs, requests = read_requests(trace_file=TRACE_FILE)
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

    def ema_next(values: list, alpha: float = 0.5):

        assert values, "no values found when computing ema"
        model = LinearRegression()
        model.fit(np.arange(len(values)).reshape(-1, 1), values)

        # Predict a value beyond known points
        extrapolated_value = model.predict(np.array([[len(values)]])).item()
        return max(min(values), extrapolated_value, 1)

    def compare_with_prev_alloc(
        adapter_groups,
        prev_adapter_groups,
        assigned_instances_per_rank,
        prev_assigned_instances_per_rank,
        adapter_to_tps=None,
        jaccard_threshold=0.5,
    ):
        """
        Compares the current and previous allocations to identify any changes in server assignments.
        assigned_instances_per_rank: {rank: [server_indices]}
        adapter_groups: [(adapter_name, routing_probability),]
        """
        server_rename_map = {}  # {old_server: new_server}

        for rank, prev_servers in prev_assigned_instances_per_rank.items():
            if rank in assigned_instances_per_rank:
                curr_servers = assigned_instances_per_rank[rank]
                if len(prev_servers) == len(curr_servers):
                    for old_server, new_server in zip(prev_servers, curr_servers):
                        server_rename_map[old_server] = new_server
                else:
                    for prev_server in prev_servers:
                        old_adapters = set(
                            adapter for adapter, _ in prev_adapter_groups[prev_server]
                        )
                        for curr_server in curr_servers:
                            new_adapters = set(
                                adapter for adapter, _ in adapter_groups[curr_server]
                            )
                            intersection_adapters_sum_tps = sum(
                                adapter_to_tps.get(adapter, 0)
                                for adapter in old_adapters & new_adapters
                            )
                            union_adapters_sum_tps = sum(
                                adapter_to_tps.get(adapter, 0)
                                for adapter in old_adapters | new_adapters
                            )
                            if (
                                union_adapters_sum_tps > 0
                                and intersection_adapters_sum_tps
                                / union_adapters_sum_tps
                                > jaccard_threshold
                            ):
                                server_rename_map[curr_server] = prev_server
                                break

        return server_rename_map

    def ensure_all_placed(adapter_groups, epsilon: float = 0.1):
        """
        Ensure that the probability of adapter placements sums to within epsilon of 1 for each adapter
        """

        utils = {}
        for group in adapter_groups:
            for adapter, util in group:
                if adapter in utils:
                    utils[adapter] += util
                else:
                    utils[adapter] = util

        for adapter, util in utils.items():
            assert (
                abs(util - 1) < epsilon
            ), f"Adapter {adapter} not fully placed, util={util}"

    def select_server(
        server_map, probability_sum, req, adapter_groups=None, adapter_demand=None
    ):
        """
        Selects a server for the given request based on the provided probability distribution.
        """
        available_servers, prob_thresholds = (
            server_map[req.adapter_dir],
            probability_sum[req.adapter_dir],
        )
        rand_prob = random.random()
        try:
            chosen_server = available_servers[
                min(
                    bisect.bisect_left(prob_thresholds, rand_prob),
                    len(available_servers) - 1,
                )
            ]
        except Exception as e:
            print(
                f"Error in bisecting {prob_thresholds} with rand_prob {rand_prob}, available_servers {available_servers}: {e}. Falling back to first in list if exists."
            )
            print(server_map)
            print(probability_sum)
            print(adapter_groups)
            print(adapter_demand)
            if available_servers:
                chosen_server = available_servers[0]
            else:
                raise Exception(f"No available servers for adapter {req.adapter_dir}")
        return chosen_server

    def flatten_dict_values(d):
        vals = list(d.values())
        if not vals:
            return []

        if all(isinstance(v, str) for v in vals):
            return vals

        if all(isinstance(v, list) for v in vals):
            return list(chain.from_iterable(vals))

        raise TypeError(
            f"Dictionary values are not uniform: mixture of str and list detected: {d}"
        )

    # server map initialization
    adapters = []
    for adapter in adapter_dirs:
        rank = int(re.search(r"rank-(\d+)", adapter).group(1))
        adapters.append((rank, adapter))
    adapters.sort()
    adapters = [x[1] for x in adapters[:]]

    # Split shuffled list into n parts as evenly as possible
    k, m = divmod(len(adapters), len(servers))
    server_map = {}
    for i, server_name in enumerate(servers):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        for adapter in adapters[start:end]:
            server_map[adapter] = server_name

    start = time.time()
    step_idx = 0
    last_time = requests[0]
    prev_alloc = None
    prev_rank_assigned_instances = None
    prev_server_adapter_sets = None  # list of sets, one per server index
    all_transfer_logs = []  # accumulate per-step transfer logs
    grand_total_transfers = 0
    debug = False
    probability_sum = defaultdict(
        list
    )  # adapter -> [prob of server 1, prob of server 1 + prob of server 2, ...]
    ema_lookback_seconds = 300
    ema_alpha = 0.5
    adapter_window_history = defaultdict(
        lambda: deque()
    )  # adapter -> deque of (window end time, tps), with newest on right

    for req in requests:
        arrival_time = start + req.req_time
        if req.req_time > last_time.req_time // 1 + step:
            alloc_start = time.time()
            demand_tps = {a: 0 for a in adapter_dirs}
            index = requests.index(last_time)
            window_end_time = req.req_time
            while index < len(requests) and requests[index].req_time < req.req_time:
                # rank = int(re.search(r'rank-(\d+)', requests[index].adapter_dir).group(1))
                r = requests[index]
                demand_tps[requests[index].adapter_dir] = (
                    demand_tps.get(r.adapter_dir) + (r.prompt_len + r.output_len) / step
                )
                index += 1

            # adapters that actually received requests in this window (pre-EMA)
            requested_adapters = {adapter for adapter, tps in demand_tps.items() if tps > 0}

            for adapter, raw_tps in demand_tps.items():
                history = adapter_window_history[adapter]
                history.append((window_end_time, raw_tps))

                history_cutoff_time = window_end_time - ema_lookback_seconds
                while history and history[0][0] < history_cutoff_time:
                    history.popleft()

                tps_values = [tps for _, tps in history]
                demand_tps[adapter] = ema_next(tps_values, alpha=ema_alpha)

            adapter_demand = []
            adapter_name_to_tps = {}
            for adapter, tps in demand_tps.items():
                rank = int(re.search(r"rank-(\d+)", adapter).group(1))
                adapter_demand.append((rank, tps, adapter))  # tps here is expected tps
                adapter_name_to_tps[adapter] = tps
            adapter_demand.sort(reverse=True)

            rank_wise_demand = {}
            for rank, tps, adapter in adapter_demand:
                if rank not in rank_wise_demand:
                    rank_wise_demand[rank] = 0
                rank_wise_demand[rank] += tps

            # 7b tp4
            server_tps = {
                8: 5500,
                16: 5400,
                32: 5250,
                64: 5000,
                128: 4500,
            }  # operating point, fn of max rank, ND96asrv 8xA100 80GB

            rank_instance_demand = {}
            for rank, tps in rank_wise_demand.items():
                rank_instance_demand[rank] = tps / server_tps[rank]

            total_instance_demand = sum(rank_instance_demand.values())

            flattened_server_map = flatten_dict_values(server_map)
            servers = sorted(list(set(flattened_server_map)))
            adapter_groups = [[] for _ in servers]
            num_servers = len(servers)
            server_occupied_tps = [0] * num_servers
            server_max_rank = [0] * num_servers

            target_util = total_instance_demand / len(servers)
            assert (
                target_util <= 1
            ), f"Target utilization exceeds 1, need more servers: {target_util}"

            with open("allocation_log.txt", "a") as f:
                f.write("\n\n************************************")
                f.write(
                    f"step {step_idx} @ time {last_time.req_time} to {req.req_time}:"
                )

            # * checking compatibility
            rank_instance_budget = [
                (
                    rank,
                    sum(tps for r, tps, _ in adapter_demand if r == rank)
                    / rank_max_tps,
                )
                for rank, rank_max_tps in server_tps.items()
            ]
            sorted_budgets = sorted(
                rank_instance_budget, key=lambda x: x[1], reverse=True
            )
            assert (
                sum(budget for _, budget in rank_instance_budget) <= num_servers
            ), "Exceeded server budget"
            if debug:
                ic(sorted_budgets, target_util)

            # * rounding
            rounded_budgets = [
                (budget, rank, round(budget / target_util))
                for rank, budget in sorted_budgets
                if round(budget / target_util) > 0
            ]
            if len(rounded_budgets) == 0:
                # Force at least one instance for the largest rank budget
                largest_rank, largest_budget = max(sorted_budgets, key=lambda x: x[1])
                rounded_budgets = [(largest_budget, largest_rank, 1)]

            rounded_budgets.sort(reverse=True, key=lambda x: (x[0] / x[2], x[1]))
            if debug:
                ic(rounded_budgets)

            zero_budgets = [
                (budget, rank, round(budget / target_util))
                for rank, budget in sorted_budgets
                if round(budget / target_util) == 0
            ]

            sum_rounded_off_budgets = sum(budget for _, _, budget in rounded_budgets)
            while sum_rounded_off_budgets < num_servers:
                # print("Increasing instances")
                first = rounded_budgets[0]
                rounded_budgets = [
                    (first[0], first[1], first[2] + 1)
                ] + rounded_budgets[1:].copy()
                rounded_budgets.sort(reverse=True, key=lambda x: (x[0] / x[2], x[1]))
                sum_rounded_off_budgets = sum(
                    budget for _, _, budget in rounded_budgets
                )
            rounded_budgets.sort(key=lambda x: x[1])
            sum_rounded_off_budgets = sum(budget for _, _, budget in rounded_budgets)
            while sum_rounded_off_budgets > num_servers:
                # print("Decreasing instances")
                first = rounded_budgets[0]
                assert first[2] > 0, "Cannot reduce instances further"
                rounded_budgets = [
                    (first[0], first[1], first[2] - 1)
                ] + rounded_budgets[1:].copy()
                rounded_budgets.sort(key=lambda x: x[1])
                sum_rounded_off_budgets = sum(
                    budget for _, _, budget in rounded_budgets
                )
            if debug:
                ic(rounded_budgets)

            # * balanced allocation within assigned instances
            ranks_with_assigned_instances = [(x[1], x[2]) for x in rounded_budgets]
            rank_assigned_instances = {}
            ranks_with_zero_instances = [x[1] for x in zero_budgets]
            if debug:
                ic("assigned", ranks_with_assigned_instances)
                ic("zero", ranks_with_zero_instances)
            last_used_server = 0
            server_util = [0] * num_servers
            leftovers = []
            for rank in ranks_with_zero_instances:
                leftovers.extend(
                    [
                        (idx, adapter, 1.0)
                        for idx, adapter in enumerate(adapter_demand)
                        if adapter[0] == rank
                    ]
                )
            for rank, budget in ranks_with_assigned_instances:
                # assign adapters of rank to num_instances greedily
                # maybe sort in descending tps order and assign fractionally from the left

                # get all adapters of this rank
                adapters_of_rank = [
                    (idx, adapter)
                    for idx, adapter in enumerate(adapter_demand)
                    if adapter[0] == rank
                ]
                adapters_of_rank.sort(
                    reverse=True, key=lambda x: x[1][1]
                )  # sort by tps descending
                if debug:
                    ic(adapters_of_rank)
                servers_used = 0
                if budget > 0:
                    rank_assigned_instances[rank] = list(
                        range(last_used_server, last_used_server + budget)
                    )
                for adapter_idx, adapter in adapters_of_rank:
                    tps, adapter_name = adapter[1], adapter[2]
                    expected_util = tps / server_tps[rank]
                    _expected_util = expected_util
                    while expected_util > 1e-4 and servers_used < budget:
                        # assign as much as possible to this server
                        if servers_used >= budget:
                            ic(servers_used, budget, adapter_idx, adapter)
                            raise Exception(
                                f"Ran out of servers: used {servers_used}, budget {budget} at adapter {adapter} at index {adapter_idx}"
                            )
                        server_idx = last_used_server + servers_used
                        # assign to this server
                        max_addable_util = min(
                            target_util - server_util[server_idx], expected_util
                        )
                        adapter_groups[server_idx].append(
                            [adapter_name, max_addable_util / _expected_util]
                        )
                        server_occupied_tps[server_idx] += (
                            max_addable_util * server_tps[rank]
                        )

                        # adapters_placed[adapter_idx] = True
                        server_max_rank[server_idx] = max(
                            server_max_rank[server_idx], rank
                        )
                        server_util[server_idx] += max_addable_util
                        if server_util[server_idx] >= target_util:
                            servers_used += 1
                        expected_util -= max_addable_util
                    if expected_util > 1e-4:
                        leftovers.append(
                            (adapter_idx, adapter, expected_util / _expected_util)
                        )
                last_used_server += budget

            if debug:
                for adapter_group in adapter_groups:
                    ic(adapter_group)
                ic(leftovers)

            # * leftovers
            space_left = (target_util * num_servers) - sum(server_util)
            demand_left = 0
            for _, adapter_tuple, util_fraction in leftovers:
                demand_left += (adapter_tuple[1] * util_fraction) / server_tps[
                    adapter_tuple[0]
                ]
            if demand_left - space_left >= 1e-3:
                with open("allocation_log.txt", "a") as f:
                    f.write(
                        f"[warn] Leftover demand ({demand_left:.6f}) exceeds space ({space_left:.6f}). Scaling leftovers proportionally.\n"
                    )
                if demand_left > 0 and space_left > 0:
                    rescaled_leftovers = True
                    scale = space_left / demand_left
                    new_leftovers = []
                    for adapter_idx, adapter_tuple, util_fraction in leftovers:
                        new_leftovers.append(
                            (adapter_idx, adapter_tuple, util_fraction * scale)
                        )
                    leftovers = new_leftovers
                else:
                    # No capacity left; drop leftover demands (will be completed in next window)
                    leftovers = []
                    with open("allocation_log.txt", "a") as f:
                        f.write(
                            "[warn] Dropping leftovers due to zero capacity; will retry next window.\n"
                        )

            if debug:
                ic("before leftovers")
                for server_idx in range(num_servers):
                    ic(server_idx, server_util[server_idx], target_util)
                ic("space left", space_left)
                ic("demand left", demand_left)

            leftovers.sort(
                reverse=True, key=lambda x: (x[1][1])
            )  # sort by tps descending
            round_robin_server_idx = 0

            for _, adapter_tuple, util_fraction in leftovers:
                adapter_rank, adapter_demand_tps, adapter_name = adapter_tuple
                _expected_util = adapter_demand_tps / server_tps[adapter_rank]
                adapter_demand_tps *= util_fraction
                expected_util = adapter_demand_tps / server_tps[adapter_rank]
                # go through the servers with higher max rank and fit the fractional demand until target utilization
                server_idx = 0
                allocated_adapter = False
                if expected_util < 1e-3:
                    adapter_groups[round_robin_server_idx].append(
                        [adapter_name, util_fraction]
                    )
                    server_max_rank[round_robin_server_idx] = max(
                        server_max_rank[round_robin_server_idx], adapter_rank
                    )
                    server_occupied_tps[round_robin_server_idx] += (
                        util_fraction * server_tps[adapter_rank]
                    )
                    server_util[round_robin_server_idx] += util_fraction
                    allocated_adapter = True
                    round_robin_server_idx = (round_robin_server_idx + 1) % num_servers
                    continue

                while (
                    expected_util > 1e-3
                    and server_idx < num_servers
                    and not allocated_adapter
                ):
                    if (
                        server_max_rank[server_idx] >= adapter_rank
                        and server_util[server_idx] < target_util
                    ):
                        max_addable_util = min(
                            target_util - server_util[server_idx], expected_util
                        )
                        adapter_groups[server_idx].append(
                            [adapter_name, max_addable_util / _expected_util]
                        )
                        server_occupied_tps[server_idx] += (
                            max_addable_util * server_tps[adapter_rank]
                        )
                        server_max_rank[server_idx] = max(
                            server_max_rank[server_idx], adapter_rank
                        )
                        server_util[server_idx] += max_addable_util
                        expected_util -= max_addable_util
                    server_idx += 1

                if expected_util == 0:
                    allocated_adapter = True

                if not allocated_adapter:
                    server_idx = 0
                    # we could not find a server with rank >= this adapters rank
                    # need to colocate with a lower rank
                    while expected_util > 1e-3 and server_idx < num_servers:
                        if debug:
                            ic(
                                server_idx,
                                server_util[server_idx],
                                target_util,
                                expected_util,
                            )
                        if server_util[server_idx] < target_util:
                            max_addable_util = min(
                                target_util - server_util[server_idx], expected_util
                            )
                            adapter_groups[server_idx].append(
                                [adapter_name, max_addable_util / _expected_util]
                            )
                            server_occupied_tps[server_idx] += (
                                max_addable_util * server_tps[adapter_rank]
                            )
                            server_max_rank[server_idx] = max(
                                server_max_rank[server_idx], adapter_rank
                            )
                            server_util[server_idx] += max_addable_util
                            expected_util -= max_addable_util
                        server_idx += 1

                if expected_util > 1e-3:
                    try:
                        for i in range(num_servers):
                            if server_util[i] + expected_util <= 1:
                                adapter_groups[i].append(
                                    [adapter_name, expected_util / _expected_util]
                                )
                                server_occupied_tps[i] += (
                                    expected_util * server_tps[adapter_rank]
                                )
                                server_max_rank[i] = max(
                                    server_max_rank[i], adapter_rank
                                )
                                server_util[i] += expected_util
                                expected_util = 0
                                break
                    except Exception as e:
                        raise Exception(
                            f"Could not allocate adapter {adapter_name} with rank {adapter_rank} and tps {adapter_demand_tps}, leftover util {expected_util}"
                        )

            with open("allocation_log.txt", "a") as f:
                f.write(f"\n{last_time} Adapter groups:\n")
                for i, group in enumerate(adapter_groups):
                    f.write(
                        f"  Server {servers[i]}: {[(adapter, used_util, adapter_name_to_tps[adapter] * used_util) for adapter, used_util in group]}\n"
                    )
                    f.write(
                        f"  Server {servers[i]} total tps: {server_occupied_tps[i]}\n"
                    )
                    f.write(
                        f"  Server {servers[i]} max tps: {server_tps.get(server_max_rank[i], 0)}\n"
                    )
                    f.write(f"  Server {servers[i]} util: {server_util[i]}\n")
                    f.write(f"  Server {servers[i]} max rank: {server_max_rank[i]}\n")
                f.write("************************************\n\n")

            # * compute adapter transfers (adapters that must be loaded because a request arrived but the adapter wasn't on that server before)
            curr_server_adapter_sets = [
                {adapter for adapter, _ in adapter_groups[i]}
                for i in range(num_servers)
            ]
            transfer_log = {}  # server_url -> {transferred_adapters, num_transferred}
            total_transfers = 0
            with open("allocation_log.txt", "a") as f:
                f.write(f"--- Adapter transfers (step {step_idx}) ---\n")
                for i, server in enumerate(servers):
                    if prev_server_adapter_sets is not None:
                        new_adapters = curr_server_adapter_sets[i] - prev_server_adapter_sets[i]
                    else:
                        # step 0: all adapters on the server are new
                        new_adapters = curr_server_adapter_sets[i]
                    # only count as a transfer if the adapter actually got a request in this window
                    transferred = sorted(new_adapters & requested_adapters)
                    transfer_log[server] = {
                        "transferred_adapters": transferred,
                        "num_transferred": len(transferred),
                    }
                    total_transfers += len(transferred)
                    if transferred:
                        f.write(
                            f"  Server {server}: {len(transferred)} transfers: {transferred}\n"
                        )
                transfer_log["total_transfers"] = total_transfers
                f.write(f"  Total transfers across all servers: {total_transfers}\n")
                f.write("--- End adapter transfers ---\n\n")

            # ensure_all_placed(adapter_groups)
            print(
                f"All adapters placed successfully for step {step_idx} from time {last_time.req_time} to {req.req_time}"
            )

            # * compare with last iteration
            server_rename_map = None
            if prev_alloc is not None and prev_rank_assigned_instances is not None:
                adapter_to_tps = {adapter: tps for _, tps, adapter in adapter_demand}
                server_rename_map = compare_with_prev_alloc(
                    adapter_groups=adapter_groups,
                    prev_adapter_groups=prev_alloc,
                    assigned_instances_per_rank=rank_assigned_instances,
                    prev_assigned_instances_per_rank=prev_rank_assigned_instances,
                    adapter_to_tps=adapter_to_tps,
                )
                if debug:
                    with open("allocation_log.txt", "a") as f:
                        f.write(
                            f"Prev rank assigned instances (step {step_idx - 1}): {prev_rank_assigned_instances}\n"
                        )
                        f.write(
                            f"Curr rank assigned instances (step {step_idx}): {rank_assigned_instances}\n"
                        )
                if server_rename_map:
                    with open("allocation_log.txt", "a") as f:
                        f.write(f"Server renames detected: {server_rename_map}\n")

                    if debug:
                        print("Server renames detected:", server_rename_map)
                        print(
                            f"Prev rank assigned instances (step {step_idx - 1}):",
                            prev_rank_assigned_instances,
                        )
                        print(
                            f"Curr rank assigned instances (step {step_idx}):",
                            rank_assigned_instances,
                        )

            server_map = defaultdict(list)  # adapter -> [server1, server2, ...]
            probability_sum = defaultdict(
                list
            )  # adapter -> [prob of server 1, prob of server 1 + prob of server 2, ...]
            for i, server in enumerate(servers):
                for adapter, util in adapter_groups[i]:
                    if not server_rename_map or server not in server_rename_map.keys():
                        server_map[adapter].append(server)
                        probability_sum[adapter].append(
                            probability_sum[adapter][-1] + util
                            if probability_sum[adapter]
                            else util
                        )
                    else:
                        server_map[adapter].append(server_rename_map[server])
            if debug:
                print(server_map)
                print(probability_sum)

            prev_alloc = adapter_groups.copy()
            prev_rank_assigned_instances = rank_assigned_instances.copy()
            prev_server_adapter_sets = curr_server_adapter_sets
            last_time = req
            alloc_end = time.time()

            with open("allocation_log.txt", "a") as f:
                f.write(
                    f"Allocation computation time: {alloc_end - alloc_start:.4f} s\n"
                )

            with open(
                f"{server_map_folder}/system/server_map_step_{step_idx}.json", "w"
            ) as f:
                json.dump(server_map, f, indent=4)
            with open(
                f"{server_map_folder}/system/probability_sum_step_{step_idx}.json", "w"
            ) as f:
                json.dump(probability_sum, f, indent=4)
            # with open(
            #     f"{server_map_folder}/system/transfer_log_step_{step_idx}.json", "w"
            # ) as f:
            #     json.dump(transfer_log, f, indent=4)

            all_transfer_logs.append({"step": step_idx, **transfer_log})
            grand_total_transfers += total_transfers

            step_idx += 1

    random.seed(42)
    shuffled = adapter_dirs.copy()
    random.shuffle(shuffled)

    # Split shuffled list into n parts as evenly as possible
    k, m = divmod(len(shuffled), len(servers))
    # server_map = {adapter:server_name for adapter in shuffled[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i, server_name in enumerate(servers)}
    baseline_server_map = {}
    for i, server_name in enumerate(servers):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        for adapter in shuffled[start:end]:
            baseline_server_map[adapter] = server_name
    with open(f"{server_map_folder}/baseline/server_map.json", "w") as f:
        json.dump(baseline_server_map, f, indent=4)

    adapters = []
    for adapter in adapter_dirs:
        rank = int(re.search(r"rank-(\d+)", adapter).group(1))
        adapters.append((rank, adapter))
    adapters.sort()
    adapters = [x[1] for x in adapters[:]]

    # Split shuffled list into n parts as evenly as possible
    k, m = divmod(len(adapters), len(servers))
    contiguous_server_map = {}
    for i, server_name in enumerate(servers):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        for adapter in adapters[start:end]:
            contiguous_server_map[adapter] = server_name

    with open(f"{server_map_folder}/contiguous/server_map.json", "w") as f:
        json.dump(contiguous_server_map, f, indent=4)

    # write combined transfer log
    combined_transfer_log = {
        "steps": all_transfer_logs,
        "grand_total_transfers": grand_total_transfers,
    }
    with open(f"{server_map_folder}/system/transfer_log.json", "w") as f:
        json.dump(combined_transfer_log, f, indent=4)
    print(f"Total transfers over all servers and all timesteps: {grand_total_transfers}")
