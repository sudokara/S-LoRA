import time
import torch
import numpy as np
import torch.nn as nn
torch.manual_seed(0)

import zmq
import os
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)

from multiprocessing import shared_memory
import punica

class Lora_GPU():
    def __init__(self, configJson, shared=False):

        self.device = torch.cuda.current_device()

        self.n_layers = configJson["n_layers"]
        self.batch_size = configJson["batch_size"]
        logging.critical("Self Batch size: {}".format(self.batch_size))

        self.lora_rank = configJson["lora_rank"]
        self.in_size = configJson["hidden_dim"]
        self.output_size = configJson["hidden_dim"] // configJson["tp_degree"]
        self.tp_degree = configJson["tp_degree"]
        self.max_prefill_batch_size = configJson["max_prefill_batch_size"]

        logging.critical("LoRA GPU Rank: {}".format(self.lora_rank))

        # normal LoRA weights
        self.q_down = torch.zeros( (self.batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda:{self.device}", dtype=torch.float16).transpose(-1, -2).contiguous()
        self.q_up = torch.zeros( (self.batch_size, self.n_layers, self.lora_rank, self.output_size), device=f"cuda:{self.device}", dtype=torch.float16).transpose(-1, -2).contiguous()

        self.k_down = torch.zeros( (self.batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda:{self.device}", dtype=torch.float16).transpose(-1, -2).contiguous()
        self.k_up = torch.zeros( (self.batch_size, self.n_layers, self.lora_rank, self.output_size), device=f"cuda:{self.device}", dtype=torch.float16).transpose(-1, -2).contiguous()

        self.v_down = torch.zeros( (self.batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda:{self.device}", dtype=torch.float16).transpose(-1, -2).contiguous()
        self.v_up = torch.zeros( (self.batch_size, self.n_layers, self.lora_rank, self.output_size), device=f"cuda:{self.device}", dtype=torch.float16).transpose(-1, -2).contiguous()

        # for prefill
        if not shared:
            self.q_down_prefill = None
            self.q_up_prefill = None

            self.k_down_prefill = None
            self.k_up_prefill = None

            self.v_down_prefill = None
            self.v_up_prefill = None

            if "aaas-cached" in configJson["mode"] or "aaas-gpu" in configJson["mode"]:
                self.q_down_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
                self.q_up_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

                self.k_down_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
                self.k_up_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

                self.v_down_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
                self.v_up_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

        else:
            # for shared. We control one prefill batch is 4 now 
            self.q_down_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.in_size, self.lora_rank), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
            self.q_up_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

            self.k_down_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.in_size, self.lora_rank), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001 
            self.k_up_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

            self.v_down_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.in_size, self.lora_rank), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
            self.v_up_prefill = torch.randn( (self.max_prefill_batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001


        # dummy 
        self.dummy_lora_down_cpu = [ 0.001*torch.randn( (1, self.in_size, self.lora_rank), device="cpu", dtype=torch.float16) for _ in range(self.n_layers) ]
        self.dummy_lora_up_cpu = [ 0.001*torch.randn( (1, self.lora_rank, self.output_size), device="cpu", dtype=torch.float16) for _ in range(self.n_layers) ]

        # sharing part for swapping
        # self.num_swapper = int(os.getenv("num_swapper"))

        # hardcoded number, does not affect anything
        self.num_swapper = 1
        self.socket_pull_list = []
        self.poller_list = []
        self.socket_push_list = []

        context = zmq.Context()
        for idx in range( self.num_swapper ):
            # context_pull = zmq.Context()
            socket_pull = context.socket(zmq.PULL)
            socket_pull.bind('tcp://*:{}'.format(5551 + idx*2 + 100 * self.device))
            #Creating poller
            poller = zmq.Poller()
            poller.register(socket_pull, zmq.POLLIN)

            #Creating a context
            # context_push = zmq.Context()
            #creating and connceting to a socket.
            socket_push = context.socket(zmq.PUSH)
            socket_push.connect('tcp://localhost:{}'.format(5550 + idx*2 + 100 * self.device))

            self.socket_pull_list.append( socket_pull )
            self.poller_list.append( poller )
            self.socket_push_list.append( socket_push )
        for idx in range( self.num_swapper ):
            print( self.socket_pull_list[idx] ) 
            print( self.socket_push_list[idx] )

        if shared:
            self.share_param( self.q_up_prefill, num_swapper=self.num_swapper)
            self.share_param( self.k_up_prefill, num_swapper=self.num_swapper)
            self.share_param( self.v_up_prefill, num_swapper=self.num_swapper )
            self.share_param( self.q_down_prefill, num_swapper=self.num_swapper)
            self.share_param( self.k_down_prefill, num_swapper=self.num_swapper)
            self.share_param( self.v_down_prefill, num_swapper=self.num_swapper)
            
            self.share_progress()

        self.select_inds = [ torch.randint(0, self.batch_size, (i+1,), device="cuda")  for i in range(self.batch_size) ]

    def invoke_one_layer_q(self, x, layer_id, prefill=False):
        bsz = x.shape[0]
        cur_inds = self.select_inds[bsz-1]
        if prefill:
            try:
                y = torch.bmm(torch.bmm(x, self.q_down_prefill[:bsz, layer_id].transpose(-1, -2).contiguous(),), \
                                self.q_up_prefill[:bsz, layer_id].transpose(-1, -2).contiguous())
            except Exception as e: 
                print("=====", e)            
                logging.critical("x shape: {}. self.q_down_prefill shape: {}. self.q_up_prefill shape: {}".format( \
                x.shape, self.q_down_prefill[:bsz, layer_id].transpose(-1, -2).contiguous().shape, \
                self.q_up_prefill[:bsz, layer_id].transpose(-1, -2).contiguous().shape ))
                exit(0)
        else:
            # y = torch.bmm(torch.bmm(x, self.q_down[layer_id, :bsz],), self.q_up[layer_id, :bsz])
            y = torch.zeros(bsz, self.output_size, device="cuda", dtype=torch.float16)
            punica.ops.add_lora_bgmv(y, x, self.q_down, self.q_up, cur_inds, layer_id, 1)
        return y

    def invoke_one_layer_k(self, x, layer_id, prefill=False):
        bsz = x.shape[0]
        cur_inds = self.select_inds[bsz-1]
        if prefill:
            y = torch.bmm(torch.bmm(x, self.k_down_prefill[:bsz, layer_id].transpose(-1, -2).contiguous(),), \
                            self.k_up_prefill[:bsz, layer_id].transpose(-1, -2).contiguous())
        else:
            # y = torch.bmm(torch.bmm(x, self.k_down[layer_id, :bsz],), self.k_up[layer_id, :bsz])
            y = torch.zeros(bsz, self.output_size, device="cuda", dtype=torch.float16)
            punica.ops.add_lora_bgmv(y, x, self.k_down, self.k_up, cur_inds, layer_id, 1)
        return y

    def invoke_one_layer_v(self, x, layer_id, prefill=False):
        bsz = x.shape[0]
        cur_inds = self.select_inds[bsz-1]
        if prefill:
            y = torch.bmm(torch.bmm(x, self.v_down_prefill[:bsz, layer_id].transpose(-1, -2).contiguous(),), \
                            self.v_up_prefill[:bsz, layer_id].transpose(-1, -2).contiguous())
        else:
            # y = torch.bmm(torch.bmm(x, self.v_down[layer_id, :bsz],), self.v_up[layer_id, :bsz])
            y = torch.zeros(bsz, self.output_size, device="cuda", dtype=torch.float16)
            punica.ops.add_lora_bgmv(y, x, self.v_down, self.v_up, cur_inds, layer_id, 1)
        return y

    def load_lora_prefill(self, batch_size):
        for module_name in ["q", "k", "v"]:
            prefill_lora_down = torch.empty( (0, self.n_layers, self.in_size, self.lora_rank), device="cuda", dtype=torch.float16)
            prefill_lora_up = torch.empty( (0, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16)
            for batch_id in range(batch_size):
                batch_weight_down = torch.empty( (0, self.in_size, self.lora_rank), device="cuda", dtype=torch.float16)
                batch_weight_up = torch.empty( (0, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16)
                for layer_id in range(self.n_layers):
                    batch_weight_down = torch.cat(  (batch_weight_down, self.dummy_lora_down_cpu[layer_id].cuda()), dim=0 )
                    batch_weight_up   = torch.cat(  (batch_weight_up, self.dummy_lora_up_cpu[layer_id].cuda()), dim=0 )
                
                prefill_lora_down = torch.cat( (prefill_lora_down, batch_weight_down.unsqueeze(0)), dim=0)
                prefill_lora_up   = torch.cat( (prefill_lora_up, batch_weight_up.unsqueeze(0)), dim=0)

            if module_name == "q":
                self.q_down_prefill = prefill_lora_down.clone().detach().transpose(-1, -2).contiguous()
                self.q_up_prefill = prefill_lora_up.clone().detach().transpose(-1, -2).contiguous()
            if module_name == "k":
                self.k_down_prefill = prefill_lora_down.clone().detach().transpose(-1, -2).contiguous()
                self.k_up_prefill = prefill_lora_up.clone().detach().transpose(-1, -2).contiguous()
            if module_name == "v":
                self.v_down_prefill = prefill_lora_down.clone().detach().transpose(-1, -2).contiguous()
                self.v_up_prefill = prefill_lora_up.clone().detach().transpose(-1, -2).contiguous()
            torch.cuda.empty_cache()
        
    def prepare_lora_prefill(self, batch_size):
        logging.critical("Init GPU LoRAs for prefill")

        self.q_down_prefill = torch.randn( (batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
        self.q_up_prefill = torch.randn( (batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

        self.k_down_prefill = torch.randn( (batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
        self.k_up_prefill = torch.randn( (batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

        self.v_down_prefill = torch.randn( (batch_size, self.n_layers, self.in_size, self.lora_rank), device=f"cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001
        self.v_up_prefill = torch.randn( (batch_size, self.n_layers, self.lora_rank, self.output_size), device="cuda", dtype=torch.float16).transpose(-1, -2).contiguous() * 0.001

    def merge_lora_prefill(self, bsz):
        logging.critical("Merge prefill GPU LoRAs for decoding")

        # re_ind = np.random.randint( max(31-bsz,0), size=1)[0]
        re_inds = np.random.randint( 0, self.batch_size, size=bsz)
        # logging.critical("re_inds: {}".format(re_inds))
        logging.critical("self.q_down shape: {}; self.q_down_prefill shape: {}".format(self.q_down.shape, self.q_down_prefill.shape))
        for idx, re_ind in enumerate(re_inds):
            self.q_down[ re_ind ][:] = self.q_down_prefill[ idx ][:]
            self.q_up[ re_ind ][:] = self.q_up_prefill[ idx ][:]

            self.k_down[ re_ind ][:] = self.k_down_prefill[ idx ][:]
            self.k_up[ re_ind ][:] = self.k_up_prefill[ idx ][:]

            self.v_down[ re_ind ][:] = self.v_down_prefill[ idx ][:]
            self.v_up[ re_ind ][:] = self.v_up_prefill[ idx ][:]

            # self.q_down[:,re_ind:(re_ind+bsz)] = self.q_down_prefill[:,:bsz]
            # self.q_up[:, re_ind:(re_ind+bsz)] = self.q_up_prefill[:,:bsz]

            # self.k_down[:, re_ind:(re_ind+bsz)] = self.k_down_prefill[:,:bsz]
            # self.k_up[:, re_ind:(re_ind+bsz)] = self.k_up_prefill[:,:bsz]

            # self.v_down[:, re_ind:(re_ind+bsz)] = self.v_down_prefill[:,:bsz]
            # self.v_up[:, re_ind:(re_ind+bsz)] = self.v_up_prefill[:,:bsz]


    def remove_lora(self, rm_ids: list, running_bsz: int):

        if running_bsz > 0:
            rm_ids = np.random.randint(0, running_bsz, size=len(rm_ids))
            for rm_id in rm_ids:
                logging.critical("rm id: {}".format(rm_id))
                self.q_down[:, rm_id:-1] = self.q_down[:, (rm_id+1):]
                self.q_up[:, rm_id:-1] = self.q_up[:, (rm_id+1):]

                self.k_down[:, rm_id:-1] = self.k_down[:, (rm_id+1):]
                self.k_up[:, rm_id:-1] = self.k_up[:, (rm_id+1):]

                self.v_down[:, rm_id:-1] = self.v_down[:, (rm_id+1):]
                self.v_up[:, rm_id:-1] = self.v_up[:, (rm_id+1):]

        # logging.critical("After rm op, self.q_down shape: {}. rm_ids: {}".format(self.q_down.shape, rm_ids))

    def share_param(self, param_tensor, num_swapper):
        for idx in range(num_swapper):
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.connect('tcp://0.0.0.0:{}'.format( 6000 + idx + 100 * self.device))

            storage = param_tensor.storage()
            (storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
            ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = storage._share_cuda_()
            sock.send_pyobj({
                "dtype": param_tensor.dtype,
                "tensor_size": param_tensor.shape,
                "tensor_stride": param_tensor.stride(),
                "tensor_offset": param_tensor.storage_offset(), # !Not sure about this one.
                "storage_cls": type(storage),
                "storage_device": storage_device,
                "storage_handle": storage_handle,
                "storage_size_bytes": storage_size_bytes,
                "storage_offset_bytes": storage_offset_bytes,
                "requires_grad": False,
                "ref_counter_handle": ref_counter_handle,
                "ref_counter_offset": ref_counter_offset,
                "event_handle": event_handle,
                "event_sync_required": event_sync_required,
            })
            sock.recv_string()


    def swap(self, swapper_id):
        # self.socket_push.send_string( "{}".format(swap_out_id) )
        # self.process_signal[swapper_id][0] = 0
        self.progress[self.device * self.num_swapper + swapper_id] = 1
        self.socket_push_list[swapper_id].send_string( "{}".format(swapper_id) )

    def share_progress(self):
        self.process_shm = []
        self.process_signal = []

        # link to one large signal
        shm_name_folder = os.getenv("shm_name_folder")

        shm_name_files = [ item for item in os.listdir(shm_name_folder) if ".shm" in item and "progress" in item ]
        assert len(shm_name_files) == 1, "{}".format(len(shm_name_files))
        shm_filename = shm_name_files[0]
        shm_name = shm_filename[:-4].split("-")[1]
        self.shm_progress = shared_memory.SharedMemory(name=shm_name)
        self.np_progress = np.ndarray(self.tp_degree * self.num_swapper, dtype=np.int8, buffer=self.shm_progress.buf)
        self.progress = torch.from_numpy(self.np_progress)

    def use_gpu(self):
        return torch.sum(self.progress)

    def use_gpu_pipeline(self, layer_id):
        if torch.sum(self.progress) == 0:
            return True
        # hard coding here
        if 0 <= layer_id < 8:
            if self.progress[0] == 0:
                return True
            else:
                return False
        if 8 <= layer_id < 16:
            if self.progress[0] >= 2:
                return True
            else:
                return False
        if 16 <= layer_id < 24:
            if self.progress[0] >= 3:
                return True
            else:
                return False
        if 24 <= layer_id < 32:
            if self.progress[0] >= 4:
                return True
            else:
                return False
        
        

if __name__ == "__main__":
    configJson = {}
    configJson["n_layers"] = 32
    configJson["batch_size"] = 32

    configJson["lora_rank"] = 64
    configJson["hidden_dim"] = 4096
    configJson["tp_degree"] = 1

    gpu_lora = Lora_GPU(configJson, shared=True)
    gpu_lora.swap(0)
    # gpu_lora.swap(1) 
    while gpu_lora.process_signal[0][0] != 64:
        pass
    print("Swapped")
    time.sleep(1000)