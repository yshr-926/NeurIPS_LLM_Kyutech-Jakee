import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lit_gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.tokenizer import Tokenizer
from lit_gpt.utils import (
    check_valid_checkpoint_dir,
    chunked_cross_entropy,
    get_default_supported_precision,
    load_checkpoint,
    num_parameters,
    quantization,
    step_csv_logger,
)
from script.prepare_alpaca import generate_prompt
# from scripts import generate_prompt
from my_utils.utils import get_optimizer, get_bnb_optimizer
# from torch.optim.swa_utils import AveragedModel
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
from optimizers import *
eval_interval = 50
save_interval = 100
eval_iters = 100
eval_max_new_tokens = 100
devices = 1
# change this value to force a maximum sequence length
override_max_seq_length = 2048

# Hyperparameters
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False
# warmup_steps = 10
eta_min = 0.0
hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}

# limit an caching allocator to allocated memory on a CUDA device
torch.cuda.set_per_process_memory_fraction(0.49, 0)

def setup(
    data_dir: Path = Path("data/alpaca"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/lora/alpaca"),
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq"]] = None,
    optim_name: str = "AdamW",
    max_iters: int = 50000,
    log_interval: int = 200,
    batch_size: int = 128,
    micro_batch_size: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 10,
    lr_type: str = "CosineAnnealingLR"
):
    precision = precision or get_default_supported_precision(training=True)

    fabric_devices = devices
    if fabric_devices > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. "
                "Please set devices=1 when using the --quantization flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
        )
    else:
        strategy = "auto"

    logger = step_csv_logger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
    fabric = L.Fabric(devices=fabric_devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, data_dir, checkpoint_dir, out_dir, max_iters, optim_name, log_interval, batch_size, micro_batch_size, learning_rate, weight_decay, lr_type, warmup_steps, quantize)


def main(
        fabric: L.Fabric, 
        data_dir: Path, 
        checkpoint_dir: Path, 
        out_dir: Path,
        max_iters,
        optim_name,
        log_interval,
        batch_size,
        micro_batch_size,
        learning_rate,
        weight_decay,
        lr_type,
        warmup_steps,
        quantize: Optional[str] = None,
        ):
    check_valid_checkpoint_dir(checkpoint_dir)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data = torch.load(data_dir / "train.pt")
    val_data = torch.load(data_dir / "test.pt")

    if not any((lora_query, lora_key, lora_value, lora_projection, lora_mlp, lora_head)):
        fabric.print("Warning: all LoRA layers are disabled!")
    config = Config.from_name(
        name=checkpoint_dir.name,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        to_query=lora_query,
        to_key=lora_key,
        to_value=lora_value,
        to_projection=lora_projection,
        to_mlp=lora_mlp,
        to_head=lora_head,
    )
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
    with fabric.init_module(empty_init=(devices > 1)), quantization(quantize):
        model = GPT(config)
    mark_only_lora_as_trainable(model)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")

    if quantize:
        # for quantization, need to load before moving to device
        load_checkpoint(fabric, model, checkpoint_path, strict=False)

    model = fabric.setup_module(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    averaged_params = deepcopy(trainable_params)
    if quantize and quantize.startswith("bnb."):
        import bitsandbytes as bnb

        optimizer = get_bnb_optimizer(optim_name, trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = get_optimizer(optim_name, trainable_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=max_iters//batch_size, eta_min=eta_min)
    if lr_type == "CosineAnnealingLR":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_steps, max_epochs=max_iters//batch_size, warmup_start_lr=0.00001, eta_min=eta_min)
        # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=max_iters//batch_size, eta_min=eta_min)  
    else:
        scheduler = "Fix"
    
    if not quantize:
        # strict=False because missing keys due to LoRA weights not contained in state dict
        load_checkpoint(fabric, model, checkpoint_path, strict=False)

    fabric.seed_everything(1337 + fabric.global_rank)

    iter_checkpoint_path = f"{max_iters}_{learning_rate}_{weight_decay}_{lr_type}"
    train_time = time.perf_counter()
    train(fabric, model, optimizer, train_data, val_data, checkpoint_dir, out_dir, speed_monitor, max_iters, log_interval, batch_size, micro_batch_size, learning_rate, trainable_params, averaged_params, scheduler, warmup_steps, iter_checkpoint_path)
    total_time = time.perf_counter()-train_time
    fabric.print(f'Total {total_time//3600:.0f}:{total_time%3600//60:02.0f}:{total_time%3600%60:02.0f}')
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / f"lit_model_lora_{lr_type}_{optim_name}_{max_iters}_{batch_size}_{micro_batch_size}_{learning_rate}_{weight_decay}.pth"
    save_lora_checkpoint(fabric, model, save_path)

    for i, p_averaged in enumerate(averaged_params):
        trainable_params[i] = p_averaged
    save_ave_path = out_dir / f"lit_model_ave_{lr_type}_{optim_name}_{max_iters}_{batch_size}_{micro_batch_size}_{learning_rate}_{weight_decay}.pth"
    save_lora_checkpoint(fabric, model, save_ave_path)

def train(
    fabric: L.Fabric,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_data: List[Dict],
    val_data: List[Dict],
    checkpoint_dir: Path,
    out_dir: Path,
    speed_monitor: SpeedMonitor,
    max_iters,
    log_interval,
    batch_size,
    micro_batch_size,
    learning_rate,
    trainable_params,
    averaged_params,
    scheduler,
    warmup_steps,
    iter_checkpoint_path
) -> None:
    tokenizer = Tokenizer(checkpoint_dir)
    max_seq_length, longest_seq_length, longest_seq_ix = get_max_seq_length(train_data)
    model.max_seq_length = max_seq_length

    validate(fabric, model, val_data, tokenizer, micro_batch_size, longest_seq_length)  # sanity check
    gradient_accumulation_iters = batch_size // micro_batch_size
    assert gradient_accumulation_iters > 0
    with torch.device("meta"):
        meta_model = GPT(model.config)
        mark_only_lora_as_trainable(meta_model)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        # this assumes that all samples have a fixed length equal to the longest sequence length
        # which is most likely false during finetuning
        x = torch.randint(0, 1, (micro_batch_size, longest_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    step_count = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    start_averaged = max_iters*0.8
    n_averaged = 0
    best_val = {"iter": 0, "val_loss": 1000}

    for iter_num in range(max_iters):
        if scheduler == "Fix":
            if step_count <= warmup_steps:
                # linear warmup
                lr = learning_rate * step_count / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids, targets = get_batch(
            fabric, train_data, micro_batch_size, longest_seq_length, longest_seq_ix if iter_num == 0 else None
        )

        is_accumulating = (iter_num + 1) % gradient_accumulation_iters != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            logits[-1] = logits[-1][..., :-1, :]
            loss = chunked_cross_entropy(logits, targets[..., 1:])
            fabric.backward(loss / gradient_accumulation_iters)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
            
            if not scheduler == "Fix":
                scheduler.step()

            # update Averaged model
            if iter_num >= start_averaged:
                update_parameters(averaged_params, trainable_params, n_averaged)

        t1 = time.perf_counter()
        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (iter_num + 1) * micro_batch_size,
            t1 - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
        )
        if iter_num % log_interval == 0:
            fabric.print(
                f"lr: {optimizer.param_groups[0]['lr']:.6f} iter {iter_num} step {step_count}: loss {loss.item():.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and step_count % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_data, tokenizer, micro_batch_size, longest_seq_length)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            if best_val['val_loss'] > val_loss.item():
                best_val['iter'] = iter_num
                best_val['val_loss'] = val_loss.item()
            fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and step_count % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-{iter_checkpoint_path}-ckpt.pth"
            save_lora_checkpoint(fabric, model, checkpoint_path)
    
    # print best val iter
    fabric.print(f"best_val iter: {best_val['iter']}  loss: {best_val['val_loss']}")


@torch.inference_mode()
def validate(
    fabric: L.Fabric, model: GPT, val_data: List[Dict], tokenizer: Tokenizer, micro_batch_size: int, longest_seq_length: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data, micro_batch_size, longest_seq_length)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(model, encoded, max_returned_tokens=len(encoded) + eval_max_new_tokens, temperature=0.8)
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    fabric.print(output)

    model.train()
    return val_loss


def get_batch(
    fabric: L.Fabric, data: List[Dict], micro_batch_size: int, longest_seq_length: int, longest_seq_ix: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (micro_batch_size,))
    if longest_seq_ix is not None:
        # force the longest sample at the beginning so potential OOMs happen right away
        ix[0] = longest_seq_ix

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    # this could be `longest_seq_length` to have a fixed size for all batches
    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    if fabric.device.type == "cuda" and x.device.type == "cpu":
        x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    else:
        x, y = fabric.to_device((x, y))
    return x, y


def get_max_seq_length(data: List[Dict]) -> Tuple[int, int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    max_seq_length = max(lengths)
    longest_seq_ix = lengths.index(max_seq_length)
    # support easy override at the top of the file
    return (
        override_max_seq_length if isinstance(override_max_seq_length, int) else max_seq_length,
        max_seq_length,
        longest_seq_ix,
    )


def save_lora_checkpoint(fabric, model, file_path: Path):
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})

def get_swa_avg_fn():
    @torch.no_grad()
    def swa_update(averaged_param, current_param, num_averaged):
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)

    return swa_update

from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def update_parameters(ave_model, model, n_averaged, use_buffers=False):
    ave_model_param = ave_model
    model_param = model
    ave_model_param_detached = []
    model_param_detached = []
    for p_averaged, p_model in zip(ave_model_param, model_param):
        p_model_ = p_model.detach().to(p_averaged.device)
        ave_model_param_detached.append(p_averaged.detach())
        model_param_detached.append(p_model_)
        if n_averaged == 0:
            p_averaged.detach().copy_(p_model_)

    if n_averaged > 0:
        grouped_tensors = _group_tensors_by_device_and_dtype([ave_model_param_detached, model_param_detached])
        for ((device, _), ([self_params, model_params], _)) in grouped_tensors.items():
            avg_fn = get_swa_avg_fn()
            n_averaged = n_averaged.to(device)
            for p_averaged, p_model in zip(self_params, model_params):
                p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
    n_averaged += 1

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)