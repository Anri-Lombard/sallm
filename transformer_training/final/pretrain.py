import os
import torch
import torch.distributed as dist

from src.config import get_config
from src.trainer import Trainer


def main():
    config = get_config()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()

    trainer = Trainer(config, rank, world_size, device)
    trainer.train()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
