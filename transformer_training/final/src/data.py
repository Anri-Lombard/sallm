import itertools
from pathlib import Path
import torch


def _load_data_shard(file: Path):
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(
    filename_pattern: str, batch_size: int, rank: int, world_size: int
):
    pattern_path = Path(filename_pattern)
    if pattern_path.is_absolute():
        files = sorted(pattern_path.parent.glob(pattern_path.name))
    else:
        files = sorted(Path.cwd().glob(filename_pattern))
    assert len(files) > 0, f"No files found matching pattern: {filename_pattern}"
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_cycle = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_cycle)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = (
                _load_data_shard(next(file_cycle)),
                0,
            )
        buf = tokens[pos + rank * local_batch_size :][: local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets
