# Cluster Command Tips

## Live SLURM queue view

Use this on HEX when interactively watching job state:

```bash
watch -c -n 5 qstat
```

Notes:

- `-n 5` refreshes every five seconds.
- `-c` preserves color in the `qstat` output.
- For automated Codex heartbeat passes, still start with:

```bash
/scratch/slurm/bin/purequota
```
