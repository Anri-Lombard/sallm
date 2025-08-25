# SaLLM

## Finetuning Belebele

The project supports finetuning on the [facebook/belebele](https://huggingface.co/datasets/facebook/belebele) reading comprehension dataset. Select a model architecture and language code, then launch training:

```bash
TASK=belebele ARCH=<architecture> LANG=<language> sbatch scripts/launch_finetune.sh
```

Supported languages: `afr`, `eng`, `sot`, `ssw`, `tsn`, `tso`, `xho`, `zul`.
