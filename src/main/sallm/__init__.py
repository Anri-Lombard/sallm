# TODO change factory.py to not be duplicated in directories
# TODO develop way for hpo to continue from crashed trial if hpc cuts short
from sallm.models.llama_compatibility import register_llama_compatibility

register_llama_compatibility()
