"""Optional model integrations that must not affect ordinary installations."""

from importlib import import_module


def register_fla_gated_deltanet() -> bool:
    """Import FLA's GatedDeltaNet module so it registers Auto classes.

    Returning false only means the optional top-level package is not installed.
    A broken FLA installation must remain visible to the caller.
    """
    try:
        import_module("fla.models.gated_deltanet")
    except ModuleNotFoundError as exc:
        if exc.name == "fla":
            return False
        raise
    return True
