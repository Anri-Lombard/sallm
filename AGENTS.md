## Quick repository rules

This file is intentionally small. Follow these rules for generated or contributed code.

- Language: Python.
- No inline comments ("# ...") and no docstrings (triple-quoted strings) anywhere in source files.
- All imports must be top-level (at the top of the file). Do not import inside functions, methods, or conditionals.
- Do not wrap imports with try/except. Avoid try/except blocks where possible; prefer explicit checks and let errors surface.
- Use clear names and type hints. Tests are the place for examples and behavioral documentation.
- When running repository commands, first run `conda activate base` to ensure you are in the correct environment.
- Research the latest version of a library before you suggest changes to an api, especially look at what defaults are given so this codebase does not specify what is already default.

If a rule must be violated for a technical reason, document the deviation in the PR and get reviewer approval.
