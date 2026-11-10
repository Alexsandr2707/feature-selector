# Project code style

These rules apply to project-owned Python code, including new and rewritten
helper functions. Describe a function's contract before implementing it.
Preserve behavior during refactoring and identify bug fixes separately.

`src/feature_selector/models/s4.py` is third-party library code and remains
unchanged. Project requirements for style, annotations and docstrings do not
apply to that file. S4 integration elsewhere follows the general rules.

## Contracts, documentation and types

- Follow the docstring pattern in `build_pipeline` in `src/feature_selector/factory.py`:
  a short description, followed by `Parameters:` and `Returns:` in that order.
  Keep the Google-style entry format: `name (type): description`.
  Omit `Parameters:` when there are no arguments other than `self`/`cls`.
  Do not add a `Raises:` section.
- In `Parameters:`, state each argument's name, type in parentheses and purpose.
  Describe accepted input shapes and relevant constraints.
- In `Returns:`, state the result type and meaning, including `None` for
  functions that return no value.
- Keep documentation brief and clear: purpose, relevant argument constraints
  and result. Do not repeat the signature, obvious operations or default values
  already visible in the signature.
- Document input mutations and other significant side effects when present.
  Avoid obvious guarantees such as "inputs are not modified", "without modifying
  self" or "parameters are stored unchanged". Explain cloning and special cases
  only when relevant to usage. Do not leave `_summary_` or `_type_` placeholders.
- Annotate arguments and return values of every new or rewritten function,
  including internal helpers.
- Use concrete types, `|` unions, `Literal` for special strings and typed
  collections when they clarify the contract. Introduce readable aliases for
  recurring complex types.
- Use `Any` when a value intentionally remains unchecked until a later stage,
  or specifying dynamic data types would add unjustified complexity.
  Annotations do not replace runtime input validation.
- For dynamically available methods that Pylance cannot see on a declared type,
  retrieve the method with `getattr(obj, "method", None)` into a local variable,
  check it with `callable` and call that variable. Example from
  `TransformerStep.fit` in `src/feature_selector/pipeline/adapters.py`:

  ```python
  fit = getattr(self.estimator, "fit", None)
  if not callable(fit):
      raise TypeError("Transformer must provide fit")
  fit(selected.X, selected.y)
  ```

  Check the callable method itself, rather than a related attribute such as
  `_parameter_constraints`. Use this approach instead of `hasattr` followed by
  direct method access, `type: ignore` or casting the object to `Any`: it lets
  Pylance handle the call and verifies at runtime that the attribute is callable.
  Do not add a Protocol solely for calls at a dynamic factory boundary.
- Write docstrings and comments in English, matching the source code language.
- Inside functions, add short comments for non-obvious operations: explain
  their purpose, library limitations or the reason for a particular choice.
  Do not restate obvious conditions, assignments or calls.
- Use clear names and small functions with one responsibility. Extract helpers
  or classes when they make the main code easier to understand.

## Additional conventions

- Examples of the agreed style under `src/feature_selector/` include
  `factory.py`, `data.py`, `temporal.py`, `evaluation.py`, `artifacts.py`,
  `config.py`, `experiment.py`, `cli.py`, `models/` and `selectors/selection.py`.
- The maximum line length is 100 characters, as set in `[tool.ruff]` in
  `pyproject.toml`. Do not run bulk formatting across the entire project.
- Separate function bodies into logical blocks with blank lines: data
  preparation, validation, main processing and result construction. Choose
  boundaries by meaning and keep closely related statements together.
- Declare the types of shared attributes created by subclasses in `fit` on the
  base class without initial values. Do not initialize fitted attributes with
  empty values in the constructor: this can break `check_is_fitted`.
- Preserve the sklearn interface: constructors store parameters, `fit` returns
  `self`, fitted attributes end in `_` and cloning remains supported.
- Class-specific parameter checks may be extracted into `_validate_params`,
  called by the class itself when used. The factory constructs components and
  does not call `_validate_params` in advance. Validate data at the consuming
  step or algorithm boundary. Leave sklearn/PyTorch constraints to those
  libraries; do not duplicate checks and full-array scans between steps.
- Fix Pylance errors with an appropriate explicit type and type checks rather
  than suppression. For example, construct a `pd.Index` from requested names
  before `Index.get_indexer`. Use `cast` only for an already justified type,
  never as a substitute for data validation.

## Docstring example

```python
def get_selected_indices(self) -> ColumnIndices:
    """Return selected column positions in output order.

    Returns:
        ColumnIndices: Copy of the fitted input column positions.
    """
    return self.selected_indices_.copy()
```

Architectural contracts are described in [architecture.md](architecture.md);
installation and usage are covered in [README.md](../README.md).
