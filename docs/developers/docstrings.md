# Docstring Standard

All public API docstrings in Deckard must use **MyST-native Google-style**
sections, rendered via
[`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
and [`myst_nb`](https://myst-nb.readthedocs.io).
No reStructuredText markup is allowed in public docstrings.

## Required Sections

| Section | When to include |
|------------|-------------------------------|
| `Args:` | Any parameter |
| `Returns:` | Non-`None` return value |
| `Raises:` | Documented exceptions |
| `Note:` | Side effects, execution notes |
| `Example:` | Canonical usage (fenced block) |

## Syntax Rules

- **No RST markup**: Forbid `:param:`, `:type:`, `:rtype:`, `..
  code-block::`, etc.
- **Inline code**: Use single backticks: `` `my_field` ``
- **Cross-references**: Use MyST role syntax: {class}`deckard.data.base.DataConfig`
- **Code examples**: Use fenced Markdown blocks (```` ```python ````), not RST directives
- `napoleon_google_docstring = True` (see `docs/conf.py`)
- `napoleon_google_docstring = True` (see `docs/conf.py`); see the [napoleon
  docs](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) for
  full section reference
- Target: `napoleon_numpy_docstring = False` once all docstrings are migrated

## Example

````python
def _sensitive_labels_from_frame(self, frame: pd.DataFrame) -> pd.Series:
    """Resolve sensitive labels from *frame* using `sensitive_columns`.

    Args:
        frame (pd.DataFrame): Input data frame.

    Returns:
        pd.Series: Series of sensitive labels.

    Note:
        This method does not mutate the input frame.

    Example:
        ```python
        labels = config._sensitive_labels_from_frame(df)
        ```
    """
````

______________________________________________________________________

**Related:** [Mixin and Plugin Rules](plugins) | [Refactor Plan](refactor_plan)
