# Developer Extensions Design and Contract

Framework and plugin extension docs are grouped here so the extension surface
is easy to navigate from the developer index.

## Purpose and Rationale

This page is the developer-facing hub for extension-specific docs. It keeps
framework and plugin guidance in one place while letting the root developer
index stay focused on the higher-level runtime families.

## Internal Architecture

The extension docs are organized by implementation family:

- Framework-specific runtime docs: {doc}[pytorch](../../api/pytorch/index)
- Framework-specific runtime docs: {doc}[transformers](../../api/plugins/index)
- Plugin-specific runtime docs: {doc}[anjana](../../api/plugins/anjana), {doc}[fairlearn](../../api/plugins/fairlearn)

Each page covers the internal contract for its subsystem and links back to the
shared rules that govern mixins and plugins.

## Execution Model

Extension docs follow the same core flow as the runtime they document: the
developer page explains the contract, then the linked pages drill into the
framework or plugin-specific lifecycle.

## Contracts and Invariants

- Keep shared extension rules in {doc}`/developers/extensions/plugins` and
	{doc}`/developers/extensions/mixins`.
- Keep framework/plugin-specific behavior in the owning developer page.
- Avoid duplicating the same extension overview text in the root developer
	index and in this hub page.

## Extension Points

- {doc}[pytorch](../../api/pytorch/index)
- {doc}[transformers](../../api/plugins/index)
- {doc}[anjana](../../api/plugins/anjana)
- {doc}[fairlearn](../../api/plugins/fairlearn)

## Validation and Guardrails

- Keep extension guidance in sync with the API and overview extension maps.
- Verify that root navigation links to this hub instead of repeating the same
	extension pages in multiple places.

## Migration and Compatibility

When adding a new extension page, surface it here first and then link to it
from the root developer index if needed.

## See also

- {doc}`../index`
- {doc}`../../overview/extensions/index`

```{toctree}
:hidden:
:maxdepth: 1

pytorch
transformers
anjana
fairlearn
```
