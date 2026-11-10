# Koharu Translator

koharu-translator provides one segment-preserving translation interface for
local GGUF models and hosted providers.

Model metadata is provider-neutral: each model has a stable provider ID, a
display name, and (for local models) selectable GGUF quantizations. Every
provider module owns its connection config, defaults, request mapping, request
limits, and model catalog or discovery.

The crate owns only provider connectivity. Provider settings such as base URLs
are persisted under `[providers]`; credentials remain in the platform keychain.
Translation model selection, target language, instructions, and generation
options belong to the caller. `koharu-pipeline` owns those values as the
configuration of its translation processor.

`Translator` is an execution engine. A caller supplies a `ModelSelection`,
`GenerationConfig`, and `TranslationRequest` for each operation. The engine
keeps the selected local model resident when possible and reads live provider
connection settings without owning workflow configuration.

A request carries one page or several pages in reading order through the same
prompt, and `GenerationConfig::max_tokens` is an output budget per page, as is
the default of a provider that must send one. The engine gives callers what
they need to group pages: an estimated token budget per request, a page's
estimated weight, whether a failure means the request was too large, and
whether the provider can split such a request. Hosted machine-translation
providers chunk requests to their limits instead.

A local model's trained context, recorded in its catalog entry, sets its
request budget. Each local request's output budget is clamped to what that
context leaves after the prompt, and a request whose prompt leaves too little
room for its translation fails as too large. LM Studio's budget follows the
context the server reports for the loaded model, or its small default context
while the model is not loaded.

A page image is kept in the form its provider reads: scaled pixels for a local
model, or only the JPEG a hosted provider sends.
