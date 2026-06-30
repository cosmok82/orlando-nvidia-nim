# orlando-nvidia-nim

NVIDIA NIM provider for pi with **thinking catalog extracted from the LangChain SDK** and **user-configurable sampling parameters**.

Same endpoint as `nvidia-nim` but separate provider name (`orlando-nvidia-nim`) to avoid collisions with other extensions.

## Setup

### 1. Get an NVIDIA NIM API Key

1. Go to [build.nvidia.com](https://build.nvidia.com)
2. Sign in or create an account
3. Navigate to any model page and click "Get API Key"
4. Copy your key (starts with `nvapi-`)

### 2. Set Your API Key

```bash
# Preferred by this extension
export NVIDIA_NIM_API_KEY=nvapi-your-key-here

# Also supported, matching NVIDIA's website examples
export NVIDIA_API_KEY=nvapi-your-key-here
```

Add one of these to your `~/.bashrc`, `~/.zshrc`, or shell profile to persist it.

### 3. Install the Extension

**As a pi package (recommended):**

```bash
pi install git:github.com/cosmok82/orlando-nvidia-nim
```

**Or load directly:**

```bash
pi -e /path/to/orlando-nvidia-nim
```

**Or copy to your extensions directory:**

```bash
cp -r orlando-nvidia-nim ~/.pi/agent/extensions/orlando-nvidia-nim
```


## Slash commands

### `/nim-config` — sampling config

Used **without arguments** displays the current state + syntax + ready-to-copy examples.
Used **with arguments** modifies and persists the config.

**Syntax**

```
/nim-config defaults {json}      -> changes global DEFAULTS (applied to all models)
/nim-config <modelId> {json}     -> PER-MODEL override (has priority over defaults)
```

Allowed parameters in `{json}`: `temperature`, `top_p`, `max_tokens`, `reasoning_budget`.

**Examples**

```
/nim-config
```
> displays state + syntax + examples on screen (self-explanatory).

```
/nim-config defaults {"temperature":0.5}
```
> sets `temperature=0.5` as global default (overwrites only that field).

```
/nim-config nvidia/nemotron-3-super-120b-a12b {"temperature":1,"top_p":0.95,"max_tokens":16384,"reasoning_budget":16384}
```
> override for that specific model (faithful to NVIDIA Playground prototype).

```
/nim-config deepseek-ai/deepseek-v4-flash {"temperature":0.3,"max_tokens":16384}
```

The config file is persisted to `~/.pi/agent/config-nvidia-nim.json`.
The **status line** in the footer (`orlando-nvidia-nim`) updates on model change / turn end / thinking change and shows:

```
T=0.3 topP=0.95 max=8k rb=16k ctx=128k 9%
```

where `ctx` = actual model contextWindow, `%` = percentage of context used, `rb` = reasoning_budget (visible only if set).

Models without per-model overrides fall back to `defaults`.
Coding default: `temperature=0.3, top_p=0.95, max_tokens=8192`.

### `/nim-refresh` — live refresh

Calls `GET /v1/models` live, matches the returned IDs with the embedded catalog (`models.json`), and re-registers the provider. Useful after NVIDIA-side additions/changes.

## Updating the model catalog (thinking / context)

When NVIDIA adds models or changes thinking flags:

1. `python C:/checkup/nim_probe_models.py`
   (requires `langchain-nvidia-ai-endpoints`) → produces `nim_langchain_dump.json`
   with all models + flags (`supports_thinking`, `thinking_param_enable/disable`,
   `supports_tools`, `deprecated`, `model_type`, ...).
2. `python C:/checkup/orlando-nvidia-nim/scripts/generate_catalog.py`
   → produces `orlando-nvidia-nim/models.json`.
3. Commit the new `models.json` + `/reload` in pi (or `/nim-refresh`).

## Technical details

- Current catalog: **136 models**, of which **25 with `supports_thinking=true`**.
- The 7 thinking models without explicit kwargs (`thinking_param_enable=null`) use
  the fallback `{"chat_template_kwargs":{"enable_thinking":true}}`.
- Models with `deprecated=true` are included with `[dep]` prefix in the display name.
- **Thinking** handled natively by pi via
  `compat.thinkingFormat: "chat-template"` + `compat.chatTemplateKwargs`
  (derived from each model's `thinking_param_enable`). No manual injection.
- Note: the `temperature/top_p/max_tokens/reasoning_budget` parameters are **NOT**
  exposed by the NVIDIA SDK. They are set **only** by the user via `/nim-config`.
- Per-model context window and max_tokens from an accurate table (~140 entries) sourced
  from the xRyul catalog; for models not in the table, conservative fallback 131072 / min(16384,ctx).

## License

MIT

---

**Author:** Cosimo Orlando (CosOr)
