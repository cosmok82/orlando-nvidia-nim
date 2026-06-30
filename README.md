# orlando-nvidia-nim

NVIDIA NIM provider for pi with **thinking catalog extracted from the LangChain SDK** and **user-configurable sampling parameters**.

Same endpoint as `nvidia-nim` but separate provider name (`orlando-nvidia-nim`) to avoid collisions with other extensions.

## Key Features

- **136-model catalog** extracted from the LangChain SDK — thinking flags, deprecation status, per-model context windows and max output tokens
- **`/nim-config`** slash command — configure `temperature`, `top_p`, `max_tokens`, `reasoning_budget` globally or per-model, persisted as JSON
- **`/nim-refresh`** slash command — queries `GET /v1/models` live and re-registers the provider
- **Native thinking** via pi `compat.thinkingFormat` + `compat.chatTemplateKwargs` — no manual injection, derived automatically from SDK metadata
- **Separate provider name** (`orlando-nvidia-nim`) — safe to install alongside other NIM extensions
- **Footer status line** showing current sampling parameters and context usage percentage


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

Calls `GET /v1/models` live, matches the returned IDs with the embedded catalog (`models.json`), and re-registers the provider. Use this to sync with the API if NVIDIA removes or renames models online.

### `/nim-refresh-catalog` — refresh catalog from SDK

Runs the Python toolchain (`scripts/nim_probe_models.py` → `scripts/generate_catalog.py`)
to fetch fresh model metadata from the LangChain SDK and regenerate `models.json`.
The provider is automatically re-registered with the updated catalog.

**Requirements:**
- Python 3.10+
- `langchain-nvidia-ai-endpoints` (`pip install langchain-nvidia-ai-endpoints`)
- `NVIDIA_API_KEY` or `NVIDIA_NIM_API_KEY` environment variable set

```
/nim-refresh-catalog
```

The command runs in two steps:
1. **Model probe** — queries NVIDIA API + LangChain SDK, dumps `nim_langchain_dump.json`
2. **Catalog generation** — converts the dump into `models.json` and reloads it

You can also run the scripts manually from the extension directory:

```bash
python scripts/nim_probe_models.py
python scripts/generate_catalog.py
```

## Updating the model catalog

Use `/nim-refresh-catalog` whenever NVIDIA adds new models or changes thinking flags.
The command automatically runs both Python scripts and reloads the catalog — no manual
steps required.

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
