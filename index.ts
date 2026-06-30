/**
 * orlando-nvidia-nim
 * -------------------
 * NVIDIA NIM provider for pi with thinking catalog extracted from the LangChain SDK
 * (see scripts/generate_catalog.py) and user-configurable sampling parameter
 * overrides.
 *
 * Provider name: orlando-nvidia-nim
 * API: openai-completions (NVIDIA-compatible endpoint)
 *
 * @author Cosimo Orlando (CosOr)
 *
 *
 * Commands:
 *   /nim-refresh  -> calls GET /v1/models live, matches with models.json, re-registers
 *   /nim-config   -> read/modify sampling config (defaults + per-model)
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { streamSimpleOpenAICompletions } from "@earendil-works/pi-ai";
import { getAgentDir, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import catalogData from "./models.json" with { type: "json" };

// =============================================================================
// Constants
// =============================================================================
const BASE_URL = "https://integrate.api.nvidia.com/v1";
const PROVIDER_NAME = "orlando-nvidia-nim";
const ENV_NAMES = ["NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"] as const;
const CONFIG_FILE = join(getAgentDir(), "config-nvidia-nim.json");

// Coding defaults (applied if user does not configure)
const DEFAULTS = {
	temperature: 0.3,
	top_p: 0.95,
	max_tokens: 8192,
	reasoning_budget: undefined as number | undefined,
};

// Embedded catalog ----------------------------------------------------------------------------
interface CatalogEntry {
	id: string;
	model_type: string;
	supports_thinking: boolean;
	thinking_param_enable: Record<string, unknown> | null;
	thinking_param_disable: Record<string, unknown> | null;
	supports_tools: boolean;
	supports_structured_output: boolean;
	deprecated: boolean;
	aliases: string[] | null;
	base_model: string | null;
}
const CATALOG: CatalogEntry[] = (catalogData as { models: CatalogEntry[] }).models;
const CATALOG_BY_ID = new Map(CATALOG.map((m) => [m.id, m]));

// Fallback kwargs for the 7 models with supports_thinking=true but thinking_param_enable=null
const DEFAULT_ENABLE_KWARGS = { chat_template_kwargs: { enable_thinking: true } };
const DEFAULT_DISABLE_KWARGS = { chat_template_kwargs: { enable_thinking: false } };

// Accurate context window / max output tokens tables (sourced from NVIDIA catalog)
const CONTEXT_WINDOWS: Record<string, number> = {
	"deepseek-ai/deepseek-v3.1": 131072,
	"deepseek-ai/deepseek-v3.1-terminus": 131072,
	"deepseek-ai/deepseek-v3.2": 131072,
	"deepseek-ai/deepseek-v4-flash": 1048576,
	"deepseek-ai/deepseek-v4-pro": 1048576,
	"deepseek-ai/deepseek-r1-distill-llama-8b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-14b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-32b": 131072,
	"deepseek-ai/deepseek-r1-distill-qwen-7b": 131072,
	"deepseek-ai/deepseek-coder-6.7b-instruct": 16384,
	"moonshotai/kimi-k2-instruct": 131072,
	"moonshotai/kimi-k2-instruct-0905": 131072,
	"moonshotai/kimi-k2-thinking": 131072,
	"moonshotai/kimi-k2.6": 262144,
	"minimaxai/minimax-m2": 1048576,
	"minimaxai/minimax-m2.1": 1048576,
	"minimaxai/minimax-m2.7": 204800,
	"meta/llama-3.1-405b-instruct": 131072,
	"meta/llama-3.1-70b-instruct": 131072,
	"meta/llama-3.1-8b-instruct": 131072,
	"meta/llama-3.2-11b-vision-instruct": 131072,
	"meta/llama-3.2-1b-instruct": 131072,
	"meta/llama-3.2-3b-instruct": 131072,
	"meta/llama-3.2-90b-vision-instruct": 131072,
	"meta/llama-3.3-70b-instruct": 131072,
	"meta/llama-4-maverick-17b-128e-instruct": 1048576,
	"meta/llama-4-scout-17b-16e-instruct": 524288,
	"meta/llama3-70b-instruct": 8192,
	"meta/llama3-8b-instruct": 8192,
	"mistralai/mistral-large-3-675b-instruct-2512": 131072,
	"mistralai/mistral-medium-3-instruct": 131072,
	"mistralai/devstral-2-123b-instruct-2512": 131072,
	"mistralai/magistral-small-2506": 131072,
	"mistralai/mistral-large": 131072,
	"mistralai/mistral-large-2-instruct": 131072,
	"mistralai/mistral-small-24b-instruct": 32768,
	"mistralai/mistral-small-3.1-24b-instruct-2503": 131072,
	"mistralai/mistral-nemotron": 131072,
	"mistralai/mixtral-8x22b-instruct-v0.1": 65536,
	"mistralai/mixtral-8x7b-instruct-v0.1": 32768,
	"mistralai/codestral-22b-instruct-v0.1": 32768,
	"mistralai/ministral-14b-instruct-2512": 131072,
	"microsoft/phi-3-medium-128k-instruct": 131072,
	"microsoft/phi-3-mini-128k-instruct": 131072,
	"microsoft/phi-3-small-128k-instruct": 131072,
	"microsoft/phi-3-medium-4k-instruct": 4096,
	"microsoft/phi-3-mini-4k-instruct": 4096,
	"microsoft/phi-3-small-8k-instruct": 8192,
	"microsoft/phi-3-vision-128k-instruct": 131072,
	"microsoft/phi-3.5-mini-instruct": 131072,
	"microsoft/phi-3.5-moe-instruct": 131072,
	"microsoft/phi-3.5-vision-instruct": 131072,
	"microsoft/phi-4-mini-instruct": 131072,
	"microsoft/phi-4-mini-flash-reasoning": 131072,
	"microsoft/phi-4-multimodal-instruct": 131072,
	"qwen/qwen2-7b-instruct": 131072,
	"qwen/qwen2.5-7b-instruct": 131072,
	"qwen/qwen2.5-coder-32b-instruct": 131072,
	"qwen/qwen2.5-coder-7b-instruct": 131072,
	"qwen/qwen3-235b-a22b": 131072,
	"qwen/qwen3-coder-480b-a35b-instruct": 262144,
	"qwen/qwen3-next-80b-a3b-instruct": 131072,
	"qwen/qwen3-next-80b-a3b-thinking": 131072,
	"qwen/qwq-32b": 131072,
	"google/gemma-2-27b-it": 8192,
	"google/gemma-2-2b-it": 8192,
	"google/gemma-2-9b-it": 8192,
	"google/gemma-3-12b-it": 131072,
	"google/gemma-3-1b-it": 32768,
	"google/gemma-3-27b-it": 131072,
	"google/gemma-3-4b-it": 131072,
	"google/gemma-3n-e2b-it": 131072,
	"google/gemma-3n-e4b-it": 131072,
	"google/codegemma-1.1-7b": 8192,
	"nvidia/llama-3.1-nemotron-ultra-253b-v1": 131072,
	"nvidia/llama-3.1-nemotron-70b-instruct": 131072,
	"nvidia/llama-3.1-nemotron-51b-instruct": 131072,
	"nvidia/llama-3.3-nemotron-super-49b-v1": 131072,
	"nvidia/llama-3.3-nemotron-super-49b-v1.5": 131072,
	"nvidia/nemotron-4-340b-instruct": 4096,
	"nvidia/nvidia-nemotron-nano-9b-v2": 131072,
	"openai/gpt-oss-120b": 131072,
	"openai/gpt-oss-20b": 131072,
	"z-ai/glm4.7": 131072,
	"z-ai/glm5": 131072,
	"stepfun-ai/step-3.5-flash": 131072,
	"bytedance/seed-oss-36b-instruct": 131072,
	"ibm/granite-3.3-8b-instruct": 131072,
	"ibm/granite-3.0-8b-instruct": 8192,
	"ibm/granite-3.0-3b-a800m-instruct": 8192,
	"ibm/granite-34b-code-instruct": 8192,
	"ibm/granite-8b-code-instruct": 8192,
	"upstage/solar-10.7b-instruct": 4096,
	"01-ai/yi-large": 32768,
	"databricks/dbrx-instruct": 32768,
	"baichuan-inc/baichuan2-13b-chat": 4096,
	"thudm/chatglm3-6b": 8192,
	"tiiuae/falcon3-7b-instruct": 8192,
	"zyphra/zamba2-7b-instruct": 4096,
	"aisingapore/sea-lion-7b-instruct": 4096,
	"mediatek/breeze-7b-instruct": 4096,
	"meta/codellama-70b": 16384,
	"mistralai/mistral-7b-instruct-v0.2": 32768,
	"mistralai/mistral-7b-instruct-v0.3": 32768,
	"nv-mistralai/mistral-nemo-12b-instruct": 131072,
	"nvidia/nemotron-mini-4b-instruct": 4096,
	"nvidia/nemotron-4-mini-hindi-4b-instruct": 4096,
	"nvidia/usdcode-llama-3.1-70b-instruct": 131072,
	"sarvamai/sarvam-m": 32768,
	"writer/palmyra-creative-122b": 32768,
	"writer/palmyra-fin-70b-32k": 32768,
	"writer/palmyra-med-70b": 8192,
	"writer/palmyra-med-70b-32k": 32768,
	"igenius/colosseum_355b_instruct_16k": 16384,
	"igenius/italia_10b_instruct_16k": 16384,
	"rakuten/rakutenai-7b-chat": 4096,
	"rakuten/rakutenai-7b-instruct": 4096,
};
const MAX_TOKENS: Record<string, number> = {
	"deepseek-ai/deepseek-v3.1": 16384,
	"deepseek-ai/deepseek-v3.1-terminus": 16384,
	"deepseek-ai/deepseek-v3.2": 16384,
	"deepseek-ai/deepseek-v4-flash": 16384,
	"deepseek-ai/deepseek-v4-pro": 16384,
	"moonshotai/kimi-k2.6": 16384,
	"moonshotai/kimi-k2-instruct": 8192,
	"moonshotai/kimi-k2-thinking": 16384,
	"minimaxai/minimax-m2": 8192,
	"minimaxai/minimax-m2.1": 8192,
	"minimaxai/minimax-m2.7": 8192,
	"meta/llama-4-maverick-17b-128e-instruct": 16384,
	"meta/llama-4-scout-17b-16e-instruct": 16384,
	"z-ai/glm4.7": 16384,
	"z-ai/glm5": 16384,
	"qwen/qwen3-coder-480b-a35b-instruct": 65536,
	"nvidia/llama-3.1-nemotron-ultra-253b-v1": 32768,
	"openai/gpt-oss-120b": 16384,
	"openai/gpt-oss-20b": 16384,
	"mistralai/mistral-large-3-675b-instruct-2512": 16384,
	"mistralai/devstral-2-123b-instruct-2512": 32768,
};
function resolveContextWindow(id: string): number {
	return CONTEXT_WINDOWS[id] ?? 131072;
}
function resolveMaxTokens(id: string, ctx: number): number {
	return MAX_TOKENS[id] ?? Math.min(16384, ctx);
}

// =============================================================================
// User sampling config
// =============================================================================
interface SamplingConfig {
	defaults: typeof DEFAULTS;
	models: Record<string, Partial<typeof DEFAULTS>>;
}
function loadConfig(): SamplingConfig {
	try {
		if (existsSync(CONFIG_FILE)) {
			const raw = JSON.parse(readFileSync(CONFIG_FILE, "utf-8")) as SamplingConfig;
			return { defaults: { ...DEFAULTS, ...raw.defaults }, models: raw.models ?? {} };
		}
	} catch {
		/* fall through */
	}
	// create file with defaults
	const fresh: SamplingConfig = { defaults: DEFAULTS, models: {} };
	mkdirSync(getAgentDir(), { recursive: true });
	writeFileSync(CONFIG_FILE, JSON.stringify(fresh, null, 2), "utf-8");
	return fresh;
}
function saveConfig(cfg: SamplingConfig): void {
	mkdirSync(getAgentDir(), { recursive: true });
	writeFileSync(CONFIG_FILE, JSON.stringify(cfg, null, 2), "utf-8");
}

// =============================================================================
// Auth key
// =============================================================================
function getApiKeyEnvName(): string | undefined {
	return ENV_NAMES.find((n) => !!process.env[n]);
}
function providerApiKeyConfig(): string {
	const env = getApiKeyEnvName() ?? "NVIDIA_API_KEY";
	return `$${env}`;
}

// =============================================================================
// Model entries construction
// =============================================================================
interface ModelEntry {
	id: string;
	name: string;
	reasoning: boolean;
	input: ("text" | "image")[];
	contextWindow: number;
	maxTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
	compat: Record<string, unknown>;
}
function makeDisplayName(id: string, deprecated: boolean): string {
	const parts = id.split("/");
	const base = (parts[parts.length - 1] ?? id).replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
	return (deprecated ? "[dep] " : "") + base;
}
function buildEntry(c: CatalogEntry): ModelEntry {
	const isVision = c.model_type.toLowerCase() === "vlm";
	const ctx = resolveContextWindow(c.id);
	const mt = resolveMaxTokens(c.id, ctx);
	const enable = c.thinking_param_enable ?? DEFAULT_ENABLE_KWARGS;
	const disable = c.thinking_param_disable ?? DEFAULT_DISABLE_KWARGS;
	const inner = (enable.chat_template_kwargs ?? {}) as Record<string, unknown>;
	const innerKey = Object.keys(inner)[0] ?? "enable_thinking";
	const entry: ModelEntry = {
		id: c.id,
		name: makeDisplayName(c.id, c.deprecated),
		reasoning: c.supports_thinking,
		input: isVision ? ["text", "image"] : ["text"],
		contextWindow: ctx,
		maxTokens: mt,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		compat: {
			supportsReasoningEffort: false,
			supportsDeveloperRole: false,
			maxTokensField: "max_tokens",
			thinkingFormat: "chat-template",
			chatTemplateKwargs: {
				[innerKey]: { $var: "thinking.enabled", omitWhenOff: true },
			},
		},
	};
	// Mistral extra flags
	if (c.id.startsWith("mistralai/")) {
		entry.compat.requiresToolResultName = true;
		entry.compat.requiresThinkingAsText = true;
		entry.compat.requiresMistralToolIds = true;
	}
	return entry;
}
function buildAllEntries(): ModelEntry[] {
	return CATALOG.map(buildEntry);
}
// placed to avoid "unused disable": disable kwargs are rawEmbed (pi handles via omitWhenOff)
void DEFAULT_DISABLE_KWARGS;

// =============================================================================
// streamSimple wrapper -> injects sampling from user config
// =============================================================================
function nimStreamSimple(model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
	if (model.provider !== PROVIDER_NAME) {
		return streamSimpleOpenAICompletions(model as Model<"openai-completions">, context, options);
	}
	const cfg = loadConfig();
	const perModel = cfg.models[model.id] ?? {};
	const sampling = { ...cfg.defaults, ...perModel };
	const modifiedOptions: SimpleStreamOptions = {
		...options,
		onPayload: (params: unknown) => {
			const p = params as Record<string, unknown>;
			if (sampling.temperature !== undefined) p.temperature = sampling.temperature;
			if (sampling.top_p !== undefined) p.top_p = sampling.top_p;
			if (sampling.max_tokens !== undefined) p.max_tokens = sampling.max_tokens;
			if (sampling.reasoning_budget !== undefined) p.reasoning_budget = sampling.reasoning_budget;
			return options?.onPayload?.(params, model);
		},
	};
	return streamSimpleOpenAICompletions(model as Model<"openai-completions">, context, modifiedOptions);
}

// =============================================================================
// Live discovery helpers
// =============================================================================
function resolveApiKey(): string | undefined {
	const env = getApiKeyEnvName();
	if (env && process.env[env]) return process.env[env];
	return undefined;
}
async function fetchLiveModels(apiKey: string): Promise<string[]> {
	const resp = await fetch(`${BASE_URL}/models`, {
		headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
		signal: AbortSignal.timeout(15000),
	});
	if (!resp.ok) return [];
	const data = (await resp.json()) as { data?: { id: string }[] };
	return Array.isArray(data.data) ? data.data.map((m: { id: string }) => m.id) : [];
}

// =============================================================================
// Entry point
// =============================================================================
// Status line: shows sampling + model count/active model context
function fmtK(n: number | undefined): string {
    if (n === undefined) return "-";
    if (n >= 1024) return `${Math.round((n / 1024) * 10) / 10}k`;
    return String(n);
}
function updateStatusLine(ctx: ExtensionContext): void {
    const model = ctx.model;
    if (!model || model.provider !== PROVIDER_NAME) {
        ctx.ui.setStatus(PROVIDER_NAME, "");
        return;
    }
    const cfg = loadConfig();
    const s = { ...cfg.defaults, ...(cfg.models[model.id] ?? {}) };
    const parts: string[] = [`T=${s.temperature ?? "-"}`, `topP=${s.top_p ?? "-"}`, `max=${fmtK(s.max_tokens)}`];
    if (s.reasoning_budget !== undefined) parts.push(`rb=${fmtK(s.reasoning_budget)}`);
    parts.push(`ctx=${fmtK(model.contextWindow)}`);
    // percentage of context used (if available)
    const usage = ctx.getContextUsage?.();
    if (usage && usage.tokens && model.contextWindow) {
        parts.push(`${Math.round((usage.tokens / model.contextWindow) * 100)}%`);
    }
    ctx.ui.setStatus(PROVIDER_NAME, parts.join(" "));
}

export default function (pi: ExtensionAPI): void {
	const models = buildAllEntries();
	pi.registerProvider(PROVIDER_NAME, {
		baseUrl: BASE_URL,
		apiKey: providerApiKeyConfig(),
		api: "openai-completions",
		authHeader: true,
		models,
		streamSimple: nimStreamSimple,
	} as Parameters<typeof pi.registerProvider>[1]);

	// Status line update on model change, session start, turn end, thinking change
	pi.on("session_start", async (_e, ctx) => updateStatusLine(ctx));
	pi.on("model_select", async (_e, ctx) => updateStatusLine(ctx));
	pi.on("thinking_level_select", async (_e, ctx) => updateStatusLine(ctx));
	pi.on("turn_end", async (_e, ctx) => updateStatusLine(ctx));

	// /nim-refresh ----------------------------------------------------------
	pi.registerCommand("nim-refresh", {
		description: "Updates the orlando-nvidia-nim provider from the live API",
		handler: async (_args, ctx) => {
			const key = resolveApiKey();
			if (!key) {
				ctx.ui.notify("orlando-nvidia-nim: API key not configured", "error");
				return;
			}
			ctx.ui.setStatus("orlando-nvidia-nim", "Refresh...");
			const live = await fetchLiveModels(key);
			const found = live.filter((id) => CATALOG_BY_ID.has(id));
			ctx.ui.notify(`orlando-nvidia-nim: ${found.length}/${live.length} live matched`, "info");
			// Re-registers (embedded catalog; live models without an entry are excluded)
			pi.registerProvider(PROVIDER_NAME, {
				baseUrl: BASE_URL,
				apiKey: providerApiKeyConfig(),
				api: "openai-completions",
				authHeader: true,
				models: buildAllEntries(),
				streamSimple: nimStreamSimple,
			} as Parameters<typeof pi.registerProvider>[1]);
			ctx.ui.setStatus("orlando-nvidia-nim", `Refresh OK (${found.length})`);
		},
	});

	// /nim-config -----------------------------------------------------------
	pi.registerCommand("nim-config", {
		description: "Configure sampling: /nim-config defaults <json> | /nim-config <modelId> <json>",
		handler: async (args, ctx) => {
			const a = args.trim();
			if (!a) {
				const cfg = loadConfig();
				const help = [
					"[orlando-nvidia-nim] sampling configuration",
					"",
					`DEFAULTS: ${JSON.stringify(cfg.defaults)}`,
					`Models with override: ${Object.keys(cfg.models).length}`,
					"",
					"SYNTAX:",
					"  /nim-config defaults {json}            changes global defaults",
					"  /nim-config <modelId> {json}            override for a model",
					"Allowed parameters: temperature, top_p, max_tokens, reasoning_budget",
					"",
					"EXAMPLES:",
					"  /nim-config defaults {\"temperature\":0.5}",
					"  /nim-config nvidia/nemotron-3-super-120b-a12b {\"temperature\":1,\"top_p\":0.95,\"max_tokens\":16384,\"reasoning_budget\":16384}",
					"",
					"(The file is in ~/.pi/agent/config-nvidia-nim.json — the status line in footer updates on model change / turn end)",
				].join("\n");
				ctx.ui.notify(help, "info");
				return;
			}
			const firstSpace = a.indexOf(" ");
			if (firstSpace === -1) {
				ctx.ui.notify("Usage: /nim-config defaults <json> | /nim-config <modelId> <json>", "error");
				return;
			}
			const target = a.slice(0, firstSpace);
			const jsonPart = a.slice(firstSpace + 1);
			try {
				const parsed = JSON.parse(jsonPart) as Partial<typeof DEFAULTS>;
				const cfg = loadConfig();
				if (target === "defaults") {
					cfg.defaults = { ...cfg.defaults, ...parsed };
				} else {
					cfg.models[target] = { ...cfg.models[target], ...parsed };
				}
				saveConfig(cfg);
				ctx.ui.notify(`orlando-nvidia-nim: configured "${target}"`, "info");
			} catch (e) {
				ctx.ui.notify(`orlando-nvidia-nim: invalid JSON (${e instanceof Error ? e.message : String(e)})`, "error");
			}
		},
	});
}
