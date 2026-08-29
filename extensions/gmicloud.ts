// GMI Cloud Inference Engine (IE) — https://console.gmicloud.ai
//
// OpenAI-compatible API base: https://api.gmi-serving.com/v1
//   - chat + vision (VLM): POST /v1/chat/completions   (image_url content for VLM)
//   - embeddings:          POST /v1/embeddings         (handled by pi's openai-compat core)
// Auth: GMI_API_KEY env var (Bearer). The key is a JWT issued by the GMI Cloud
//       console (scope `ie_model`, product `IE`) — paste it from
//       https://console.gmicloud.ai → API Keys.
//
// Model IDs use the `provider/Model-Name` form, e.g.
//   deepseek-ai/DeepSeek-R1-Distill-Llama-70B
//   openai/gpt-5.2
//   anthropic/claude-opus-4.7
//   meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
// The full library is listed at
// https://docs.gmicloud.ai/model-quickstarts/model-library
//
// This is a plain OpenAI-compatible endpoint, so — unlike NVIDIA NIM — there is
// no second "genai" host and no special-cased request body. We delegate all
// streaming to pi's openai-completions core and only normalize the model
// catalog (seed list + background /v1/models refresh).

import { mkdir, writeFile } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { Image, Markdown } from "@earendil-works/pi-tui";

const UNIFIED_BASE = "https://api.gmi-serving.com/v1";
const API_KEY_ENV = "GMI_API_KEY";

// `openAICompletionsApi` lives on the bare `@earendil-works/pi-ai` export in
// older pi-ai builds but moved to the `@earendil-works/pi-ai/api/openai-completions.lazy`
// subpath in newer ones. Resolve it defensively so the extension loads on both.
const openAICompletionsApi = await (async () => {
  try {
    return (await import("@earendil-works/pi-ai/api/openai-completions.lazy")).openAICompletionsApi;
  } catch {
    return (await import("@earendil-works/pi-ai")).openAICompletionsApi;
  }
})();

// Models served on startup (a representative subset; the full 80+ model
// catalog is fetched live from /v1/models in the background and cached). These
// IDs are taken from the live /v1/models response so instant startup never
// offers a stale id. Reasoning models are flagged by id pattern; the live
// catalog refines everything else (context window, pricing, …).
const SEED = [
  // deepseek
  "deepseek-ai/DeepSeek-V3.2",
  "deepseek-ai/DeepSeek-R1-0528",
  // openai
  "openai/gpt-5.5",
  "openai/gpt-5.4-mini",
  "openai/gpt-oss-120b",
  // anthropic
  "anthropic/claude-opus-4.7",
  "anthropic/claude-sonnet-4.6",
  "anthropic/claude-haiku-4.5",
  // google
  "google/gemini-3.7-flash",
  "google/gemma-4-31b-it",
  // moonshot
  "moonshotai/Kimi-K2.6",
  "moonshotai/Kimi-K2-Thinking",
  // nvidia
  "nvidia/nemotron-3-ultra-550b-a55b",
  // zhipu
  "zai-org/GLM-5.3-Flash",
  // qwen
  "Qwen/Qwen3.8-27B",
  // others
  "tencent/hy3-preview",
  "kwaipilot/kat-coder-pro-v2",
  "x-ai/grok-4.5",
  "MiniMaxAI/MiniMax-M3",
  "XiaomiMiMo/MiMo-V2.5",
  "stepfun-ai/Step-3.7-Flash",
];

// ---------------------------------------------------------------------------
// Model helpers
// ---------------------------------------------------------------------------

// Reasoning models: distill-R1, *-Thinking, o1/o3/o4, qwq, etc.
function isReasoningId(id: string) {
  return /r1|r1-|thinking|reasoning|o1|o3|o4|qwq|-r-|deepcoder/i.test(id);
}
// Embedding models (e.g. laion/CLIP-ViT-B-32-...). Not streamed here — see
// streamGmi below — but flagged so they are excluded from chat selection.
function isEmbeddingId(id: string) {
  return /clip|embed/i.test(id);
}
// Vision (image-input) models, when the live /v1/models response exposes it.
function modelVisionHint(model: any): boolean {
  if (model == null) return false;
  if (model.vision === true) return true;
  const caps = model.capabilities ?? model.capability ?? {};
  if (caps.vision === true || caps.image === true) return true;
  const modalities: string[] = Array.isArray(model.input_modalities)
    ? model.input_modalities
    : Array.isArray(model.modalities)
      ? model.modalities
      : [];
  return modalities.some((m) => /^image$/i.test(m));
}

function convertModel(model: any) {
  const id = typeof model?.id === "string" ? model.id : String(model?.id ?? "");
  const embedding = isEmbeddingId(id);
  const reasoning = !embedding && isReasoningId(id);
  const vision = !embedding && modelVisionHint(model);

  // GMI's /v1/models returns context_length and per-token pricing; fall back
  // to sane defaults if a model omits them.
  const ctx = Number(model?.context_length ?? model?.contextWindow ?? 0);
  const contextWindow = Number.isFinite(ctx) && ctx > 0 ? ctx : 131072;
  const maxTokens = Number.isFinite(ctx) && ctx > 0 ? Math.min(ctx, 32768) : 16384;

  const pricing = model?.pricing ?? {};
  const perMillion = (value: unknown) => {
    const n = Number(value);
    return Number.isFinite(n) ? n * 1_000_000 : 0;
  };

  return {
    id,
    name: typeof model?.name === "string" && model.name ? model.name : id,
    reasoning,
    // Vision-capable models also accept image input.
    input: vision ? ["text", "image"] : ["text"],
    // GMI pricing is USD per token; Pi costs are USD per million tokens.
    cost: {
      input: perMillion(pricing.prompt),
      output: perMillion(pricing.completion),
      cacheRead: perMillion(pricing.input_cache_read ?? pricing.cache_read),
      cacheWrite: perMillion(pricing.input_cache_write ?? pricing.cache_write),
    },
    contextWindow,
    maxTokens,
    // Private metadata (pi ignores unknown fields); consumed by streamGmi.
    gmiEmbeddingModel: embedding,
  };
}

async function fetchModels(baseUrl: string, signal?: AbortSignal) {
  const apiKey = process.env[API_KEY_ENV];
  const headers: Record<string, string> = {};
  if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

  const res = await fetch(`${baseUrl}/models`, { headers, redirect: "follow", signal });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

  const payload = await res.json();
  // OpenAI /v1/models returns { data: [{ id, ... }] }
  const data = Array.isArray(payload?.data)
    ? payload.data
    : Array.isArray(payload)
      ? payload
      : [];
  return data
    .filter((m: any) => m && m.id)
    .map((m: any) => convertModel(m));
}

// ---------------------------------------------------------------------------
// Model catalog helpers for /gmi-models (FREE badge + capability)
// ---------------------------------------------------------------------------

// During the 2026-08-24 → 09-06 promo, GMI lists free chat models as a
// *duplicate* /v1/models entry whose pricing is all-zero. So a chat model is
// FREE iff its per-million cost is 0. (Verified: MiniMax-M3 / M2.7 each appear
// twice — once priced, once at 0; GLM-5.3-Flash / M2.5 keep real prices.)
function isChatFree(model: any): boolean {
  return model?.cost?.input === 0 && model?.cost?.output === 0;
}

// GMI may return two /v1/models rows per free id (priced + zero-price). Collapse
// to one row, preferring the FREE (zero-cost) copy.
function dedupChatModels(models: any[]): any[] {
  const map = new Map<string, any>();
  for (const m of models) {
    const id = m?.id;
    if (!id) continue;
    const existing = map.get(id);
    if (!existing) { map.set(id, m); continue; }
    if (isChatFree(m) && !isChatFree(existing)) map.set(id, m);
  }
  return [...map.values()];
}

function chatCapability(model: any): string {
  if (model?.gmiEmbeddingModel) return "Embedding";
  if (model?.reasoning) return "Reasoning";
  if (Array.isArray(model?.input) && model.input.includes("image")) return "Vision";
  return "Chat";
}

// Media (console request-queue) models expose no structured price/free flag, so
// the FREE badge for them is derived from the verified promo families.
const MEDIA_FREE_PATTERNS: RegExp[] = [
  /minimax-tts-speech-2\.8/i,
  /minimax-audio-voice-clone-speech-2\.8/i,
  /minimax-music-3\.0/i,
];

function mediaCapability(id: string): string {
  const s = String(id).toLowerCase();
  if (/voice-clone|voice_clone/.test(s)) return "Voice Clone";
  if (/tts|speech/.test(s)) return "TTS / Speech";
  if (/music/.test(s)) return "Music";
  if (/video/.test(s)) return "Video";
  if (/image|dream|flux|imagen|wan|seedream|sd3|stable-diffusion/.test(s)) return "Image";
  if (/avatar/.test(s)) return "Avatar";
  return "Media";
}

function mediaFree(id: string): boolean {
  return MEDIA_FREE_PATTERNS.some((re) => re.test(id));
}

// GMI's console media-model list also returns user/internal entries whose ids
// are bare UUIDs (auto-generated custom / fine-tune models). They are not
// selectable media models and only clutter the listing, so we drop them and
// de-duplicate. (Real, usable media models have names like
// `minimax-tts-speech-2.8-turbo`.)
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function isUuid(id: string): boolean {
  return UUID_RE.test(id);
}
function cleanMediaIds(ids: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of ids) {
    const id = String(raw ?? "").trim();
    if (!id || isUuid(id) || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------
// GMI Cloud is fully OpenAI-compatible, so chat/VLM simply delegates to pi's
// openai-completions core. Embedding models are not streamed through this path
// (pi has no openai-embeddings stream type); we surface a clear error instead
// of sending an embedding model to /chat/completions.

function streamGmi(model: any, context: any, options?: any) {
  if (model?.gmiEmbeddingModel) {
    const stream = createAssistantMessageEventStream();
    const output = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "pending",
      timestamp: Date.now(),
    };
    (async () => {
      try {
        stream.push({ type: "start", partial: output });
        throw new Error(
          `GMI model "${model.id}" is an embedding model and is not streamed through chat completions. Use a chat model from the GMI library.`,
        );
      } catch (error) {
        output.stopReason = options?.signal?.aborted ? "aborted" : "error";
        output.errorMessage = error instanceof Error ? error.message : String(error);
        stream.push({ type: "error", reason: output.stopReason, error: output });
        stream.end();
      }
    })();
    return stream;
  }

  return openAICompletionsApi().streamSimple(model, context, options);
}

// ---------------------------------------------------------------------------
// Media generation — audio, music, and images (MiniMax Speech 2.8 / Music 3.0 / image models)
// ---------------------------------------------------------------------------
// These use GMI Cloud's async *request-queue* API on the console host — it is
// NOT OpenAI-compatible and is separate from the chat base (api.gmi-serving.com).
// All media models share the same request-queue endpoint:
//   POST {AUDIO_BASE}/ie/requestqueue/apikey/requests
//   GET  {AUDIO_BASE}/ie/requestqueue/apikey/requests/{id}
// (Music also historically exposes /api/v1/apikey/requests; we use the
// request-queue endpoint for every media type for consistency.)
// Each call returns a `request_id`; you poll GET until status is "success",
// then read the asset URL from outcome.{audio_url, media_urls, medias}.
// During the 2026-08-24 → 09-06 promo, Speech 2.8 + Music 3.0 are free
// (H3 and image models are NOT — they return 402 Insufficient credits).

const AUDIO_BASE = "https://console.gmicloud.ai/api/v1";
const TTS_REQUEST_PATH = "/ie/requestqueue/apikey/requests";
const MUSIC_REQUEST_PATH = "/apikey/requests";
const TTS_MODEL_DEFAULT = "minimax-tts-speech-2.8-turbo";
const MUSIC_MODEL = "minimax-music-3.0";
const DEFAULT_VOICE = "English_expressive_narrator";

// Convert an absolute path to a clickable Markdown link (OSC 8 hyperlink).
function fileLink(p: string, label = p) {
  return `[${label}](${pathToFileURL(String(p)).href})`;
}

// Build a theme safe to hand to pi-tui's Markdown/Image renderers. pi-tui calls a
// large set of `theme.<method>(text)` functions; if the theme the host passes to
// a custom entry renderer is partial (e.g. only `fallbackColor`), re-rendering a
// heading/table crashes with "theme.underline is not a function". This forwards
// every access to the real theme when present and supplies a no-op pass-through
// for any method the provided theme is missing, so rendering never throws.
const MARKDOWN_THEME_METHODS = [
  "bold", "code", "codeBlock", "codeBlockBorder", "cursor", "description",
  "fallbackColor", "heading", "highlightCode", "hint", "hr", "italic",
  "label", "link", "linkUrl", "listBullet", "noMatch", "quote",
  "quoteBorder", "scrollInfo", "selectList", "selectedText", "strikethrough",
  "underline", "value",
];
function markdownTheme(theme: any): any {
  const base = theme && typeof theme === "object" ? theme : {};
  const noop = (s: any) => s;
  return new Proxy(base, {
    get(t, p) {
      const v = (t as any)[p];
      if (v !== undefined) return v;
      if (typeof p === "string" && MARKDOWN_THEME_METHODS.includes(p)) return noop;
      return v;
    },
  });
}

// Pull an audio URL out of the (varied) completion payload shapes.
function extractAudioUrl(json: any): string | null {
  const o = json?.outcome ?? {};
  if (typeof o.audio_url === "string" && o.audio_url) return o.audio_url;
  if (Array.isArray(o.media_urls)) {
    const u = o.media_urls.find((x: any) => x?.url)?.url;
    if (u) return u;
  }
  if (o.media_urls && typeof o.media_urls === "object") {
    const v = Object.values(o.media_urls).find((x) => typeof x === "string");
    if (v) return v as string;
  }
  if (Array.isArray(o.medias)) {
    const u = o.medias.find((x: any) => x?.url)?.url;
    if (u) return u;
  }
  return null;
}

async function saveAssetFile(buffer: Buffer, modelId: string, ext: string, dir: string) {
  const directory = join(process.cwd(), ".pi", dir);
  await mkdir(directory, { recursive: true });
  const safe = String(modelId).replace(/[\/.]/g, "-");
  const filePath = join(directory, `${safe}-${Date.now()}.${ext}`);
  await writeFile(filePath, buffer);
  return filePath;
}

async function generateAsset(opts: {
  path: string;
  model: string;
  payload: Record<string, unknown>;
  apiKey: string;
  signal?: AbortSignal;
  timeoutMs?: number;
  dir?: string;
  label?: string;
}): Promise<{ filePath: string; url: string; model: string }> {
  const { path, model, payload, apiKey, signal, timeoutMs = 300_000, dir = "generated-audio", label = "audio" } = opts;
  const headers = { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" };

  const postRes = await fetch(`${AUDIO_BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify({ model, payload }),
    signal,
  });
  if (!postRes.ok) {
    const txt = await postRes.text().catch(() => "");
    throw new Error(`GMI ${label} request failed (HTTP ${postRes.status}): ${txt.slice(0, 300)}`);
  }
  const created = await postRes.json().catch(() => ({} as any));
  const requestId = created?.request_id;
  if (!requestId) {
    throw new Error(`GMI audio: no request_id in response: ${JSON.stringify(created).slice(0, 300)}`);
  }

  const getUrl = `${AUDIO_BASE}${path}/${requestId}`;
  const deadline = Date.now() + timeoutMs;
  let last: any = created;
  while (Date.now() < deadline) {
    if (signal?.aborted) throw new Error("GMI audio generation cancelled");
    await new Promise((r) => setTimeout(r, 5000));
    try {
      const statusRes = await fetch(getUrl, { headers, signal });
      if (!statusRes.ok) continue;
      last = await statusRes.json().catch(() => last);
    } catch {
      continue;
    }
    const status = String(last?.status ?? "");
    if (status === "failed" || status === "error" || status === "cancelled") {
      throw new Error(`GMI ${label} generation ${status}: ${JSON.stringify(last?.outcome ?? last).slice(0, 300)}`);
    }
    // GMI reports a finished job as "success" (also seen: "completed"/"succeeded").
    if (status === "success" || status === "completed" || status === "succeeded") break;
  }

  const assetUrl = extractAudioUrl(last);
  if (!assetUrl) {
    throw new Error(`GMI ${label} generation finished but no URL found: ${JSON.stringify(last).slice(0, 400)}`);
  }

  const dl = await fetch(assetUrl, { headers, redirect: "follow", signal });
  if (!dl.ok) throw new Error(`GMI ${label} download failed (HTTP ${dl.status})`);
  const buf = Buffer.from(await dl.arrayBuffer());
  const mime = dl.headers.get("content-type") ?? "";
  const ext = (String(assetUrl).split("?")[0].split(".").pop()
    || (mime.includes("png") ? "png" : mime.includes("webp") ? "webp" : mime.includes("jpeg") ? "jpg" : "bin"))
    .toLowerCase().replace(/[^a-z0-9]/g, "");
  const filePath = await saveAssetFile(buf, model, ext || "bin", dir);
  return { filePath, url: assetUrl, model };
}

function registerMediaCommands(pi: any) {
  pi.registerEntryRenderer?.("gmi-audio", (entry: any, _options: any, theme: any) => {
    const md = entry?.data?.markdown ?? String(entry?.data?.path ?? "");
    try {
      return new Markdown(md, 1, 0, markdownTheme(theme));
    } catch {
      return new Markdown(md, 1, 0, markdownTheme(null));
    }
  });

  // Inline image renderer (pi-tui has Image but no Audio, so audio uses Markdown).
  pi.registerEntryRenderer?.("gmi-image", (entry: any, _options: any, theme: any) => {
    const image = entry.data ?? {};
    const imageTheme = markdownTheme(theme);
    try {
      const data = readFileSync(image.path).toString("base64");
      return new Image(data, image.mimeType || "image/png", imageTheme, { maxWidthCells: 80, maxHeightCells: 30 });
    } catch {
      return new Markdown(`Generated image unavailable: ${fileLink(image.path ?? "unknown path")}`, 1, 0, imageTheme);
    }
  });

  function output(ctx: any, markdown: string, entryName = "gmi-audio") {
    if (ctx?.mode === "tui") pi.appendEntry(entryName, { markdown });
    else if (ctx?.hasUI) ctx.ui.notify(markdown, "info");
    else console.log(markdown);
  }

  function parseFlags(args: string) {
    const flags: Record<string, string> = {};
    const rest: string[] = [];
    const tokens = (args || "").trim().split(/\s+/).filter(Boolean);
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      if (t.startsWith("--")) {
        flags[t.slice(2)] = tokens[++i] ?? "";
      } else {
        rest.push(t);
      }
    }
    return { flags, text: rest.join(" ") };
  }

  pi.registerCommand("gmi-tts", {
    description:
      "Generate speech via GMI MiniMax Speech 2.8 TTS (free 2026-08-24→09-06). Usage: /gmi-tts [--voice <id>] [--model minimax-tts-speech-2.8-turbo|hd] <text>",
    handler: async (args: string, ctx: any) => {
      const apiKey = process.env[API_KEY_ENV];
      if (!apiKey) {
        output(ctx, "⚠️ Set GMI_API_KEY to use GMI audio generation.");
        return;
      }
      const { flags, text } = parseFlags(args);
      if (!text) {
        output(ctx, "⚠️ Provide text to synthesize, e.g. `/gmi-tts Hello world`.");
        return;
      }
      const model = flags.model || TTS_MODEL_DEFAULT;
      const payload: Record<string, unknown> = { text, voice_id: flags.voice || DEFAULT_VOICE, format: "mp3" };
      output(ctx, `🔊 Generating speech with \`${model}\` …`);
      try {
        const { filePath, url } = await generateAsset({
          path: TTS_REQUEST_PATH,
          model,
          payload,
          apiKey,
          signal: ctx?.signal,
          timeoutMs: 240_000,
        });
        output(ctx, `### 🔊 GMI TTS\n- Model: \`${model}\`\n- Saved: ${fileLink(filePath)}\n- Source: ${url}`);
      } catch (e) {
        output(ctx, `❌ GMI TTS failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
  });

  pi.registerCommand("gmi-music", {
    description:
      "Generate music via GMI MiniMax Music 3.0 (free 2026-08-24→09-06). Usage: /gmi-music [--prompt \"<style>\"] <lyrics>",
    handler: async (args: string, ctx: any) => {
      const apiKey = process.env[API_KEY_ENV];
      if (!apiKey) {
        output(ctx, "⚠️ Set GMI_API_KEY to use GMI audio generation.");
        return;
      }
      const { flags, text } = parseFlags(args);
      if (!text) {
        output(ctx, "⚠️ Provide lyrics, e.g. `/gmi-music [verse]\\nHello world`. Use --prompt for style.");
        return;
      }
      const model = flags.model || MUSIC_MODEL;
      const payload: Record<string, unknown> = {
        lyrics: text,
        format: "mp3",
        sample_rate: 44100,
        bitrate: 256000,
      };
      if (flags.prompt) payload.prompt = flags.prompt;
      output(ctx, `🎵 Generating music with \`${model}\` …`);
      try {
        const { filePath, url } = await generateAsset({
          path: MUSIC_REQUEST_PATH,
          model,
          payload,
          apiKey,
          signal: ctx?.signal,
          timeoutMs: 480_000,
        });
        output(ctx, `### 🎵 GMI Music\n- Model: \`${model}\`\n- Saved: ${fileLink(filePath)}\n- Source: ${url}`);
      } catch (e) {
        output(ctx, `❌ GMI Music failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
  });

  pi.registerCommand("gmi-image", {
    description:
      "Generate an image via GMI (seedream-5.0-pro by default; others via --model). Usage: /gmi-image [--model <id>] [--size 2K|3K|WxH] <prompt>",
    handler: async (args: string, ctx: any) => {
      const apiKey = process.env[API_KEY_ENV];
      if (!apiKey) {
        output(ctx, "⚠️ Set GMI_API_KEY to use GMI image generation.");
        return;
      }
      const { flags, text } = parseFlags(args);
      if (!text) {
        output(ctx, "⚠️ Provide a prompt, e.g. `/gmi-image a watercolor lighthouse at sunset`.");
        return;
      }
      const model = flags.model || "seedream-5.0-pro";
      const payload: Record<string, unknown> = { prompt: text };
      if (flags.size) payload.size = flags.size;
      output(ctx, `🖼️ Generating image with \`${model}\` …`);
      try {
        const { filePath, url } = await generateAsset({
          path: TTS_REQUEST_PATH, // image models use the same request-queue endpoint as TTS
          model,
          payload,
          apiKey,
          signal: ctx?.signal,
          timeoutMs: 300_000,
          dir: "generated-images",
          label: "image",
        });
        if (ctx?.mode === "tui") pi.appendEntry("gmi-image", { path: filePath, mimeType: "image/png" });
        else if (ctx?.hasUI) ctx.ui.notify(`🖼️ GMI image saved: ${fileLink(filePath)}\nSource: ${url}`, "info");
        else console.log(`🖼️ GMI image saved: ${fileLink(filePath)}\nSource: ${url}`);
      } catch (e) {
        output(ctx, `❌ GMI image failed: ${e instanceof Error ? e.message : String(e)}`, "gmi-image");
      }
    },
  });

  pi.registerCommand("gmi-audio-models", {
    description: "List GMI audio/music models and the default TTS voice. Usage: /gmi-audio-models [tts|music|all]",
    handler: async (args: string, ctx: any) => {
      const apiKey = process.env[API_KEY_ENV];
      if (!apiKey) {
        output(ctx, "⚠️ Set GMI_API_KEY to list models.");
        return;
      }
      const kind = (args || "all").trim().split(/\s+/)[0] || "all";
      try {
        const listRes = await fetch(`${AUDIO_BASE}/ie/requestqueue/apikey/models`, {
          headers: { Authorization: `Bearer ${apiKey}` },
        });
        if (!listRes.ok) throw new Error(`HTTP ${listRes.status}`);
        const list = await listRes.json();
        const ids: string[] = cleanMediaIds(Array.isArray(list?.model_ids) ? list.model_ids : []);
        const audio = ids.filter((id) => /tts|music|voice|audio|speech/i.test(id));
        const wanted =
          kind === "tts" ? audio.filter((id) => /tts|voice|speech/i.test(id))
            : kind === "music" ? audio.filter((id) => /music/i.test(id))
              : audio;
        if (wanted.length === 0) {
          output(ctx, "No audio models found for that filter.");
          return;
        }
        const rows: { id: string; name: string; brief: string }[] = [];
        for (const id of wanted) {
          try {
            const r = await fetch(`${AUDIO_BASE}/ie/requestqueue/apikey/models/${encodeURIComponent(id)}`, {
              headers: { Authorization: `Bearer ${apiKey}` },
            });
            if (!r.ok) { rows.push({ id, name: id, brief: "" }); continue; }
            const m = await r.json();
            rows.push({ id, name: m.display_name || id, brief: m.brief_description || "" });
          } catch {
            rows.push({ id, name: id, brief: "" });
          }
        }
        const md = [
          "# GMI audio / music models",
          "",
          "| Model ID | Display | Notes |",
          "|---|---|---|",
          ...rows.map((r) => `| \`${r.id}\` | ${r.name} | ${(r.brief || "").replace(/\n/g, " ").slice(0, 80)} |`),
          "",
          "_Use with `/gmi-tts` (text-to-speech, default voice `English_expressive_narrator`, override with `--voice <id>`) or `/gmi-music` (music)._",
        ].join("\n");
        output(ctx, md);
      } catch (e) {
        output(ctx, `❌ Failed to list models: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
  });

  pi.registerCommand("gmi-models", {
    description:
      "List GMI models with a FREE badge + capability. Chat + media default to FREE-only; use --all to show every model. Usage: /gmi-models [chat|audio|music|image|video|all] [--free|--all]",
    handler: async (args: string, ctx: any) => {
      const apiKey = process.env[API_KEY_ENV];
      if (!apiKey) {
        output(ctx, "⚠️ Set GMI_API_KEY to list models.");
        return;
      }
      const tokens = (args || "").trim().split(/\s+/).filter(Boolean);
      let freeOnly = false;
      let showAll = false;
      const kinds: string[] = [];
      for (const t of tokens) {
        if (t === "--free") freeOnly = true;
        else if (t === "--all") showAll = true;
        else kinds.push(t.toLowerCase());
      }
      const kind = kinds[0] || "all";
      const wantChat = kind === "chat" || kind === "all";
      const wantMedia = ["audio", "music", "image", "video", "all"].includes(kind);
      const sections: string[] = [];
      try {
        if (wantChat) {
          let models: any[];
          try {
            models = dedupChatModels(await fetchModels(UNIFIED_BASE, ctx?.signal));
          } catch {
            models = dedupChatModels(SEED.map((id) => {
              const m = convertModel({ id });
              m.cost = { input: -1, output: -1, cacheRead: 0, cacheWrite: 0 };
              return m;
            }));
          }
          // Chat defaults to FREE-only (promo models); --all expands to every model.
          if (!showAll || freeOnly) models = models.filter(isChatFree);
          models.sort((a: any, b: any) => {
            const fa = isChatFree(a) ? 0 : 1, fb = isChatFree(b) ? 0 : 1;
            if (fa !== fb) return fa - fb;
            return String(a.id).localeCompare(String(b.id));
          });
          const rows = models.map((m) => {
            const cap = chatCapability(m);
            const free = isChatFree(m) ? "🟢 FREE" : "";
            const ctxN = m.contextWindow >= 1000 ? `${Math.round(m.contextWindow / 1000)}k` : `${m.contextWindow}`;
            const cost = m.cost.input < 0 ? "unknown" : `${m.cost.input.toFixed(2)} → ${m.cost.output.toFixed(2)}`;
            return `| \`${m.id}\` | Chat | ${cap} | ${free} | ${ctxN} | ${cost} |`;
          });
          sections.push(
            ["### Chat / Reasoning / Vision / Embedding", "",
             "| Model ID | Type | Capability | FREE | Context | Cost (USD/M, in→out) |",
             "|---|---|---|---|---|---|", ...rows].join("\n"),
          );
        }

        if (wantMedia) {
          const listRes = await fetch(`${AUDIO_BASE}/ie/requestqueue/apikey/models`, {
            headers: { Authorization: `Bearer ${apiKey}` }, signal: ctx?.signal,
          });
          if (!listRes.ok) throw new Error(`HTTP ${listRes.status}`);
          const list = await listRes.json();
          let ids: string[] = cleanMediaIds(Array.isArray(list?.model_ids) ? list.model_ids : []);
          if (kind !== "all") {
            const hay = (id: string) => (mediaCapability(id) + " " + id).toLowerCase();
            ids = ids.filter((id) => hay(id).includes(kind));
          }
          // Media defaults to FREE-only (the promo models); --all expands to every
          // named, non-UUID media model. --free also restricts to FREE.
          if (!showAll || freeOnly) ids = ids.filter(mediaFree);
          ids.sort((a, b) => {
            const fa = mediaFree(a) ? 0 : 1, fb = mediaFree(b) ? 0 : 1;
            if (fa !== fb) return fa - fb;
            return a.localeCompare(b);
          });
          const rows = ids.map((id) => {
            const cap = mediaCapability(id);
            const free = mediaFree(id) ? "🟢 FREE" : "";
            return `| \`${id}\` | Media | ${cap} | ${free} | — |`;
          });
          sections.push(
            ["### Media (audio / music / image / video)", "",
             "| Model ID | Type | Capability | FREE | Notes |",
             "|---|---|---|---|---|", ...rows].join("\n"),
          );
        }

        if (sections.length === 0) {
          output(ctx, "No models matched that filter.");
          return;
        }
        const allShown = showAll && !freeOnly;
        const footer = allShown
          ? "\n_FREE badge: chat FREE = live pricing is 0 (GMI lists promo models as a duplicate zero-price row); media FREE = verified promo families (Speech 2.8, Music 3.0). All others require account balance (402). Internal/UUID media models are hidden._"
          : "\n_FREE badge: chat FREE = live pricing is 0 (GMI lists promo models as a duplicate zero-price row); media FREE = verified promo families (Speech 2.8, Music 3.0). All others require account balance (402). Chat + media show FREE-only by default — pass `--all` to list every model (internal/UUID media models are hidden)._";
        output(ctx, sections.join("\n\n") + footer);
      } catch (e) {
        output(ctx, `❌ Failed to list models: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
  });
}

// ---------------------------------------------------------------------------
// Extension entry point (synchronous — no network on startup)
// ---------------------------------------------------------------------------

export default function (pi: any) {
  const baseUrl = UNIFIED_BASE;
  const apiKeyEnv = API_KEY_ENV;

  pi.registerProvider("gmi", {
    name: "GMI Cloud",
    baseUrl,
    // Keep this as an env reference even when the variable is absent. Pi can
    // then mark the provider as unconfigured instead of using a placeholder key.
    apiKey: `$${apiKeyEnv}`,
    api: "openai-completions",
    streamSimple: streamGmi,
    models: SEED.map((id) => convertModel({ id })),

    async refreshModels({ signal, stored, publish, allowNetwork }) {
      const cachedModels = Array.isArray(stored?.models) ? stored.models : undefined;
      const seedModels = SEED.map((id) => convertModel({ id }));

      if (allowNetwork === false || signal?.aborted) {
        return cachedModels?.length ? cachedModels : seedModels;
      }

      let models;
      try {
        const fetched = await fetchModels(UNIFIED_BASE, signal);
        // GMI lists free chat models as a duplicate (priced + zero-price) /v1/models
        // row; collapse to one entry (preferring the FREE copy) so the picker and
        // catalog don't show the same id twice.
        models = dedupChatModels(fetched);
      } catch {
        return cachedModels?.length ? cachedModels : seedModels;
      }

      if (models.length > 0) {
        await publish({ persist: { provider: "gmi", models } });
        return models;
      }

      return cachedModels?.length ? cachedModels : seedModels;
    },
  });

  // Media generation commands (TTS / music / image via GMI request-queue API).
  registerMediaCommands(pi);
}
