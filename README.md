# pi-gmicloud-ai

Pi extension for **GMI Cloud Inference Engine (IE)** — the hosted inference
platform behind [console.gmicloud.ai](https://console.gmicloud.ai).

It wires GMI Cloud's **OpenAI-compatible** API into Pi so you can use the whole
GMI model library (DeepSeek, GPT, Claude, Gemini, Llama, Kimi, GLM, Qwen, …)
directly from Pi.

## Setup

```bash
export GMI_API_KEY=eyJhbGci...          # JWT from https://console.gmicloud.ai → API Keys
```

The key is a JWT issued by GMI Cloud with scope `ie_model` / product `IE`.
Paste the full token as the value of `GMI_API_KEY`.

Install like any pi extension:

```bash
npm install pi-gmicloud-ai
```

Then restart pi so the `gmi` provider is loaded.

## Capabilities

| Capability | Endpoint | Status |
|---|---|---|
| Chat (text) | `POST https://api.gmi-serving.com/v1/chat/completions` | ✅ live-verified (base URL from official docs) |
| Vision (VLM, image→text) | same, `image_url` content part | ✅ OpenAI-compatible (delegated to pi core) |
| Embeddings | `POST https://api.gmi-serving.com/v1/embeddings` | ⚠️ catalog-flagged, not streamed through this path yet |
| **Speech (TTS)** | `POST /api/v1/ie/requestqueue/apikey/requests` (poll + download) | ✅ wired as `/gmi-tts` (MiniMax Speech 2.8) |
| **Music** | `POST /api/v1/apikey/requests` (poll + download) | ✅ wired as `/gmi-music` (MiniMax Music 3.0) |
| **Image (text→image)** | `POST /api/v1/ie/requestqueue/apikey/requests` (poll + download) | ✅ wired as `/gmi-image` (e.g. `seedream-5.0-pro`) — *not* free (402 without balance) |

### API base

GMI Cloud exposes a single **OpenAI-compatible** base — no second "genai" host
and no special-cased request body (unlike NVIDIA NIM):

- **`https://api.gmi-serving.com/v1`** serves chat, vision, and embeddings
  (addressed by `model` in the request body). This is what the extension uses.

You can see the exact base URL in GMI's own docs — e.g. the
[CURSOR integration guide](https://docs.gmicloud.ai/coding-tools/cursor) tells
you to set the "Override OpenAI Base URL" to `https://api.gmi-serving.com/v1`.

### Model IDs

GMI model IDs use the `provider/Model-Name` form, e.g.

- `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
- `openai/gpt-5.2`
- `anthropic/claude-opus-4.7`
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`

The full library is listed at
[docs.gmicloud.ai/model-quickstarts/model-library](https://docs.gmicloud.ai/model-quickstarts/model-library).

## Media generation commands (TTS / Music / Image)

GMI's audio/music/image models use a separate **async request-queue API** on the
console host (not OpenAI-compatible), so they are exposed as pi **commands**
rather than chat models:

- **`/gmi-models [chat|audio|music|image|video|all] [--free|--all]`** — list GMI
  models with a **🟢 FREE** badge and capability. Chat models show
  Chat / Reasoning / Vision / Embedding; media models show TTS / Voice Clone /
  Music / Image / Video. **Chat + media default to FREE-only** (chat: MiniMax
  M3 / M2.7; media: Speech 2.8 TTS / Voice-Clone + Music 3.0); pass `--all` to
  list every model, or a capability like `image --all` to expand just that group.
  The FREE
  badge is **data-driven**: chat FREE = the live `/v1/models` `pricing` is `0`
  (GMI lists promo models as a duplicate zero-price row — verified for
  `MiniMax-M3` / `MiniMax-M2.7`); media FREE = verified promo families (Speech
  2.8, Music 3.0). Pass `--free` to list only free models.
- **`/gmi-tts <text>`** — MiniMax Speech 2.8 TTS. Options:
  - `--voice <id>` — voice id (default `English_expressive_narrator`)
  - `--model minimax-tts-speech-2.8-turbo|hd` — model (default `…-turbo`)
  - Example: `/gmi-tts --voice English_expressive_narrator Hello from Pi`
- **`/gmi-music --prompt "<style>" <lyrics>`** — MiniMax Music 3.0. Example:
  - `/gmi-music --prompt "lo-fi, chill" [verse]\nWalking down the empty street`
- **`/gmi-image [--model <id>] [--size 2K|3K|WxH] <prompt>`** — text→image
  (default `seedream-5.0-pro`). Example:
  - `/gmi-image a watercolor lighthouse at sunset`
- **`/gmi-audio-models [tts|music|all]`** — list the available audio/music model
  ids, display names, and descriptions (handy for picking a `--model`/`--voice`).

TTS and music POST a generation request, poll until `status: success`, download
the resulting audio, save it to `.pi/generated-audio/`, and show a clickable
file link (pi has no inline `Audio` renderer, so the file opens in your player).
Images render inline in the TUI via pi-tui's `Image` renderer and are also saved
to `.pi/generated-images/`.

> Requires `GMI_API_KEY`. During the 2026-08-24 → 09-06 promo, **Speech 2.8 and
> Music 3.0 are free** (H3 is excluded). **Image generation is NOT in the free
> range** and returns `402 Insufficient balance` on a zero-balance account.

## Free promo (2026-08-24 → 2026-09-06)

GMI Cloud + MiniMax are running a 14-day free campaign. On a **zero-balance**
account (no credits purchased), only these are **verified free** during the promo:

- `MiniMaxAI/MiniMax-M3` ✅ (chat — verified `200` through this extension)
- `MiniMaxAI/MiniMax-M2.7` ✅ (chat/reasoning — verified `200` through this extension)
- MiniMax **Speech 2.8** and **Music 3.0** (audio / music — wired as
  `/gmi-tts` and `/gmi-music`; TTS verified end-to-end)

> 💡 **To chat for free right now**, select `MiniMaxAI/MiniMax-M3` or
> `MiniMaxAI/MiniMax-M2.7` as your model. Any other chat model (e.g.
> `zai-org/GLM-5.3-Flash`, `openai/gpt-*`, `deepseek-ai/*`, `meta-llama/*`, …)
> returns **`402 Insufficient balance`** on a zero-balance account — that is
> expected, not a bug. Add credits to use the rest of the 80+ model catalog.

⚠️ Other MiniMax chat models (`MiniMax-M2.5`, `MiniMax-M1`, etc.) were **not**
verified free and also returned `402` — stick to M3 / M2.7 for free chat.
⚠️ **MiniMax-H3 is NOT in the free range** — it still requires account balance.
⚠️ **Image generation is also not in the free range** — `/gmi-image` (e.g.
`seedream-5.0-pro`) returns `402 Insufficient balance` on a zero-balance account.
The full catalog is still fetched from `/v1/models`; just pick a free id during
the promo to avoid the 402.

## Notes

- The seed list is a representative subset of IDs taken from the **live**
  `/v1/models` response (so instant startup never offers a stale id). The full
  catalog (80+ models) is fetched from `/v1/models` in the background and cached
  on disk. The live catalog is broader and newer than the
  [docs model library](https://docs.gmicloud.ai/model-quickstarts/model-library)
  snapshot — e.g. it includes `openai/gpt-5.5`, `google/gemini-3.7-flash`,
  `anthropic/claude-opus-4.7`, `moonshotai/kimi-k3`, `x-ai/grok-4.5`,
  `Qwen/Qwen3.8-*`, and others.
- Each model's `context_length` and per-token `pricing` (prompt / completion /
  cache read / cache write) are read from `/v1/models` and converted to Pi's
  per-million-token cost fields, so `/list-models` and usage accounting reflect
  real GMI prices.
- Reasoning models (`*-r1`, `*-Thinking`, `o1/o3`, `qwq`, …) are flagged via id
  pattern so Pi can route thinking budgets correctly.
- Vision (image-input) capability is detected from the live `/v1/models`
  response when GMI reports modalities; otherwise models default to text input.
- Embedding models in the library (e.g. `laion/CLIP-ViT-B-32-...`) are flagged
  but not streamed through chat completions in this extension yet.
