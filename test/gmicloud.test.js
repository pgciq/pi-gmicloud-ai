import assert from "node:assert/strict";
import test from "node:test";
import extension from "../extensions/gmicloud.ts";

function loadExtension() {
  const commands = {};
  const renderers = {};
  let config;
  extension({
    registerProvider(_name, providerConfig) {
      config = providerConfig;
    },
    registerEntryRenderer(name, fn) {
      renderers[name] = fn;
    },
    registerCommand(name, def) {
      commands[name] = def;
    },
    appendEntry() {},
  });
  return { config, commands, renderers };
}

function getProviderConfig() {
  return loadExtension().config;
}

test("registers the gmi provider with the OpenAI-compatible base and env key", () => {
  const config = getProviderConfig();
  assert.equal(config.name, "GMI Cloud");
  assert.equal(config.baseUrl, "https://api.gmi-serving.com/v1");
  assert.equal(config.apiKey, "$GMI_API_KEY");
  assert.equal(config.api, "openai-completions");
  assert.equal(typeof config.streamSimple, "function");
  assert.equal(typeof config.refreshModels, "function");
});

test("flags reasoning and chat seed models correctly", () => {
  const { models } = getProviderConfig();
  assert.ok(Array.isArray(models) && models.length > 0);

  const reasoning = models.find((m) => m.reasoning);
  assert.ok(reasoning, "a reasoning model is seeded");
  assert.deepEqual(reasoning.input, ["text"]);

  const chat = models.find((m) => !m.reasoning && !m.gmiEmbeddingModel);
  assert.ok(chat, "a plain chat model is seeded");
  assert.deepEqual(chat.input, ["text"]);
});

test("uses a valid model array when refresh falls back to the cache", async () => {
  const config = getProviderConfig();
  const cachedModels = [{
    id: "cached-model",
    name: "Cached model",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 4096,
  }];

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("offline");
  };

  try {
    const result = await config.refreshModels({
      signal: new AbortController().signal,
      stored: { provider: "gmi", models: cachedModels },
      publish: async () => true,
    });
    assert.deepEqual(result, cachedModels);
    assert.ok(Array.isArray(result));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("uses the seed list when offline with no cache", async () => {
  const config = getProviderConfig();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("offline");
  };

  try {
    const result = await config.refreshModels({
      signal: new AbortController().signal,
      stored: undefined,
      publish: async () => true,
    });
    assert.ok(Array.isArray(result));
    assert.ok(result.length > 0);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("registers all gmi commands and both renderers", () => {
  const { commands, renderers } = loadExtension();
  for (const name of ["gmi-tts", "gmi-music", "gmi-image", "gmi-audio-models", "gmi-models"]) {
    assert.equal(typeof commands[name], "object", `${name} registered`);
    assert.equal(typeof commands[name].handler, "function", `${name} has a handler`);
  }
  assert.equal(typeof renderers["gmi-audio"], "function");
  assert.equal(typeof renderers["gmi-image"], "function");
});

test("gmi-tts handler reports missing key without throwing", async () => {
  const { commands } = loadExtension();
  const prev = process.env.GMI_API_KEY;
  delete process.env.GMI_API_KEY;
  let notified = null;
  try {
    await commands["gmi-tts"].handler("hello", { hasUI: true, ui: { notify: (m) => (notified = m) } });
    assert.match(notified, /Set GMI_API_KEY/);
  } finally {
    if (prev === undefined) delete process.env.GMI_API_KEY;
    else process.env.GMI_API_KEY = prev;
  }
});
