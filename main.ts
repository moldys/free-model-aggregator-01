// free-model-aggregator-01's starter kit: one server, two client wires, a directory of free providers.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth roughly a quarter of the score: use it to see the shape of a
// trial, then replace its routing and its wires with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// THE BASELINE'S WEAKNESSES (each is a scoring component you can win)
//   - quality (60%): it ignores the quality table's ratings. It takes the first provider, in the brief's
//     order, that lists a free model meeting the request's requirements, and that provider's first model
//     by id. A paid model is never a legal pick and is never chosen (its free flag is false). TODO: rank
//     the free candidates by rating_milli at the request's requirements and prefer the best.
//   - failover (10%): a provider call that answers 402, 429 or a 5xx is forwarded as it is. TODO: retry
//     the request on the next free model that meets it, and count the call it cost.
//   - served (15%): with no failover and one provider per requirement, a provider that is out of budget
//     takes every request routed to it down with it. TODO: spread calls over providers under their
//     per-trial budgets.
//   - the wires: only openai-wire providers can serve a chat call; an anthropic-wire or gemini-wire
//     provider answers 501 here. The catalog probe already speaks all three wires. TODO: build the other
//     two upstream requests and their event shapes.
//   - the catalog (5%): the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; a listing that hides its prices leaves the row short.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
}

/** Every free model that meets the requirements, ranked by the quality table's rating_milli. */
function routesFor(requires: Requirement): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model });
    }
  }
  out.sort((a, b) => {
    const ra = ratingOfModel(brief.ratings, a.probe.provider.id, a.model.id)?.rating_milli ?? 0;
    const rb = ratingOfModel(brief.ratings, b.probe.provider.id, b.model.id)?.rating_milli ?? 0;
    return rb - ra;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the routed provider's Chat Completions route. */
async function callProvider(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

function safeArgs(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string') {
    try {
      const v: unknown = JSON.parse(value);
      return isRecord(v) ? v : {};
    } catch {
      return {};
    }
  }
  return {};
}

/** OpenAI Chat Completions body as an Anthropic Messages request. */
function openAIToAnthropicReq(body: Json, model: string): Json {
  const src = Array.isArray(body.messages) ? body.messages : [];
  const messages: Json[] = [];
  let system = '';
  for (const raw of src) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') { system += typeof raw.content === 'string' ? raw.content : ''; continue; }
    if (role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : '' }] });
      continue;
    }
    const parts: Json[] = [];
    const c = raw.content;
    if (typeof c === 'string') parts.push({ type: 'text', text: c });
    else if (Array.isArray(c)) {
      for (const b of c) {
        if (!isRecord(b)) continue;
        if (typeof b.text === 'string') parts.push({ type: 'text', text: b.text });
        else if (b.type === 'image_url' || b.type === 'image') parts.push({ type: 'image', source: { type: 'base64', media_type: 'image/png', data: '' } });
      }
    }
    if (role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      const blocks: Json[] = [];
      const t = parts.length > 0 && typeof parts[0].text === 'string' ? (parts[0].text as string) : '';
      if (t !== '') blocks.push({ type: 'text', text: t });
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: safeArgs(call.function.arguments) });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 2048, messages };
  if (system !== '') out.system = system;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools: Json[] = [];
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    }
    if (tools.length > 0) out.tools = tools;
  }
  return out;
}

/** An Anthropic Messages answer as a Chat Completions object. */
function anthropicToOpenAIAnswer(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const calls: Json[] = [];
  for (const blk of content) {
    if (!isRecord(blk)) continue;
    if (blk.type === 'text' && typeof blk.text === 'string') text += blk.text;
    else if (blk.type === 'tool_use') calls.push({ id: typeof blk.id === 'string' ? blk.id : 'toolu_unknown', type: 'function', function: { name: typeof blk.name === 'string' ? blk.name : '', arguments: JSON.stringify(isRecord(blk.input) ? blk.input : {}) } });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const inTok = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const outTok = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-anthropic',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text !== '' ? text : null, ...(calls.length > 0 ? { tool_calls: calls } : {}) }, finish_reason: body.stop_reason === 'max_tokens' ? 'length' : body.stop_reason === 'tool_use' ? 'tool_calls' : 'stop' }],
    usage: { prompt_tokens: inTok, completion_tokens: outTok, total_tokens: inTok + outTok },
  };
}

/** OpenAI Chat Completions body as a Gemini generateContent request. */
function openAIToGemini(body: Json, model: string): Json {
  const src = Array.isArray(body.messages) ? body.messages : [];
  const contents: Json[] = [];
  let system = '';
  for (const raw of src) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') { system += typeof raw.content === 'string' ? raw.content : ''; continue; }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const c = raw.content;
    if (typeof c === 'string') parts.push({ text: c });
    else if (Array.isArray(c)) {
      for (const b of c) {
        if (!isRecord(b)) continue;
        if (typeof b.text === 'string') parts.push({ text: b.text });
        else if (b.type === 'image_url' || b.type === 'image') parts.push({ inline_data: { mime_type: 'image/png', data: '' } });
      }
    }
    if (raw.role === 'tool') parts.push({ text: '[tool result] ' + (typeof raw.content === 'string' ? raw.content : '') });
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const fns: Json[] = [];
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      fns.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    }
    if (fns.length > 0) out.tools = [{ functionDeclarations: fns }];
  }
  const cfg: Json = {};
  if (typeof body.temperature === 'number') cfg.temperature = body.temperature;
  if (typeof body.max_tokens === 'number') cfg.maxOutputTokens = body.max_tokens;
  if (Object.keys(cfg).length > 0) out.generationConfig = cfg;
  return out;
}

/** A Gemini generateContent answer as a Chat Completions object. */
function geminiToOpenAI(body: Json, model: string): Json {
  const cands = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(cands[0]) ? cands[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const p of parts) {
    if (isRecord(p) && typeof p.text === 'string') text += p.text;
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const inTok = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const outTok = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text !== '' ? text : null }, finish_reason: 'stop' }],
    usage: { prompt_tokens: inTok, completion_tokens: outTok, total_tokens: inTok + outTok },
  };
}

async function callProviderAnthropic(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'anthropic-version': '2023-06-01', ...authHeaders(route.probe.provider.auth, route.probe.provider.key) };
  return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(openAIToAnthropicReq(body, route.model.id)) });
}

async function callProviderGemini(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(route.probe.provider.auth, route.probe.provider.key) };
  return fetch(`${base}/v1beta/models/${route.model.id}:generateContent`, { method: 'POST', headers, body: JSON.stringify(openAIToGemini(body, route.model.id)) });
}

/** Emit a Chat Completions SSE stream from one non-streamed answer object. */
function writeOpenAIStream(res: ServerResponse, answer: Json, model: string): void {
  const id = typeof answer.id === 'string' ? answer.id : 'chatcmpl-agg';
  const created = 1767225600;
  const choices0 = Array.isArray(answer.choices) ? answer.choices : [];
  const c0 = isRecord(choices0[0]) ? choices0[0] : {};
  const msg = isRecord(c0.message) ? c0.message : {};
  const content = typeof msg.content === 'string' ? msg.content : '';
  const calls: Json[] = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
  const finish = typeof c0.finish_reason === 'string' ? c0.finish_reason : 'stop';
  const usage = isRecord(answer.usage) ? answer.usage : {};
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.write(sseEvent('chat.completion.chunk', { id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] }));
  if (content !== '') res.write(sseEvent('chat.completion.chunk', { id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { content }, finish_reason: null }] }));
  if (calls.length > 0) res.write(sseEvent('chat.completion.chunk', { id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { tool_calls: calls }, finish_reason: null }] }));
  res.write(sseEvent('chat.completion.chunk', { id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: {}, finish_reason: finish }] }));
  res.write(sseEvent('chat.completion.chunk', { id, object: 'chat.completion.chunk', created, model, choices: [], usage }));
  res.end();
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free provider with failover. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const routes = routesFor(requirementsFor(body));
  if (routes.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  for (const route of routes) {
    const wire = route.probe.provider.wire;
    const upBody: Json = { ...body, model: route.model.id };
    if (body.stream === true && wire !== 'openai') upBody.stream = false;
    let upstream: Response;
    try {
      if (wire === 'anthropic') upstream = await callProviderAnthropic(route, upBody);
      else if (wire === 'gemini') upstream = await callProviderGemini(route, upBody);
      else upstream = await callProvider(route, upBody);
    } catch {
      continue;
    }
    const status = upstream.status;
    if (status === 402 || status === 429 || status >= 500) {
      await upstream.text();
      continue;
    }
    if (body.stream === true) {
      if (wire === 'openai' && upstream.body !== null) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        await pipeStream(upstream.body, res);
        return;
      }
      const raw: unknown = await upstream.json();
      const conv = wire === 'anthropic' ? anthropicToOpenAIAnswer(isRecord(raw) ? raw : {}, route.model.id) : geminiToOpenAI(isRecord(raw) ? raw : {}, route.model.id);
      writeOpenAIStream(res, conv, route.model.id);
      return;
    }
    if (wire === 'anthropic' && status < 300) {
      const answer: unknown = await upstream.json();
      sendJson(res, 200, anthropicToOpenAIAnswer(isRecord(answer) ? answer : {}, route.model.id));
      return;
    }
    if (wire === 'gemini' && status < 300) {
      const answer: unknown = await upstream.json();
      sendJson(res, 200, geminiToOpenAI(isRecord(answer) ? answer : {}, route.model.id));
      return;
    }
    const text = await upstream.text();
    res.writeHead(status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  sendError(res, 'openai', 503, 'no_free_model', 'every candidate provider refused the request (402/429/5xx)');
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to a free provider and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const routes = routesFor(requirementsFor(body));
  if (routes.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  for (const route of routes) {
    const wire = route.probe.provider.wire;
    const requested = typeof body.model === 'string' && body.model !== '' ? body.model : route.model.id;
    let upstream: Response;
    try {
      if (wire === 'anthropic') {
        const base = joinBase(ENV_URL, route.probe.provider.base);
        const headers: Record<string, string> = { 'content-type': 'application/json', 'anthropic-version': '2023-06-01', ...authHeaders(route.probe.provider.auth, route.probe.provider.key) };
        upstream = await fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify({ ...body, model: route.model.id }) });
      } else if (wire === 'gemini') {
        upstream = await callProviderGemini(route, { ...body, model: route.model.id });
      } else {
        upstream = await callProvider(route, anthropicToOpenAI(body, route.model.id));
      }
    } catch {
      continue;
    }
    const status = upstream.status;
    if (status === 402 || status === 429 || status >= 500) {
      await upstream.text();
      continue;
    }
    if (body.stream === true && status < 300 && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (wire === 'anthropic') {
        await pipeStream(upstream.body, res);
      } else if (wire === 'gemini') {
        const text = await upstream.text();
        res.write(text);
        res.end();
      } else {
        for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
        res.end();
      }
      return;
    }
    if (status < 300) {
      if (wire === 'anthropic') {
        const text = await upstream.text();
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(text);
        return;
      }
      const answer: unknown = await upstream.json();
      sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
      return;
    }
    const text = await upstream.text();
    sendJson(res, status, anthropicError('api_error', `the provider answered HTTP ${status}: ${text.slice(0, 500)}`));
    return;
  }
  sendError(res, 'anthropic', 503, 'overloaded_error', 'every candidate provider refused the request (402/429/5xx)');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
