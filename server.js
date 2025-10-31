// server.js
require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const { OpenAI } = require("openai");
const multer = require("multer");
const FormData = require("form-data");
const fs = require("fs");
const path = require("path");
const { VertexAI } = require("@google-cloud/vertexai");
const { Pool } = require('pg');
const sdk = require("microsoft-cognitiveservices-speech-sdk");
const WebSocket = require("ws");
const http = require("http");

const crypto = require('crypto');
const ALGO = 'aes-256-gcm';
const KEY = Buffer.from(process.env.API_KEYS_MASTER_KEY || '', 'base64');
if (!KEY || KEY.length !== 32) throw new Error('API_KEYS_MASTER_KEY invalida (32 bytes base64)');

function decryptSecret(b64) {
    const buf = Buffer.from(b64, 'base64');
    const iv = buf.subarray(0, 12);
    const tag = buf.subarray(12, 28);
    const ct = buf.subarray(28);
    const decipher = crypto.createDecipheriv(ALGO, KEY, iv);
    decipher.setAuthTag(tag);
    const plain = Buffer.concat([decipher.update(ct), decipher.final()]);
    return plain.toString('utf8');
}

const AZ_ENDPOINT = process.env.AZURE_REALTIME_OPENAI_ENDPOINT;          // es. https://2707llm.openai.azure.com
const AZ_KEY = process.env.AZURE_REALTIME_OPENAI_API_KEY;          // KEY1/KEY2
const AZ_DEPLOY = process.env.AZURE_REALTIME_OPENAI_REALTIME_DEPLOYMENT; // nome deployment (NON il nome del modello)
const AZ_VER = process.env.AZURE_REALTIME_OPENAI_API_VERSION || "2025-04-01-preview";


/* ----------------------------- START --------------------------*/

// ======== AZURE TTS POOL (riuso connessioni) ========
const AZ_TTS_KEY = process.env.AZURE_TTS_KEY_AI_SERVICES;
const AZ_TTS_REGION = process.env.AZURE_REGION_AI_SERVICES;

// Dimensione massima pool per formato (aumenta se hai più concorrenza)
const MAX_SYNTH_PER_FORMAT = 6;

const MAX_QUEUE_PER_FORMAT = parseInt(process.env.AZ_TTS_MAX_QUEUE_PER_FORMAT || '50', 10);

// Mappa: { webm: [worker, ...], mp3: [worker, ...] }
const ttsPools = {
    webm: [],
    mp3: []
};

function createSynthesizerForFormat(format) {
    if (!AZ_TTS_KEY || !AZ_TTS_REGION) {
        throw new Error("Missing Azure Speech env vars (AZURE_TTS_KEY_AI_SERVICES, AZURE_REGION_AI_SERVICES)");
    }
    const speechConfig = sdk.SpeechConfig.fromSubscription(AZ_TTS_KEY, AZ_TTS_REGION);
    if (format === "webm") {
        speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Webm24Khz16BitMonoOpus;
    } else {
        speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3;
    }

    // Nessun AudioConfig: leggeremo i chunk dall'evento "synthesizing"
    const synthesizer = new sdk.SpeechSynthesizer(speechConfig);

    // Apri connessione in anticipo (warm)
    try {
        const conn = sdk.Connection.fromSpeechSynthesizer(synthesizer);
        conn.openConnectionAsync(
            () => { /* warm ok */ },
            (err) => { console.warn("openConnectionAsync failed:", err?.message || err); }
        );
    } catch (e) {
        console.warn("Connection warmup failed:", e?.message || e);
    }

    // Un "worker" incapsula un synthesizer riutilizzabile, con coda jobs
    return {
        synthesizer,
        busy: false,
        queue: [],
    };
}

function retireAndReplaceWorker(worker, format) {
    try { worker.synthesizer.close(); } catch { }
    const pool = ttsPools[format];
    const i = pool.indexOf(worker);
    if (i >= 0) pool.splice(i, 1);
    // crea e inserisci un nuovo worker “pulito”
    const replacement = createSynthesizerForFormat(format);
    pool.push(replacement);
    return replacement;
}


function getOrCreateWorker(format) {
    const pool = ttsPools[format];
    // prova a trovarne uno non occupato
    const free = pool.find(w => !w.busy);
    if (free) return free;
    // se tutti occupati ma c'è spazio, creane uno nuovo
    if (pool.length < MAX_SYNTH_PER_FORMAT) {
        const w = createSynthesizerForFormat(format);
        pool.push(w);
        return w;
    }
    // altrimenti scegli quello con coda più corta
    let best = pool[0];
    for (const w of pool) {
        if (w.queue.length < best.queue.length) best = w;
    }
    return best;
}

function enqueueTtsJob(format, job) {
    const worker = getOrCreateWorker(format);

    // Conta tutti i job in attesa/attivi nel pool di questo formato
    const totalQueued = ttsPools[format].reduce(
        (sum, w) => sum + w.queue.length + (w.busy ? 1 : 0),
        0
    );

    if (totalQueued >= MAX_QUEUE_PER_FORMAT) {
        return false; // istanza satura → il chiamante risponderà 429
    }

    worker.queue.push(job);
    if (!worker.busy) runNextJob(worker, format);
    return true;
}

function runNextJob(worker, format) {
    if (worker.queue.length === 0) return;
    worker.busy = true;

    const job = worker.queue.shift();
    const { ssml, res, req, contentType } = job;

    let started = false;
    let totalBytes = 0;
    let headersSent = false;
    let clientAborted = false;

    const sendHeadersOnce = () => {
        if (!headersSent) {
            res.setHeader("Content-Type", contentType);
            res.setHeader("Transfer-Encoding", "chunked");
            res.setHeader("Cache-Control", "no-store");
            if (typeof res.flushHeaders === "function") res.flushHeaders();
            headersSent = true;
        }
    };

    // Se il client chiude, non scrivere più
    req.on("aborted", () => { clientAborted = true; });

    // Watchdog: se non parte entro 15s, annulla job
    const watchdog = setTimeout(() => {
        if (!started && !res.headersSent) {
            cleanupHandlers();
            try { res.status(504).json({ error: "Azure TTS timeout before first audio chunk" }); } catch { }
            worker.busy = false;
            return runNextJob(worker, format);
        }
    }, 15000);

    // Handlers evento
    const onSynthesizing = (_s, e) => {
        try {
            if (clientAborted) return;
            const bytes = e?.result?.audioData;
            if (bytes && bytes.byteLength) {
                if (!started) {
                    sendHeadersOnce();
                    started = true;
                }
                totalBytes += bytes.byteLength;
                res.write(Buffer.from(bytes));
            }
        } catch (err) {
            console.error("write chunk error:", err);
        }
    };

    const onCompleted = (_s, e) => {
        clearTimeout(watchdog);
        cleanupHandlers();
        // Se non è mai arrivato niente → errore
        if (!started || totalBytes === 0) {
            if (!res.headersSent && !clientAborted) {
                try { res.status(502).json({ error: "No audio produced by Azure TTS" }); } catch { }
            } else {
                try { res.end(); } catch { }
            }
        } else {
            try { res.end(); } catch { }
        }
        worker.busy = false;
        runNextJob(worker, format);
    };

    const onCanceled = (_s, e) => {
        clearTimeout(watchdog);
        cleanupHandlers();
        const details = e?.errorDetails || "synthesis canceled";
        retireAndReplaceWorker(worker, format);
        if (!res.headersSent && !clientAborted) {
            try { res.status(502).json({ error: "Azure TTS canceled", details }); } catch { }
        } else {
            try { res.end(); } catch { }
        }
        worker.busy = false;
        runNextJob(worker, format);
    };

    function cleanupHandlers() {
        // rimuovi i listener per evitare leak tra job
        try { worker.synthesizer.synthesizing = undefined; } catch { }
        try { worker.synthesizer.synthesisCompleted = undefined; } catch { }
        try { worker.synthesizer.canceled = undefined; } catch { }
    }

    // Collega i listener per QUESTO job
    worker.synthesizer.synthesizing = onSynthesizing;
    worker.synthesizer.synthesisCompleted = onCompleted;
    worker.synthesizer.canceled = onCanceled;

    // Avvia la sintesi (SSML contiene già la voice)
    try {
        worker.synthesizer.speakSsmlAsync(
            ssml,
            () => { /* handled by events */ },
            (err) => {
                clearTimeout(watchdog);
                cleanupHandlers();
                console.error("speakSsmlAsync error:", err);
                retireAndReplaceWorker(worker, format);
                if (!res.headersSent && !clientAborted) {
                    try { res.status(500).json({ error: "Azure Speech TTS failed", details: String(err) }); } catch { }
                } else {
                    try { res.end(); } catch { }
                }
                worker.busy = false;
                runNextJob(worker, format);
            }
        );
    } catch (err) {
        clearTimeout(watchdog);
        cleanupHandlers();
        console.error("speakSsmlAsync throw:", err);
        retireAndReplaceWorker(worker, format);
        if (!res.headersSent && !clientAborted) {
            try { res.status(500).json({ error: "Azure Speech TTS init failed", details: String(err) }); } catch { }
        } else {
            try { res.end(); } catch { }
        }
        worker.busy = false;
        runNextJob(worker, format);
    }
}

/* ----------------------------- END ----------------------------*/




// Multer setup for file uploads
const upload = multer({ storage: multer.memoryStorage() });

const app = express();
const port = process.env.PORT || 8080;

// Write Google service account key to disk and set env var
const saKey = process.env.GOOGLE_SERVICE_ACCOUNT_KEY;
const keyPath = path.join("/tmp", "sa-key.json");
fs.writeFileSync(keyPath, saKey);
process.env.GOOGLE_APPLICATION_CREDENTIALS = keyPath;

// OpenAI SDK
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY_SIMULATEUR });
const openaiCreps = new OpenAI({ apiKey: process.env.OPENAI_API_KEY_SIMULATEUR });

// Vertex AI setup
const vertexAI = new VertexAI({
    project: process.env.GCLOUD_PROJECT,
    location: process.env.VERTEX_LOCATION
});
const vertexModel = vertexAI.getGenerativeModel({
    model: process.env.VERTEX_MODEL_ID,
    generationConfig: { maxOutputTokens: 2048 }
});

// Axios with timeout (for Azure/OpenAI HTTP calls)
const API_TIMEOUT = 320000; // 5 minutes
const axiosInstance = axios.create({ timeout: API_TIMEOUT });

// --- HeyGen axios ---
const heygen = axios.create({
    baseURL: "https://api.heygen.com",
    timeout: API_TIMEOUT,
    headers: { "X-Api-Key": process.env.HEYGEN_API_KEY }
});

// Global middlewares
app.use(cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "x-user-api-key", "authorization", "x-user-id", "x-user-email", "x-chatbot-id", "x-timer-chatbot-id"]
}));
app.use(express.json());

// pool DEVE stare prima di qualsiasi route che lo usa
const pool = new Pool({
    connectionString: process.env.PG_CONNECTION,
    ssl: {
        rejectUnauthorized: true,
        ca: process.env.PG_SSL_CA
    }
});

// START to set code with timer fot chatbot for service openaiSimulateur and azureOpenaiNotStream
// ================== TIMER PER-ID (IN-MEMORY) ==================
const THREE_YEARS = 3 * 365 * 24 * 60 * 60 * 1000;
// Summer + 2h - Winter + 1h
const TIMER_ID_POLICY = {
    //  Day * hr * mn * Sec* Ml (7 * 24 * 60 * 60 * 1000)
    "testTimer": {
        ttlMs: THREE_YEARS,
        sliding: false,
        hardStopAt: "2025-10-09T18:00:00Z"
    }
};

// Stato runtime: timerId -> { expiresAt:number }
const runtimeTimers = new Map();
const expiredSticky = new Set();
const REQUIRE_TIMER_ID = true;

function getTimerId(req) {
    // SOLO variabile "timer_chatbot_id" (body / query / header)
    return (
        req.body?.timer_chatbot_id ||
        req.query?.timer_chatbot_id ||
        req.get("x-timer-chatbot-id") ||
        ""
    ).toString().trim();
}
function getTimerPolicy(id) { return id ? TIMER_ID_POLICY[id] : null; }
const now = () => Date.now();

function ensureTimerSession(id) {
    const pol = getTimerPolicy(id);
    if (!pol) return null;               // id non gestito → nessun timer applicato
    if (expiredSticky.has(id)) {
        // Mantieni un record con expiresAt=0 così remaining=0
        let s = runtimeTimers.get(id);
        if (!s) { s = { expiresAt: 0 }; runtimeTimers.set(id, s); }
        return s;
    }

    let s = runtimeTimers.get(id);
    const ttl = Number(pol.ttlMs || 0);
    const hardStopAt = pol.hardStopAt ? new Date(pol.hardStopAt).getTime() : null;

    if (!s) {
        const base = ttl > 0 ? now() + ttl : now();
        const expiresAt = hardStopAt ? (isNaN(hardStopAt) ? base : Math.min(base, hardStopAt)) : base;
        s = { expiresAt };
        runtimeTimers.set(id, s);
    } else if (pol.sliding && ttl > 0 && s.expiresAt > now()) {
        const renewed = now() + ttl;
        s.expiresAt = hardStopAt ? Math.min(renewed, hardStopAt) : renewed;
    }
    return s;
}

function remainingTimerMs(id) {
    const s = runtimeTimers.get(id);
    return s ? Math.max(0, s.expiresAt - now()) : Infinity; // Infinity = nessun limite (id non in policy)
}
function isTimerExpired(id) {
    const pol = getTimerPolicy(id);
    if (!pol) return false; // id non gestito → non scade mai
    const rem = remainingTimerMs(id);
    const expired = rem <= 0;
    if (expired) {
        expiredSticky.add(id);       // non permettere ricreazione
        runtimeTimers.set(id, { expiresAt: 0 }); // garantisci remaining=0
    }
    return expired;
}

function rejectJsonTimer(res, id, reason = "session_expired", status = 403) {
    if (id) { expiredSticky.add(id); runtimeTimers.set(id, { expiresAt: 0 }); }
    res.status(status)
        .setHeader("Access-Control-Allow-Origin", "*")
        .setHeader("Access-Control-Expose-Headers", "X-Session-Remaining")
        .setHeader("X-Session-Remaining", "0")
        .json({ ok: false, error: reason, timer_chatbot_id: id, remaining_ms: 0 });
}

function rejectSSETimer(res, id, reason = "session_expired") {
    if (id) { expiredSticky.add(id); runtimeTimers.set(id, { expiresAt: 0 }); }
    try {
        res.write(`data: ${JSON.stringify({ error: true, message: reason, timer_chatbot_id: id, remaining_ms: 0 })}\n\n`);
        res.write("data: [DONE]\n\n");
    } finally {
        res.end();
    }
}

// END to set code with timer fot chatbot for service openaiSimulateur and azureOpenaiNotStream

// SSE helper for OpenAI Threads
async function streamAssistant(assistantId, messages, userId, res) {
    const thread = await openai.beta.threads.create({ messages });
    const run = await openai.beta.threads.runs.createAndStream(
        thread.id,
        { assistant_id: assistantId, stream: true, user: userId }
    );
    for await (const event of run) {
        const delta = event.data?.delta?.content;
        if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
    }
    res.write("data: [DONE]\n\n");
    res.end();
}

function escapeXml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");
}

function buildSSML({ text, voice }) {
    const locale = (voice || "fr-FR-RemyMultilingualNeural").substring(0, 5);
    return `
<speak version="1.0" xml:lang="${locale}">
  <voice name="${voice}">${escapeXml(text)}</voice>
</speak>`.trim();
}

// Whisper transcription endpoint
app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
    console.log("🔹 /api/transcribe, req.file:", req.file?.originalname, req.file?.size);
    const apiKey = process.env.OPENAI_API_KEY_SIMULATEUR;
    if (!apiKey) return res.status(500).json({ error: "OpenAI API key missing" });
    if (!req.file) return res.status(400).json({ error: "No audio file uploaded" });
    try {
        const form = new FormData();
        form.append("file", req.file.buffer, { filename: req.file.originalname });
        form.append("model", "whisper-1");
        const response = await axios.post(
            "https://api.openai.com/v1/audio/transcriptions",
            form,
            { headers: { ...form.getHeaders(), Authorization: `Bearer ${apiKey}` } }
        );
        console.log("🎉 Whisper response:", response.data);
        return res.json(response.data);
    } catch (err) {
        const details = err.response?.data || err.message;
        console.error("❌ Whisper transcription error details:", details);
        return res.status(err.response?.status || 500)
            .json({ error: "Transcription failed", details });
    }
});

// Main API router
app.post("/api/:service", upload.none(), async (req, res) => {
    const { service } = req.params;
    console.log("🔹 Servizio ricevuto:", service);
    console.log("🔹 Dati ricevuti:", JSON.stringify(req.body));
    try {

        // Azure OpenAI Chat (Simulator)
        if (service === "azureOpenai") {
            const apiKey = process.env.AZURE_OPENAI_KEY_SIMULATEUR;
            const endpoint = process.env.AZURE_OPENAI_ENDPOINT_SIMULATEUR;
            const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_SIMULATEUR;
            const apiVersion = process.env.AZURE_OPENAI_API_VERSION || "2024-08-01-preview";

            const apiUrl = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

            // headers SSE
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");
            res.setHeader("X-Accel-Buffering", "no");
            res.flushHeaders();

            const { messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty } = req.body || {};
            const payload = {
                messages: messages || [],
                stream: true,
                ...(temperature !== undefined ? { temperature } : {}),
                ...(max_tokens !== undefined ? { max_tokens } : {}),
                ...(top_p !== undefined ? { top_p } : {}),
                ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
                ...(presence_penalty !== undefined ? { presence_penalty } : {}),
            };

            try {
                const axiosResp = await axiosInstance.post(apiUrl, payload, {
                    headers: { "api-key": apiKey, "Content-Type": "application/json" },
                    responseType: "stream",
                    // importantissimo con SSE dietro proxy:
                    decompress: true,
                    transitional: { clarifyTimeoutError: true }
                });

                axiosResp.data.on("data", (chunk) => {
                    res.write(chunk.toString("utf8")); // passthrough 1:1 (Azure manda già "data: ...\n\n")
                });
                axiosResp.data.on("end", () => res.end());
                axiosResp.data.on("error", (e) => {
                    console.error("Azure SSE stream error:", e?.message || e);
                    res.write(`data: ${JSON.stringify({ error: true, message: "stream_error", details: String(e) })}\n\n`);
                    res.write("data: [DONE]\n\n");
                    res.end();
                });

            } catch (err) {
                const status = err?.response?.status || 500;
                const headers = err?.response?.headers || {};
                const requestId = headers["x-request-id"] || headers["x-ms-request-id"] || headers["apim-request-id"] || "";

                let body = err?.response?.data;
                if (Buffer.isBuffer(body)) { try { body = body.toString("utf8"); } catch { body = "<buffer>"; } }
                if (typeof body === "object") { try { body = JSON.stringify(body); } catch { } }

                console.error("❌ Azure SSE error:", { status, requestId, body });

                // Invia i DETTAGLI al client come SSE "data: ..."
                res.write(`data: ${JSON.stringify({
                    error: true,
                    message: "azureOpenai error",
                    status,
                    requestId,
                    details: body || null
                })}\n\n`);
                res.write("data: [DONE]\n\n");
                res.end();
            }
        }

        else if (service === "azureOpenaiNotStream") {
            const apiKey = process.env.AZURE_OPENAI_KEY_SIMULATEUR;
            const endpoint = process.env.AZURE_OPENAI_ENDPOINT_SIMULATEUR;
            const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_SIMULATEUR;
            const apiVersion = process.env.AZURE_OPENAI_API_VERSION || "2024-11-20";

            const apiUrl = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

            const { messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty } = req.body || {};
            const payload = {
                messages: messages || [],
                stream: false, // ⬅️ importante
                ...(temperature !== undefined ? { temperature } : {}),
                ...(max_tokens !== undefined ? { max_tokens } : {}),
                ...(top_p !== undefined ? { top_p } : {}),
                ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
                ...(presence_penalty !== undefined ? { presence_penalty } : {}),
            };

            try {
                const { data } = await axiosInstance.post(apiUrl, payload, {
                    headers: { "api-key": apiKey, "Content-Type": "application/json" },
                });

                // Estraggo il testo in modo robusto (stringa o array di parti)
                let content = "";
                const choice = data?.choices?.[0];
                if (choice?.message?.content) {
                    content = Array.isArray(choice.message.content)
                        ? choice.message.content.filter(p => p.type === "text").map(p => p.text).join("")
                        : String(choice.message.content);
                }

                return res.status(200).json({ ok: true, content, raw: data });
            } catch (err) {
                const status = err?.response?.status || 500;
                const headers = err?.response?.headers || {};
                const requestId = headers["x-request-id"] || headers["x-ms-request-id"] || headers["apim-request-id"] || "";
                let details = err?.response?.data;

                try {
                    if (Buffer.isBuffer(details)) details = details.toString("utf8");
                } catch { }

                // una sola risposta, niente write/end multipli ⇒ niente ERR_HTTP_HEADERS_SENT
                return res.status(status).json({
                    ok: false,
                    message: "azureOpenai error",
                    status,
                    requestId,
                    details,
                });
            }
        }

        // Azure OpenAI non-stream con timer_chatbot_id
        else if (service === "azureOpenaiNotStreamTimer") {
            const tId = getTimerId(req);

            // Richiesta di ID obbligatorio e whitelistato
            if (REQUIRE_TIMER_ID && !tId) {
                return rejectJsonTimer(res, tId, "missing_timer_id", 400);
            }
            if (REQUIRE_TIMER_ID && !getTimerPolicy(tId)) {
                return rejectJsonTimer(res, tId, "invalid_timer_id", 403);
            }

            ensureTimerSession(tId);
            if (tId && isTimerExpired(tId)) {
                return rejectJsonTimer(res, tId);
            }

            const apiKey = process.env.AZURE_OPENAI_KEY_SIMULATEUR;
            const endpoint = process.env.AZURE_OPENAI_ENDPOINT_SIMULATEUR;
            const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_SIMULATEUR;
            const apiVersion = process.env.AZURE_OPENAI_API_VERSION || "2024-11-20";
            const apiUrl = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

            const { messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty } = req.body || {};
            const payload = {
                messages: messages || [],
                stream: false,
                ...(temperature !== undefined ? { temperature } : {}),
                ...(max_tokens !== undefined ? { max_tokens } : {}),
                ...(top_p !== undefined ? { top_p } : {}),
                ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
                ...(presence_penalty !== undefined ? { presence_penalty } : {}),
            };

            try {
                const { data } = await axiosInstance.post(apiUrl, payload, {
                    headers: { "api-key": apiKey, "Content-Type": "application/json" },
                });

                if (getTimerPolicy(tId)) {
                    res.setHeader("Access-Control-Allow-Origin", "*");
                    res.setHeader("Access-Control-Expose-Headers", "X-Session-Remaining");
                    res.setHeader("X-Session-Remaining", String(remainingTimerMs(tId)));
                }

                let content = "";
                const choice = data?.choices?.[0];
                if (choice?.message?.content) {
                    content = Array.isArray(choice.message.content)
                        ? choice.message.content.filter(p => p.type === "text").map(p => p.text).join("")
                        : String(choice.message.content);
                }

                return res.status(200).json({ ok: true, content, raw: data });
            } catch (err) {
                const status = err?.response?.status || 500;
                const headers = err?.response?.headers || {};
                const requestId = headers["x-request-id"] || headers["x-ms-request-id"] || headers["apim-request-id"] || "";
                let details = err?.response?.data;
                try { if (Buffer.isBuffer(details)) details = details.toString("utf8"); } catch { }

                return res.status(status).json({
                    ok: false,
                    message: "azureOpenai error",
                    status,
                    requestId,
                    details,
                });
            }
        }



        // Vertex Chat (batch + streaming) 
        else if (service === "vertexChat") {
            // CORS already applied globally
            const { messages, stream = true } = req.body;
            const promptText = messages.map(m => `${m.role.toUpperCase()}: ${m.content}`).join("\n");
            const request = { contents: [{ role: "user", parts: [{ text: promptText }] }] };

            // Batch invocation
            if (stream === false) {
                try {
                    const result = await vertexModel.generateContent({
                        ...request,
                        generationConfig: { maxOutputTokens: 2048 }
                    });
                    const text = result.response?.candidates?.[0]?.content?.parts?.[0]?.text || "";
                    return res.json({ text });
                } catch (err) {
                    console.error("Vertex AI batch error:", err);
                    return res.status(500).json({ error: err.message });
                }
            }

            // Streaming invocation
            try {
                res.setHeader("Content-Type", "text/event-stream");
                res.setHeader("Cache-Control", "no-cache");
                res.flushHeaders();
                const result = await vertexModel.generateContentStream(request);
                for await (const item of result.stream) {
                    const delta = item.candidates?.[0]?.content?.parts?.[0]?.text;
                    if (delta) res.write(`data: ${JSON.stringify({ delta })}\n\n`);
                }
                res.write("data: [DONE]\n\n");
                return res.end();
            } catch (err) {
                console.error("Vertex AI streaming error:", err);
                if (!res.headersSent) {
                    return res.status(500).json({ error: err.message });
                }
                res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
                res.write("data: [DONE]\n\n");
                return res.end();
            }
        }

        // OpenAI streaming with key openai dans BD -database-
        else if (service === "openaiSimulateurBlearn") {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.flushHeaders();

            // 1) Se presente, usa chiave passata da proxy
            let userKey =
                req.get("x-user-api-key") ||
                (req.get("authorization") || "").replace(/^Bearer\s+/i, "");

            try {
                // 2) Fallback: recupera dal DB in base al chatbot_id (storyline_key) e, se disponibile, all'utente
                if (!userKey) {
                    const chatbotIdStr = ((req.body && req.body.chatbot_id) || req.query.chatbot_id || req.get("x-chatbot-id") || "").toString().trim();
                    if (!chatbotIdStr) {
                        res.write(`data: ${JSON.stringify({ error: true, message: "chatbot_id manquant" })}\n\n`);
                        res.write("data: [DONE]\n\n"); return res.end();
                    }

                    // Opzionale: prova a identificare l'utente
                    const userIdHdr = req.get("x-user-id");
                    const userEmailHdr = req.get("x-user-email");

                    let userIdNum = userIdHdr ? Number(userIdHdr) : null;
                    if (!userIdNum && userEmailHdr) {
                        const rUid = await pool.query("SELECT id FROM users WHERE user_mail = $1", [userEmailHdr]);
                        userIdNum = rUid.rows[0]?.id || null;
                    }

                    let rKey;
                    if (userIdNum) {
                        // Chiave per utente + provider + chatbot_id (STRINGA = storyline_key)
                        rKey = await pool.query(
                            `SELECT enc_key FROM api_keys
           WHERE user_id = $1 AND provider = 'openai' AND chatbot_id = $2
           ORDER BY updated_at DESC LIMIT 1`,
                            [userIdNum, chatbotIdStr]
                        );
                    } else {
                        // Fallback: prendi la più recente per quel chatbot_id (se non distingui l'utente)
                        rKey = await pool.query(
                            `SELECT enc_key FROM api_keys
           WHERE provider = 'openai' AND chatbot_id = $1
           ORDER BY updated_at DESC LIMIT 1`,
                            [chatbotIdStr]
                        );
                    }

                    if (rKey.rows.length > 0) {
                        try { userKey = decryptSecret(rKey.rows[0].enc_key); } catch { }
                    }

                    // Ultimo fallback (se vuoi tenerlo): env globale
                    /*if (!userKey && !process.env.OPENAI_API_KEY_SIMULATEUR) {
                        res.write(`data: ${JSON.stringify({ error: true, message: "Aucune clé API disponible" })}\n\n`);
                        res.write("data: [DONE]\n\n"); return res.end();
                    }*/
                }

                const client = userKey ? new OpenAI({ apiKey: userKey }) : openai;

                // 3) Stream OpenAI identico a prima
                const stream = await client.chat.completions.create({
                    model: req.body.model,
                    messages: req.body.messages,
                    stream: true
                });
                for await (const part of stream) {
                    const delta = part.choices?.[0]?.delta?.content;
                    if (delta) res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: delta } }] })}\n\n`);
                }
                res.write("data: [DONE]\n\n");
                return res.end();
            } catch (err) {
                res.write(`data: ${JSON.stringify({ error: true, message: err.message })}\n\n`);
                res.write("data: [DONE]\n\n"); return res.end();
            }
        }

        // OpenAI streaming chat-completition (SDK)
        /*
        else if (service === "openaiSimulateur") {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.flushHeaders();
            const stream = await openai.chat.completions.create({
                model: req.body.model,
                messages: req.body.messages,
                stream: true
            });
            for await (const part of stream) {
                const delta = part.choices?.[0]?.delta?.content;
                if (delta) {
                    res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: delta } }] })}\n\n`);
                }
            }
            const totalTokens = stream.usage?.total_tokens || 0;
            res.write(`data: ${JSON.stringify({ usage: { total_tokens: totalTokens } })}\n\n`);
            res.write("data: [DONE]\n\n");
            return res.end();
        }
        */

        else if (service === "openaiSimulateur") {
            // ---- header SSE verso il browser ----
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("X-Accel-Buffering", "no");
            res.flushHeaders();

            try {
                const {
                    model,
                    messages,
                    temperature,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                    max_tokens,
                    max_output_tokens
                } = req.body || {};

                // ⚠️ Responses API usa:
                //   - "input" invece di "messages"
                //   - "max_output_tokens" invece di "max_tokens" per limitare la lunghezza dell'output. :contentReference[oaicite:2]{index=2}
                //
                // Buona notizia: puoi passare direttamente l'array `messages`
                // ( [ { role:"system"/"user"/"assistant", content:"..." }, ... ] )
                // dentro `input`. Questo è supportato dalla Responses API. :contentReference[oaicite:3]{index=3}
                //
                // Mappiamo max_tokens -> max_output_tokens per retrocompatibilità lato UI.
                const effectiveMaxOutputTokens =
                    max_output_tokens !== undefined
                        ? max_output_tokens
                        : (max_tokens !== undefined ? max_tokens : undefined);

                // --- chiamiamo la Responses API in streaming ---
                const stream = await openai.responses.create({
                    model: model || "gpt-4o", // fallback se il client non manda il modello
                    input: messages || [],
                    stream: true,
                    ...(temperature !== undefined ? { temperature } : {}),
                    ...(top_p !== undefined ? { top_p } : {}),
                    ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
                    ...(presence_penalty !== undefined ? { presence_penalty } : {}),
                    ...(effectiveMaxOutputTokens !== undefined ? { max_output_tokens: effectiveMaxOutputTokens } : {}),
                });

                // Ci teniamo l'ultima usage per mandarla al frontend alla fine
                let usageSnapshot = null;

                // La Responses API ci dà una serie di eventi.
                // Esempi di eventi:
                // - { type: "response.output_text.delta", delta: "ciao" }
                // - { type: "response.completed", response: { usage: {...} } }
                // - { type: "response.error", error: {...} }
                // :contentReference[oaicite:4]{index=4}
                for await (const event of stream) {
                    const t = event.type;

                    if (t === "response.output_text.delta") {
                        // chunk di testo incrementale
                        const deltaText = event.delta || "";
                        if (deltaText) {
                            // Manteniamo ESATTAMENTE il payload che il frontend già si aspetta
                            // così NON dobbiamo toccare il codice front.
                            res.write(
                                `data: ${JSON.stringify({
                                    choices: [{ delta: { content: deltaText } }]
                                })}\n\n`
                            );
                        }
                    }

                    else if (t === "response.completed") {
                        // Fine completamento. Qui possiamo leggere le usage tokens
                        // La struttura tipica è event.response.usage:
                        // { input_tokens, output_tokens, total_tokens, ... }
                        usageSnapshot = event.response?.usage || null;

                        const totalTokens =
                            (usageSnapshot?.total_tokens !== undefined
                                ? usageSnapshot.total_tokens
                                : ((usageSnapshot?.input_tokens || 0) +
                                    (usageSnapshot?.output_tokens || 0)
                                )
                            );

                        res.write(
                            `data: ${JSON.stringify({
                                usage: { total_tokens: totalTokens }
                            })}\n\n`
                        );
                    }

                    else if (t === "response.error") {
                        // Errore dal modello durante lo stream
                        res.write(
                            `data: ${JSON.stringify({
                                error: true,
                                message: event.error?.message || "openai error"
                            })}\n\n`
                        );
                    }

                    // Puoi ignorare altri tipi di evento (output_item.added, ecc.)
                }

                // chiusura stream verso frontend
                res.write("data: [DONE]\n\n");
                return res.end();

            } catch (err) {
                console.error("openaiSimulateur error:", err);

                // Se qualcosa va storto lato server/SDK,
                // mandiamo un ultimo pacchetto SSE d'errore + chiudiamo.
                try {
                    res.write(`data: ${JSON.stringify({
                        error: true,
                        message: "stream_failed",
                        details: String(err?.message || err)
                    })}\n\n`);
                    res.write("data: [DONE]\n\n");
                } catch { /* ignore */ }
                return res.end();
            }
        }

        else if (service === "openaiSimulateurCreps") {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.flushHeaders();
            const stream = await openaiCreps.chat.completions.create({
                model: req.body.model,
                messages: req.body.messages,
                stream: true
            });
            for await (const part of stream) {
                const delta = part.choices?.[0]?.delta?.content;
                if (delta) {
                    res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: delta } }] })}\n\n`);
                }
            }
            const totalTokens = stream.usage?.total_tokens || 0;
            res.write(`data: ${JSON.stringify({ usage: { total_tokens: totalTokens } })}\n\n`);
            res.write("data: [DONE]\n\n");
            return res.end();
        }

        // OpenAI streaming chat-completion (SDK) con timer_chatbot_id
        else if (service === "openaiSimulateurTimer") {
            // SSE headers
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache");
            res.setHeader("Connection", "keep-alive");
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("X-Accel-Buffering", "no");

            // Timer
            const tId = getTimerId(req);

            // ID richiesto e whitelistato
            if (REQUIRE_TIMER_ID && !tId) {
                return rejectSSETimer(res, tId, "missing_timer_id");
            }
            if (REQUIRE_TIMER_ID && !getTimerPolicy(tId)) {
                return rejectSSETimer(res, tId, "invalid_timer_id");
            }

            // Crea/aggiorna sessione; se scaduta → tombstone e chiudi
            ensureTimerSession(tId);
            if (tId && isTimerExpired(tId)) {
                return rejectSSETimer(res, tId);
            }

            if (getTimerPolicy(tId)) {
                res.setHeader("Access-Control-Expose-Headers", "X-Session-Remaining");
                res.setHeader("X-Session-Remaining", String(remainingTimerMs(tId)));
            }

            res.flushHeaders();

            let closed = false;
            const killer = getTimerPolicy(tId) ? setInterval(() => {
                if (closed) return;
                if (isTimerExpired(tId)) {
                    closed = true;
                    try { rejectSSETimer(res, tId); } catch { }
                }
            }, 1000) : null;

            res.on("close", () => { closed = true; if (killer) clearInterval(killer); });

            try {
                const { model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty } = req.body || {};
                const stream = await openai.chat.completions.create({
                    model,
                    messages,
                    stream: true,
                    ...(temperature !== undefined ? { temperature } : {}),
                    ...(max_tokens !== undefined ? { max_tokens } : {}),
                    ...(top_p !== undefined ? { top_p } : {}),
                    ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
                    ...(presence_penalty !== undefined ? { presence_penalty } : {}),
                });

                for await (const part of stream) {
                    if (closed) break;
                    if (getTimerPolicy(tId) && isTimerExpired(tId)) {
                        closed = true;
                        try { rejectSSETimer(res, tId); } catch { }
                        break;
                    }
                    const delta = part.choices?.[0]?.delta?.content;
                    if (delta) res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: delta } }] })}\n\n`);
                }

                const totalTokens = stream.usage?.total_tokens || 0;
                try {
                    res.write(`data: ${JSON.stringify({ usage: { total_tokens: totalTokens } })}\n\n`);
                    res.write("data: [DONE]\n\n");
                    res.end();
                } catch { }
            } catch (err) {
                console.error("openaiSimulateurTimer error:", err);
                if (!closed) {
                    try {
                        res.write(`data: ${JSON.stringify({ error: true, message: "stream_failed", details: String(err?.message || err) })}\n\n`);
                        res.write("data: [DONE]\n\n");
                        res.end();
                    } catch { }
                }
            } finally {
                if (killer) clearInterval(killer);
            }
        }

        // OpenAI Analyse (chat completions NON-stream, risposta uniforme)
        /*
        else if (service === "openaiAnalyse") {
            try {
                const {
                    model,
                    messages,
                    temperature,
                    max_tokens,
                    top_p,
                    frequency_penalty,
                    presence_penalty
                } = req.body || {};
                const resp = await openai.chat.completions.create({
                    model: model || "gpt-4.1-mini",
                    messages: messages || [],
                    stream: false,
                    ...(temperature !== undefined ? { temperature } : {}),
                    ...(max_tokens !== undefined ? { max_tokens } : {}),
                    ...(top_p !== undefined ? { top_p } : {}),
                    ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
                    ...(presence_penalty !== undefined ? { presence_penalty } : {}),
                });

                // estrai testo in modo robusto
                let content = "";
                const choice = resp?.choices?.[0];
                if (choice?.message?.content) {
                    content = Array.isArray(choice.message.content)
                        ? choice.message.content
                            .filter(p => p && (p.type === "text" || typeof p === "string"))
                            .map(p => (typeof p === "string" ? p : (p.text || "")))
                            .join("")
                        : String(choice.message.content);
                } else if (typeof choice?.text === "string") {
                    content = choice.text;
                }

                return res.status(200).json({ ok: true, content, raw: resp });
            } catch (err) {
                const status = err?.response?.status || 500;
                let details = err?.response?.data;
                try { if (Buffer.isBuffer(details)) details = details.toString("utf8"); } catch { }
                return res.status(status).json({
                    ok: false,
                    message: "openaiAnalyse error",
                    status,
                    details
                });
            }
        }
        */

        // OpenAI Analyse (Responses API NON-stream, risposta uniforme)
        else if (service === "openaiAnalyse") {
            try {
                const {
                    model,
                    messages = [],
                    temperature,
                    max_tokens,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                } = req.body || {};

                // 1. Estrai eventuale system/developer prompt come instructions
                let instructions;
                const conversationForInput = [];

                for (const m of messages) {
                    // Normalizza il contenuto in stringa (può essere string o array di parti)
                    let textContent = "";
                    if (typeof m.content === "string") {
                        textContent = m.content;
                    } else if (Array.isArray(m.content)) {
                        textContent = m.content
                            .map(p =>
                                typeof p === "string"
                                    ? p
                                    : (p && (p.text || p.content || "")) // fallback sicuro
                            )
                            .join("");
                    } else if (m && typeof m.content === "object" && m.content !== null) {
                        // tipo { text: "..."} ecc.
                        textContent = m.content.text || m.content.content || "";
                    }

                    if (
                        (m.role === "system" || m.role === "developer") &&
                        instructions === undefined
                    ) {
                        // prendi solo il PRIMO system/dev come instructions
                        instructions = textContent;
                    } else {
                        // tutto il resto va nell'input conversazionale
                        // La Responses API accetta array di messaggi {role, content}
                        conversationForInput.push({
                            role: m.role,      // "user" | "assistant"
                            content: textContent,
                        });
                    }
                }

                // Se per qualche ragione non abbiamo messaggi (edge case),
                // evita di mandare [] vuoto: manda stringa vuota
                const finalInput =
                    conversationForInput.length === 0
                        ? ""
                        : conversationForInput;

                // 2. Chiama la nuova Responses API
                const resp = await openai.responses.create({
                    model: model || "gpt-4o-mini", // era "gpt-4.1-mini", puoi scegliere il tuo default
                    input: finalInput,
                    ...(instructions ? { instructions } : {}),

                    // Parametri di controllo stile Chat Completions
                    ...(temperature !== undefined ? { temperature } : {}),
                    ...(top_p !== undefined ? { top_p } : {}),
                    // Chat Completions usava max_tokens; Responses usa max_output_tokens
                    ...(max_tokens !== undefined
                        ? { max_output_tokens: max_tokens }
                        : {}),
                    ...(frequency_penalty !== undefined
                        ? { frequency_penalty }
                        : {}),
                    ...(presence_penalty !== undefined
                        ? { presence_penalty }
                        : {}),
                });

                // 3. Estrai testo finale
                // Niente più choices[0].message.content:
                // Responses API ti dà direttamente .output_text
                const content = resp.output_text || "";

                return res.status(200).json({
                    ok: true,
                    content,
                    raw: resp, // <- oggetto completo: usage, tokens, ecc.
                });
            } catch (err) {
                const status = err?.response?.status || 500;
                let details = err?.response?.data;
                try {
                    if (Buffer.isBuffer(details)) details = details.toString("utf8");
                } catch { /* ignore */ }

                return res.status(status).json({
                    ok: false,
                    message: "openaiAnalyse error",
                    status,
                    details,
                });
            }
        }

        // RIMUOVERE
        // Azure OpenAI Analyse (batch)
        else if (service === "azureOpenaiAnalyse") {
            const apiKey = process.env.AZURE_OPENAI_KEY_SIMULATEUR;
            const endpoint = process.env.AZURE_OPENAI_ENDPOINT_SIMULATEUR;
            const deployment = process.env.AZURE_OPENAI_DEPLOYMENT_COACH;
            const apiVersion = process.env.AZURE_OPENAI_API_VERSION_COACH || "2024-11-20";

            if (!apiKey || !endpoint || !deployment) {
                return res.status(500).json({
                    ok: false, message: "Azure env missing", details: {
                        hasKey: !!apiKey, hasEndpoint: !!endpoint, hasDeployment: !!deployment
                    }
                });
            }
            const apiUrl = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

            const { messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty } = req.body || {};
            const payload = {
                messages: messages || [],
                stream: false,
                ...(temperature !== undefined ? { temperature } : {}),
                ...(max_tokens !== undefined ? { max_tokens } : {}),
                ...(top_p !== undefined ? { top_p } : {}),
                ...(frequency_penalty !== undefined ? { frequency_penalty } : {}),
                ...(presence_penalty !== undefined ? { presence_penalty } : {}),
            };

            try {
                const { data } = await axiosInstance.post(apiUrl, payload, {
                    headers: { "api-key": apiKey, "Content-Type": "application/json" },
                });

                // estrai testo in modo robusto (Azure può dare stringa o array di parti)
                let content = "";
                const choice = data?.choices?.[0];
                if (choice?.message?.content) {
                    content = Array.isArray(choice.message.content)
                        ? choice.message.content
                            .filter(p => p && (p.type === "text" || typeof p === "string"))
                            .map(p => (typeof p === "string" ? p : (p.text || "")))
                            .join("")
                        : String(choice.message.content);
                } else if (typeof choice?.text === "string") {
                    content = choice.text;
                }

                return res.status(200).json({ ok: true, content, raw: data });
            } catch (err) {
                const status = err?.response?.status || 500;
                const headers = err?.response?.headers || {};
                const requestId = headers["x-request-id"] || headers["x-ms-request-id"] || headers["apim-request-id"] || "";
                let details = err?.response?.data;
                try { if (Buffer.isBuffer(details)) details = details.toString("utf8"); } catch { }

                return res.status(status).json({
                    ok: false,
                    message: "azureOpenaiAnalyse error",
                    status,
                    requestId,
                    details,
                });
            }
        }

        // OpenAI TTS (batch)
        else if (service === "openai-tts") {
            const apiKey = process.env.OPENAI_API_KEY_SIMULATEUR;
            if (!apiKey) return res.status(500).json({ error: "OpenAI API key missing" });
            const { text, selectedVoice } = req.body;
            if (!text) return res.status(400).json({ error: "Text is required" });
            const allowedVoices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"];
            const voice = allowedVoices.includes((selectedVoice || "").trim().toLowerCase())
                ? selectedVoice.trim().toLowerCase()
                : "fable";
            try {
                const response = await axios.post(
                    "https://api.openai.com/v1/audio/speech",
                    { model: "gpt-4o-mini-tts", input: text, voice, instructions: "Speak in a gentle, slow and friendly way." },
                    { headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" }, responseType: "arraybuffer" }
                );
                res.setHeader("Content-Type", "audio/mpeg");
                return res.send(response.data);
            } catch (err) {
                console.error("OpenAI TTS error:", err.response?.data || err.message);
                return res.status(err.response?.status || 500).json({ error: "OpenAI TTS failed", details: err.message });
            }
        }

        else if (service === "azureTTS-Scaleway") {
            const {
                text,
                selectedLanguage,
                selectedVoice,
            } = req.body;

            if (!text) return res.status(400).json({ error: "Text is required" });

            const apiKey = process.env.AZURE_TTS_KEY_AI_SERVICES;
            const region = process.env.AZURE_REGION_AI_SERVICES;
            if (!apiKey || !region) {
                return res.status(500).json({
                    error: "Missing Azure Speech env vars (AZURE_TTS_KEY_AI_SERVICES, AZURE_REGION_AI_SERVICES)"
                });
            }

            const endpoint = `https://${region}.tts.speech.microsoft.com/cognitiveservices/v1`;

            const voiceMap = {
                "français": "fr-FR-RemyMultilingualNeural",
                "espagnol": "es-ES-ElviraNeural",
                "anglais": "en-US-JennyNeural"
            };
            const lang = (selectedLanguage || "").trim().toLowerCase();
            const voice = (selectedVoice && selectedVoice.trim()) || voiceMap[lang] || "fr-FR-RemyMultilingualNeural";

            const ssml = buildSSML({
                text,
                voice
            });

            try {
                const responseTTS = await axios.post(
                    endpoint,
                    ssml,
                    {
                        headers: {
                            "Ocp-Apim-Subscription-Key": apiKey,
                            "Content-Type": "application/ssml+xml; charset=utf-8",
                            "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3"
                        },
                        responseType: "arraybuffer",
                        timeout: API_TIMEOUT
                    }
                );

                res.setHeader("Content-Type", "audio/mpeg");
                return res.send(responseTTS.data);
            } catch (err) {
                const status = err.response?.status || 500;
                const headers = err.response?.headers || {};
                const requestId = headers["x-requestid"] || headers["x-ms-requestid"] || "";

                let textErr = err.response?.data;
                if (Buffer.isBuffer(textErr)) {
                    try { textErr = textErr.toString("utf8"); } catch { textErr = "<buffer>"; }
                } else if (typeof textErr === "object") {
                    textErr = JSON.stringify(textErr);
                }

                console.error("Azure Speech TTS error:", {
                    status,
                    requestId,
                    textErr,
                    ssml
                });

                const DEBUG_TTS = process.env.DEBUG_TTS === "true";

                return res.status(status).json({
                    error: "Azure Speech TTS failed",
                    azureStatus: status,
                    requestId,
                    details: textErr || "no-body",
                    ...(DEBUG_TTS ? { ssml } : {})
                });
            }

        }

        else if (service === "azureTTS-websocked-Scaleway") {
            const { text, selectedLanguage, selectedVoice } = req.body;
            const qFormat = (req.query.format || "").toLowerCase(); // ?format=webm / mp3
            const wantedFormat = (qFormat === "mp3" || qFormat === "webm") ? qFormat : "webm";

            if (!text || !text.trim()) {
                return res.status(400).json({ error: "Text is required" });
            }
            if (!AZ_TTS_KEY || !AZ_TTS_REGION) {
                return res.status(500).json({
                    error: "Missing Azure Speech env vars (AZURE_TTS_KEY_AI_SERVICES, AZURE_REGION_AI_SERVICES)"
                });
            }

            // Mappa lingua -> voce (default)
            const voiceMap = {
                "français": "fr-FR-RemyMultilingualNeural",
                "espagnol": "es-ES-ElviraNeural",
                "anglais": "en-US-JennyNeural"
            };
            const lang = (selectedLanguage || "").trim().toLowerCase();
            const voice = (selectedVoice && selectedVoice.trim()) || voiceMap[lang] || "fr-FR-RemyMultilingualNeural";

            const ssml = buildSSML({ text, voice });
            const contentType = wantedFormat === "webm" ? "audio/webm" : "audio/mpeg";

            const ok = enqueueTtsJob(wantedFormat, { ssml, res, req, contentType });
            if (!ok) {
                // Istanza satura: rispondi 429 così il client può ritentare o un'altra istanza prenderà il carico
                return res.status(429).json({ error: "Too many TTS requests on this instance, please retry" });
            }
        }



        // OpenAI Streaming TTS (SDK)
        else if (service === "streaming-openai-tts") {
            const { text, selectedVoice } = req.body;
            if (!text) return res.status(400).json({ error: "Text is required" });
            const allowed = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"];
            const voice = allowed.includes((selectedVoice || "").trim().toLowerCase())
                ? selectedVoice.trim().toLowerCase()
                : "fable";
            try {
                const ttsResp = await openai.audio.speech.create({ model: "tts-1", input: text, voice, instructions: "Speak in a cheerful and positive tone.", response_format: "mp3" });
                res.setHeader("Content-Type", "audio/mpeg");
                res.setHeader("Transfer-Encoding", "chunked");
                ttsResp.body.pipe(res);
            } catch (err) {
                console.error("OpenAI TTS error:", err);
                return res.status(500).json({ error: "OpenAI TTS failed" });
            }
        }

        // Azure Text-to-Speech
        else if (service === "azureTextToSpeech") {
            const { text, selectedVoice } = req.body;
            if (!text) return res.status(400).json({ error: "Text is required" });
            const endpoint = process.env.AZURE_TTS_ENDPOINT;
            const apiKey = process.env.AZURE_TTS_KEY;
            const deployment = "tts";
            const apiVersion = "2025-03-01-preview";
            const url = `${endpoint}/openai/deployments/${deployment}/audio/speech?api-version=${apiVersion}`;
            const voiceMap = { alloy: "alloy", echo: "echo", fable: "fable", onyx: "onyx", nova: "nova", shimmer: "shimmer" };
            const voice = voiceMap[(selectedVoice || "").trim().toLowerCase()] || "fable";
            try {
                const response = await axios.post(url, { model: "tts-1", input: text, voice },
                    { headers: { "Content-Type": "application/json", "api-key": apiKey, "Accept": "audio/mpeg" }, responseType: "arraybuffer" }
                );
                res.setHeader("Content-Type", "audio/mpeg");
                return res.send(response.data);
            } catch (err) {
                console.error("Azure TTS error:", err.response?.data || err.message);
                return res.status(err.response?.status || 500).json({ error: "Azure TTS failed", details: err.message });
            }
        }
        else if (service === "userList") {
            // aggiungi timeSession dal body (stringa 'HH:MM:SS')
            const { chatbotID, userID, userName, userScore,
                historique, rapport, usergroup, timeSession } = req.body;

            try {
                const result = await pool.query(
                    `INSERT INTO userlist (chatbot_name, user_email, name, score, chat_history, chat_analysis, usergroup, timesession)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::interval)
                    RETURNING *`,
                    [
                        chatbotID,
                        userID,
                        userName,
                        userScore,
                        historique,
                        rapport,
                        usergroup,
                        timeSession || 'N/A'
                    ]
                );

                return res
                    .status(201)
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                    .header("Access-Control-Allow-Headers", "Content-Type")
                    .json({ message: "Utente inserito!", data: result.rows[0] });
            } catch (err) {
                console.error("❌ Errore inserimento userList:", err);
                res
                    .status(500)
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                    .header("Access-Control-Allow-Headers", "Content-Type")
                    .json({ error: err.message });
            }
        }

        else if (service === "updateUserGroup") {
            const { userID, usergroup } = req.body;
            try {
                console.log("=== AGGIORNAMENTO GRUPPO UTENTE ===");
                console.log("UserID:", userID);
                console.log("Nuovo gruppo:", usergroup);

                const result = await pool.query(
                    "UPDATE userlist SET usergroup = $1 WHERE user_email = $2",
                    [usergroup, userID]
                );

                console.log("Record aggiornati:", result.rowCount);

                return res
                    .status(200)
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                    .header("Access-Control-Allow-Headers", "Content-Type")
                    .json({
                        message: "Record aggiornati!",
                        count: result.rowCount,
                        userID: userID,
                        usergroup: usergroup
                    });
            } catch (err) {
                console.error("❌ Errore aggiornamento gruppo:", err);
                res
                    .status(500)
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
                    .header("Access-Control-Allow-Headers", "Content-Type")
                    .json({ error: err.message });
            }
        }

        else if (service === "updateUserReview") {
            const { userID, stars, review } = req.body;

            try {
                const result = await pool.query(
                    `UPDATE userlist
                    SET stars = $1, review = $2
                    WHERE id = (
                    SELECT id FROM userlist 
                    WHERE user_email = $3
                    ORDER BY created_at DESC NULLS LAST, id DESC
                    LIMIT 1
                    )
                    RETURNING *`,
                    [stars, review, userID]
                );

                return res
                    .status(200)
                    .header("Access-Control-Allow-Origin", "*")
                    .json({
                        message: "Recensione aggiornata!",
                        count: result.rowCount,
                        data: result.rows[0]
                    });
            } catch (err) {
                console.error("❌ Errore aggiornamento recensione:", err);
                res.status(500).json({ error: err.message });
            }
        }

        // === Realtime Azure OpenAI (placeholder HTTP) ===
        else if (service === "fullCustomRealtimeAzureOpenAI") {
            // La Realtime richiede WebSocket, non HTTP POST.
            // Questo endpoint serve solo a dare un errore chiaro al client HTTP.
            return res.status(426).json({
                error: "Use WebSocket for Azure Realtime",
                websocket_endpoint: "/api/fullCustomRealtimeAzureOpenAI"
            });
        }
        else if (service === "elevenlabs-tts") {
            // La Realtime richiede WebSocket, non HTTP POST.
            // Questo endpoint serve solo a dare un errore chiaro al client HTTP.
            return res.status(426).json({
                error: "Use WebSocket for elevenlabs tts",
                websocket_endpoint: "/api/elevenlabs-tts"
            });
        }
        // ElevenLabs TTS
        else if (service === "elevenlabs") {
            const apiKey = process.env.ELEVENLAB_API_KEY;
            if (!apiKey) return res.status(500).json({ error: "ElevenLabs API key missing" });
            const { text, selectedLanguage } = req.body;
            const voiceMap = { espagnol: "l1zE9xgNpUTaQCZzpNJa", français: "1a3lMdKLUcfcMtvN772u", anglais: "7tRwuZTD1EWi6nydVerp" };
            const lang = (selectedLanguage || "").trim().toLowerCase();
            const voiceId = voiceMap[lang];
            if (!voiceId) return res.status(400).json({ error: "Not supported language" });
            const apiUrl = `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream`;
            try {
                const response = await axios.post(apiUrl,
                    { text, model_id: "eleven_flash_v2_5", voice_settings: { stability: 0.6, similarity_boost: 0.7, style: 0.1 } },
                    { headers: { "xi-api-key": apiKey, "Content-Type": "application/json" }, responseType: "arraybuffer" }
                );
                console.log("Audio received from ElevenLabs!");
                res.setHeader("Content-Type", "audio/mpeg");
                return res.send(response.data);
            } catch (err) {
                if (err.response) {
                    let msg;
                    try { msg = err.response.data.toString(); } catch { msg = "Unknown error"; }
                    console.error("ElevenLabs error:", msg);
                    return res.status(err.response.status).json({ error: msg });
                }
                console.error("Unknown error ElevenLabs:", err.message);
                return res.status(500).json({ error: "Unknown error with ElevenLabs" });
            }
        }

        // Fallback invalid service
        else {
            return res.status(400).json({ error: "Invalid service" });
        }
    } catch (error) {
        const status = error?.response?.status || 500;
        const headers = error?.response?.headers || {};
        const requestId = headers["x-requestid"] || headers["x-ms-requestid"] || "";

        let details = error?.response?.data;
        if (Buffer.isBuffer(details)) {
            try { details = details.toString("utf8"); } catch { details = "<buffer>"; }
        } else if (typeof details === "object") {
            details = JSON.stringify(details);
        }

        console.error(`❌ API error on /api/${req.params.service}`, {
            status,
            message: error.message,
            requestId,
            details,
        });

        return res.status(status).json({
            error: "API request error",
            service: req.params.service,
            status,
            message: error.message,
            requestId,
            details
        });
    }
});

// Secure endpoint to obtain Azure Speech token
app.get("/get-azure-token", async (req, res) => {
    const apiKey = process.env.AZURE_SPEECH_API_KEY;
    const region = process.env.AZURE_REGION_AI_SERVICES;
    if (!apiKey || !region) return res.status(500).json({ error: "Azure keys missing in the backend" });
    try {
        const tokenRes = await axios.post(
            `https://${region}.api.cognitive.microsoft.com/sts/v1.0/issueToken`,
            null,
            { headers: { "Ocp-Apim-Subscription-Key": apiKey } }
        );
        res.json({ token: tokenRes.data, region });
    } catch (err) {
        console.error("Failed to generate Azure token:", err.response?.data || err.message);
        res.status(500).json({ error: "Failed to generate token" });
    }
});

// ------------------ start heygen ----------------------

// === HEYGEN: Streaming token (client -> server -> HeyGen) ===
app.get("/api/heygen/streaming-token", async (req, res) => {
    try {
        const r = await heygen.post("/v1/streaming.create_token");
        const token = r.data?.data?.token;
        if (!token) return res.status(502).json({ error: "No token from HeyGen" });
        res.json({ token });
    } catch (e) {
        const status = e?.response?.status || 500;
        return res.status(status).json({ error: "HeyGen token error", details: e?.response?.data || e.message });
    }
});

// === HEYGEN: Lista streaming avatars (per UI di scelta) ===
app.get("/api/heygen/streaming/avatars", async (req, res) => {
    try {
        const r = await heygen.get("/v1/streaming/avatar.list");
        res.json(r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen avatars error", details: e?.response?.data || e.message });
    }
});

// === HEYGEN: Lista voices (v2) ===
app.get("/api/heygen/voices", async (req, res) => {
    try {
        const r = await heygen.get("/v2/voices");
        res.json(r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen voices error", details: e?.response?.data || e.message });
    }
});

// === HEYGEN: Lista avatars (v2) ===
app.get("/api/heygen/avatars", async (req, res) => {
    try {
        const r = await heygen.get("/v2/avatars");
        res.json(r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen avatars v2 error", details: e?.response?.data || e.message });
    }
});

// === HEYGEN: Generazione video (v2) ===
app.post("/api/heygen/video/generate", async (req, res) => {
    const {
        avatar_id = process.env.HEYGEN_DEFAULT_AVATAR_ID || "default",
        voice_id = process.env.HEYGEN_DEFAULT_VOICE_ID,
        text,
        language = process.env.HEYGEN_DEFAULT_LANG || "fr",
        ratio = "16:9",        // opzionale: "16:9" | "9:16" | "1:1"
        background = "green_screen" // o "transparent" (dipende dal piano) / "office" / ecc.
    } = req.body || {};

    if (!text) return res.status(400).json({ error: "text is required" });

    try {
        const r = await heygen.post("/v2/video/generate", {
            avatar_id,
            voice_id,
            text,
            language,
            ratio,
            background
        });
        // risposta contiene data.video_id
        res.json(r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen video generate error", details: e?.response?.data || e.message });
    }
});

// === HEYGEN: LiveKit v2 endpoints (proxy sicuro) ===
app.post("/api/heygen/streaming/new", async (req, res) => {
    try {
        const { avatar_id, voice_id, language = "fr", version = "v2" } = req.body || {};
        const r = await heygen.post("/v1/streaming.new", { version, avatar_id, voice_id, language, background: "transparent" });
        res.json(r.data?.data || r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen streaming.new error", details: e?.response?.data || e.message });
    }
});

app.post("/api/heygen/streaming/start", async (req, res) => {
    try {
        const { session_id } = req.body || {};
        const r = await heygen.post("/v1/streaming.start", { session_id });
        res.json(r.data?.data || r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen streaming.start error", details: e?.response?.data || e.message });
    }
});

app.post("/api/heygen/streaming/task", async (req, res) => {
    try {
        const { session_id, text, task_type = "talk" } = req.body || {};
        const r = await heygen.post("/v1/streaming.task", { session_id, text, task_type });
        res.json(r.data?.data || r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen streaming.task error", details: e?.response?.data || e.message });
    }
});

app.post("/api/heygen/streaming/stop", async (req, res) => {
    try {
        const { session_id } = req.body || {};
        const r = await heygen.post("/v1/streaming.stop", { session_id });
        res.json(r.data?.data || r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen streaming.stop error", details: e?.response?.data || e.message });
    }
});

// === HEYGEN: Stato video (polling) ===
app.get("/api/heygen/video/status", async (req, res) => {
    const { video_id } = req.query;
    if (!video_id) return res.status(400).json({ error: "video_id is required" });
    try {
        // endpoint status
        const r = await heygen.get("/v1/video_status.get", { params: { video_id } });
        res.json(r.data);
    } catch (e) {
        res.status(e?.response?.status || 500).json({ error: "HeyGen video status error", details: e?.response?.data || e.message });
    }
});

// ------------------ end heygen ----------------------

// Start server
/*
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
*/
const server = http.createServer(app);

const wss = new WebSocket.Server({ noServer: true });
const wssEl = new WebSocket.Server({ noServer: true });
const wssAzureTTS = new WebSocket.Server({ noServer: true });

// ✅ Router unico per gli upgrade WS
server.on("upgrade", (req, socket, head) => {
    let pathname = "/";
    try {
        const url = new URL(req.url, `http://${req.headers.host}`);
        pathname = url.pathname;
    } catch { }

    if (pathname === "/api/fullCustomRealtimeAzureOpenAI") {
        wss.handleUpgrade(req, socket, head, (ws) => {
            wss.emit("connection", ws, req);
        });
        return;
    }

    if (pathname === "/api/elevenlabs-tts") {
        wssEl.handleUpgrade(req, socket, head, (ws) => {
            wssEl.emit("connection", ws, req);
        });
        return;
    }

    if (pathname === "/api/azure-tts-ws") {                // 👈 nuovo
        wssAzureTTS.handleUpgrade(req, socket, head, (ws) => {
            wssAzureTTS.emit("connection", ws, req);
        });
        return;
    }

    // path sconosciuto → chiudi
    socket.destroy();
});


server.listen(port, () => {
    console.log(`HTTP+WS server on http://localhost:${port}`);
});

// helper: decode base64 el_vs e normalizza i valori
function parseElVS(b64) {
    if (!b64) return null;
    try {
        const raw = JSON.parse(Buffer.from(b64, "base64").toString("utf8"));
        const clamp01 = v => Math.max(0, Math.min(1, Number(v)));
        const out = {};
        if (raw.stability !== undefined) out.stability = clamp01(raw.stability);
        if (raw.similarity_boost !== undefined) out.similarity_boost = clamp01(raw.similarity_boost);
        if (raw.style !== undefined) out.style = clamp01(raw.style);
        if (raw.use_speaker_boost !== undefined) out.use_speaker_boost = !!raw.use_speaker_boost;
        return Object.keys(out).length ? out : null;
    } catch (e) {
        console.warn("[Realtime] invalid el_vs:", e?.message || e);
        return null;
    }
}

// util per SSML con stile, rate e pitch
function buildSSMLv2({ text, voice, style, styleDegree, rate, pitch }) {
    const v = voice || "fr-FR-RemyMultilingualNeural";
    const locale = v.substring(0, 5);
    const safe = escapeXml(text || "");
    const prosody = (rate || pitch)
        ? `<prosody${rate ? ` rate="${rate}"` : ""}${pitch ? ` pitch="${pitch}"` : ""}>${safe}</prosody>`
        : safe;
    const body = style
        ? `<mstts:express-as style="${style}"${styleDegree ? ` styledegree="${styleDegree}"` : ""}>${prosody}</mstts:express-as>`
        : prosody;

    return `
<speak version="1.0" xml:lang="${locale}" xmlns:mstts="https://www.w3.org/2001/mstts">
  <voice name="${v}">${body}</voice>
</speak>`.trim();
}

wssAzureTTS.on("connection", (ws, req) => {
    if (!AZ_TTS_KEY || !AZ_TTS_REGION) {
        try { ws.close(1011, "Missing Azure Speech env vars"); } catch { }
        return;
    }

    // parametri opzionali in query: ?voice=fr-FR-RemyMultilingualNeural
    const urlObj = new URL(req.url, `http://${req.headers.host}`);
    const qVoice = urlObj.searchParams.get("voice");

    // istanzio un sintetizzatore per QUESTA connessione, formato PCM 24k 16bit mono
    const speechConfig = sdk.SpeechConfig.fromSubscription(AZ_TTS_KEY, AZ_TTS_REGION);
    speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm;
    const synth = new sdk.SpeechSynthesizer(speechConfig);

    let busy = false;
    const queue = [];
    let closed = false;

    const doNext = () => {
        if (closed || busy || queue.length === 0) return;
        busy = true;
        const job = queue.shift(); // { ssml }
        let sentAny = false;

        // stream chunk-by-chunk
        synth.synthesizing = (_s, e) => {
            const bytes = e?.result?.audioData;
            if (bytes && bytes.byteLength) {
                sentAny = true;
                ws.send(JSON.stringify({ audio: Buffer.from(bytes).toString("base64") }));
            }
        };
        synth.synthesisCompleted = () => {
            ws.send(JSON.stringify({ done: true }));
            busy = false;
            // IMPORTANT: rimuovi i listener per il prossimo job
            synth.synthesizing = undefined;
            synth.synthesisCompleted = undefined;
            synth.canceled = undefined;
            doNext();
        };
        synth.canceled = (_s, e) => {
            ws.send(JSON.stringify({ error: e?.errorDetails || "synthesis canceled" }));
            busy = false;
            synth.synthesizing = undefined;
            synth.synthesisCompleted = undefined;
            synth.canceled = undefined;
            doNext();
        };

        synth.speakSsmlAsync(job.ssml, () => { }, (err) => {
            ws.send(JSON.stringify({ error: String(err) }));
            busy = false;
            synth.synthesizing = undefined;
            synth.synthesisCompleted = undefined;
            synth.canceled = undefined;
            doNext();
        });
    };

    ws.on("message", (data) => {
        let msg; try { msg = JSON.parse(data.toString()); } catch { return; }
        if (typeof msg.text === "string") {
            const ssml = buildSSMLv2({
                text: msg.text,
                voice: msg.voice || qVoice || "fr-FR-RemyMultilingualNeural",
                style: msg.style,
                styleDegree: msg.styleDegree,
                rate: msg.rate,
                pitch: msg.pitch
            });
            queue.push({ ssml });
            doNext();
        } else if (msg.flush) {
            // niente da fare specifico con Azure; i job vanno a fine con Completed
        }
    });

    const cleanup = () => {
        closed = true;
        try { synth.close(); } catch { }
    };
    ws.on("close", cleanup);
    ws.on("error", cleanup);
});


// === WebSocket bridge per Azure Realtime ===
// Unico endpoint WS: /api/fullCustomRealtimeAzureOpenAI
//const wss = new WebSocket.Server({ server, path: "/api/fullCustomRealtimeAzureOpenAI" });

//const wssEl = new WebSocket.Server({ server, path: "/api/elevenlabs-tts" });

wssEl.on("connection", (client, req) => {
    const urlObj = new URL(req.url, `http://${req.headers.host}`);
    const clean = s => (s || "").replace(/[^A-Za-z0-9_\-]/g, "").slice(0, 64);

    const elVoiceId = clean(urlObj.searchParams.get("el_voice")) || process.env.ELEVENLABS_DEFAULT_VOICE_ID;
    const elModelId = clean(urlObj.searchParams.get("el_model")) || process.env.ELEVENLABS_MODEL_ID || "eleven_flash_v2_5";
    const voiceSettings = parseElVS(urlObj.searchParams.get("el_vs"));

    let el;
    try {
        el = openElevenLabsWs({ voiceId: elVoiceId, modelId: elModelId, voiceSettings });
    } catch (e) {
        try { client.close(1011, e.message); } catch { }
        return;
    }

    // ElevenLabs -> Client
    el.ws.on("message", (m) => {
        try {
            const d = JSON.parse(m.toString("utf8"));
            if (d.audio) client.send(JSON.stringify({ audio: d.audio }));   // base64 PCM 24k
            if (d.isFinal) client.send(JSON.stringify({ done: true }));
        } catch { }
    });
    el.ws.on("close", () => { try { client.close(); } catch { } });
    el.ws.on("error", (err) => { try { client.close(1011, err?.message || "eleven error"); } catch { } });

    // Client -> ElevenLabs
    client.on("message", (data) => {
        let msg; try { msg = JSON.parse(data.toString()); } catch { return; }
        if (typeof msg.text === "string") el.sendText(msg.text);
        if (msg.flush) el.flushAndClose();
    });

    client.on("close", () => {
        try { el.flushAndClose(); el.ws.close(); } catch { }
    });
});


// ------- AGGIUNTO 25/08 -----------
function openElevenLabsWs({
    voiceId,
    modelId = process.env.ELEVENLABS_MODEL_ID || "eleven_flash_v2_5",
    voiceSettings = null
}) {
    const vId = voiceId || process.env.ELEVENLABS_DEFAULT_VOICE_ID;
    if (!vId) throw new Error("Missing ELEVENLABS voiceId (pass ?el_voice=... or set ELEVENLABS_DEFAULT_VOICE_ID)");
    const apiKey = process.env.ELEVENLAB_API_KEY;
    if (!apiKey) throw new Error("Missing ELEVENLAB_API_KEY");

    const url = `wss://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(vId)}/stream-input?model_id=${encodeURIComponent(modelId)}&output_format=pcm_24000`;

    const elWs = new WebSocket(url, { perMessageDeflate: false, headers: { "xi-api-key": apiKey } });
    let ready = false;
    const queue = [];
    let wantFlush = false; // <-- NOVITÀ

    elWs.on("open", () => {
        ready = true;
        const initMsg = {
            text: " ",
            voice_settings: voiceSettings || undefined,
            generation_config: { chunk_length_schedule: [120, 160, 250, 290] }
        };
        try { elWs.send(JSON.stringify(initMsg)); } catch { }

        // invia i testi in coda nell'ordine
        for (const t of queue.splice(0)) {
            try { elWs.send(JSON.stringify({ text: t })); } catch { }
        }

        // se nel frattempo era arrivato flush → invialo ORA
        if (wantFlush) {
            try {
                elWs.send(JSON.stringify({ flush: true }));
                elWs.send(JSON.stringify({ text: "" }));
            } catch { }
            wantFlush = false;
        }
    });

    return {
        ws: elWs,
        isReady: () => ready && elWs.readyState === WebSocket.OPEN,
        sendText: (t) => {
            if (!t) return;
            if (ready && elWs.readyState === WebSocket.OPEN) {
                try { elWs.send(JSON.stringify({ text: t })); } catch { }
            } else {
                queue.push(t);
            }
        },
        flushAndClose: () => {
            if (elWs.readyState === WebSocket.OPEN && ready) {
                try {
                    elWs.send(JSON.stringify({ flush: true }));
                    elWs.send(JSON.stringify({ text: "" }));
                } catch { }
            } else {
                // non ancora open → ricorda il flush
                wantFlush = true;
            }
        }
    };
}
// ---------------------- //


wss.on("connection", (clientWs, req) => {
    const endpointHost = (AZ_ENDPOINT || "").replace(/^https?:\/\//, "").replace(/\/+$/, "");
    if (!endpointHost || !AZ_KEY || !AZ_DEPLOY) {
        console.error("[Realtime] Missing env: endpoint/key/deployment");
        try { clientWs.close(1011, "Missing Azure Realtime env vars"); } catch { }
        return;
    }

    // ✅ LEGGI LA VOCE DALLA QUERY (?voice=alloy/echo/fable/onyx/nova/shimmer)
    const urlObj = new URL(req.url, `http://${req.headers.host}`);
    const qVoice = (urlObj.searchParams.get("voice") || "").toLowerCase();
    const ALLOWED_RT_VOICES = new Set(["alloy", "echo", "fable", "onyx", "nova", "shimmer"]);
    const initialVoice = ALLOWED_RT_VOICES.has(qVoice)
        ? qVoice
        : (process.env.AZURE_REALTIME_VOICE || "echo");

    const ttsProvider = (urlObj.searchParams.get("tts") || "azure").toLowerCase(); // ------- AGGIUNTO 25/08 -----------

    // prendili dal client, con sanitizzazione
    const clean = (s) => (s || "").replace(/[^A-Za-z0-9_\-]/g, "").slice(0, 64); // ------- AGGIUNTO 25/08 -----------
    let elVoiceId = clean(urlObj.searchParams.get("el_voice")) || process.env.ELEVENLABS_DEFAULT_VOICE_ID; // ------- AGGIUNTO 25/08 -----------
    let elModelId = clean(urlObj.searchParams.get("el_model")) || process.env.ELEVENLABS_MODEL_ID || "eleven_flash_v2_5"; // ------- AGGIUNTO 25/08 -----------
    const qDeploy = clean(urlObj.searchParams.get("az_deploy"));
    const DEPLOY = qDeploy || AZ_DEPLOY;

    const elVsB64 = urlObj.searchParams.get("el_vs");
    const voiceSettings = parseElVS(elVsB64);

    // opzionale: whitelist dei modelli consentiti
    const ALLOWED_EL_MODELS = new Set([ // ------- AGGIUNTO 25/08 -----------
        "eleven_flash_v2_5", // ------- AGGIUNTO 25/08 -----------
        "eleven_turbo_v2_5", // ------- AGGIUNTO 25/08 -----------
        "eleven_multilingual_v2" // ------- AGGIUNTO 25/08 -----------
    ]); // ------- AGGIUNTO 25/08 -----------
    if (!ALLOWED_EL_MODELS.has(elModelId)) elModelId = "eleven_flash_v2_5"; // ------- AGGIUNTO 25/08 -----------

    // Costruisco l’URL WS verso Azure. Aggiungo api-key anche in query per massima compatibilità.
    const azureUrl =
        `wss://${endpointHost}/openai/realtime` +
        `?api-version=${encodeURIComponent(AZ_VER)}` +
        `&deployment=${encodeURIComponent(DEPLOY)}` +
        `&api-key=${encodeURIComponent(AZ_KEY)}`;

    console.log("[Realtime] Dialing Azure WS:", azureUrl.replace(/api-key=[^&]+/, "api-key=***"));

    // Connessione WS verso Azure
    const azureWs = new WebSocket(azureUrl, {
        // niente header qui: la chiave è già in query; lasciamo handshake il più compatibile possibile
        perMessageDeflate: false,
    });

    let azureReady = false;
    const pendingFromClient = []; // coda messaggi client → Azure, finché Azure non è OPEN
    let heartbeatTimer = null;

    const closeBoth = (code = 1000, reason = "") => {
        try { clientWs.close(code, reason); } catch { }
        try { azureWs.close(code, reason); } catch { }
        if (heartbeatTimer) { clearInterval(heartbeatTimer); heartbeatTimer = null; }
    };

    // Keepalive per evitare idle timeout sugli LB
    heartbeatTimer = setInterval(() => {
        try { if (clientWs.readyState === WebSocket.OPEN) clientWs.ping(); } catch { }
        try { if (azureWs.readyState === WebSocket.OPEN) azureWs.ping(); } catch { }
    }, 15000);

    // AGGIUNTO 25/08
    const elevenByResp = new Map();

    const cleanupEL = () => {
        if (elevenByResp.size === 0) return;
        for (const [, el] of elevenByResp) {
            try { el.flushAndClose(); } catch { }
            try { el.ws.close(); } catch { }
        }
        elevenByResp.clear();
    };
    //---------------


    // Quando Azure è OPEN, invio session.update e svuoto la coda
    azureWs.on("open", () => {
        azureReady = true;
        console.log("[Realtime] Azure WS OPEN");

        const sessionUpdate = {
            type: "session.update",
            session: {
                // Impostazioni base: voce + formati audio + trascrizione
                voice: initialVoice,
                modalities: (ttsProvider === "azure") ? ["text", "audio"] : ["text"], // AGGIUNTO 25/08 // ["text", "audio"],
                input_audio_transcription: { model: "whisper-1" },
                input_audio_format: "pcm16",
                ...(ttsProvider === "azure" ? { output_audio_format: "pcm16" } : {}), // AGGIUNTO 25/08 //output_audio_format: "pcm16",
                // VAD lato server: se mandi audio, creerà automaticamente una response al silenzio
                turn_detection: {
                    type: "server_vad",
                    threshold: 0.5,
                    prefix_padding_ms: 300,
                    silence_duration_ms: 200,
                    create_response: true
                },
                // Istruzioni di default (puoi cambiarle dal client con un altro session.update)
                instructions: "Reponds en francais"
            }
        };

        try { azureWs.send(JSON.stringify(sessionUpdate)); }
        catch (e) { console.error("[Realtime] session.update send error:", e); }

        // Svuota la coda dei messaggi arrivati dal client mentre Azure era in CONNECTING
        while (pendingFromClient.length) {
            const frame = pendingFromClient.shift();
            try { azureWs.send(frame); }
            catch (e) { console.error("[Realtime] flush->Azure send error:", e); }
        }
    });

    // Azure → Client (forward 1:1, preservando binario/testo)
    /*
    azureWs.on("message", (data, isBinary) => {
        if (clientWs.readyState !== WebSocket.OPEN) return;
        try {
            clientWs.send(data, { binary: isBinary });
        } catch (e) {
            console.error("[Realtime] forward Azure->Client error:", e);
        }
    });*/

    // ---------- AGGIUNTO 25/08 ------------------
    azureWs.on("message", (data, isBinary) => {
        if (clientWs.readyState !== WebSocket.OPEN) return;

        // Se NON usi ElevenLabs, inoltra tutto com'è (comportamento attuale)
        if (ttsProvider === "azure") {
            try { clientWs.send(data, { binary: isBinary }); } catch (e) {
                console.error("[Realtime] forward Azure->Client error:", e);
            }
            return;
        }

        // ---------- ttsProvider === "elevenlab" ----------
        // Azure: tieni SOLO gli eventi testuali, droppa gli audio (li farà ElevenLabs)

        // Se frame binario, lo ignoriamo (Azure audio binario non serve)
        if (isBinary) return;

        let txt;
        try { txt = data.toString("utf8"); } catch { return; }

        let msg;
        try { msg = JSON.parse(txt); } catch {
            // Se non è JSON, avanti (ma in pratica Azure manda JSON)
            return;
        }

        const t = msg.type || "";

        // 0) Inoltra SEMPRE gli eventi testuali al client (così il tuo typer funziona)
        //    … ma se sono eventi audio Azure, scartali.
        if (/^response\.(output_audio|audio)\./.test(t)) {
            // DROP audio Azure
        } else {
            try { clientWs.send(JSON.stringify(msg)); } catch { }
        }

        // 1) response.created → apri una WS ElevenLabs per questo responseId e pipe audio->client
        if (t === "response.created" && (msg.response?.id || msg.id)) {
            const rid = msg.response?.id || msg.id;

            // evita doppioni
            if (!elevenByResp.has(rid)) {
                const el = openElevenLabsWs({
                    voiceId: elVoiceId,
                    modelId: elModelId,
                    voiceSettings   // 👈 passa le impostazioni dal client
                });

                // Audio da ElevenLabs → re-impacchettato come eventi Azure-like per il frontend
                el.ws.on("message", (m) => {
                    try {
                        const d = JSON.parse(m.toString("utf8"));
                        if (d.audio) {
                            const out = { type: "response.output_audio.delta", response_id: rid, delta: d.audio };
                            clientWs.send(JSON.stringify(out));
                        }
                        if (d.isFinal) {
                            const done = { type: "response.output_audio.done", response_id: rid };
                            clientWs.send(JSON.stringify(done));
                        }
                    } catch {
                        // ignora frame non JSON
                    }
                });

                el.ws.on("close", () => { elevenByResp.delete(rid); });
                el.ws.on("error", () => { elevenByResp.delete(rid); });

                elevenByResp.set(rid, el);
            }
            return;
        }

        // 2) Delta di testo → invialo anche a ElevenLabs (così genera l’audio)
        if (
            t === "response.output_text.delta" ||
            t === "response.text.delta" ||
            t === "response.delta" ||
            t === "message.delta"
        ) {
            const rid =
                msg.response_id ||
                msg.response?.id ||
                msg.id;

            const chunk =
                (typeof msg.delta === "string" && msg.delta) ||
                (msg.output_text && typeof msg.output_text.delta === "string" && msg.output_text.delta) ||
                (msg.text && typeof msg.text.delta === "string" && msg.text.delta) ||
                "";

            if (rid && chunk) {
                const el = elevenByResp.get(rid);
                if (el) el.sendText(chunk);
            }
            return;
        }

        // 3) Fine turno → flush & close ElevenLabs (spinge gli ultimi frame)
        if (t === "response.completed" || t === "response.done") {
            const rid = msg.response_id || msg.response?.id || msg.id;
            const el = rid && elevenByResp.get(rid);
            if (el) el.flushAndClose();
            return;
        }
    });
    // ---------- RIATTIVA azureWs.on("message") che si trova subito sopra ------------------

    azureWs.on("error", (err) => {
        console.error("[Realtime] Azure WS ERROR:", err?.message || err);
        cleanupEL(); // AGGIUNTO 25/08
        closeBoth(1011, "azure error");
    });

    azureWs.on("close", (code, reason) => {
        console.warn("[Realtime] Azure WS CLOSED:", code, reason?.toString?.() || "");
        cleanupEL();  // AGGIUNTO 25/08
        closeBoth(code || 1000, "azure closed");
    });

    // Client → Azure (se Azure non è ancora OPEN, accodo)
    clientWs.on("message", (data, isBinary) => {
        if (!azureReady) {
            // Salvo come stringa o buffer mantenendo il tipo
            pendingFromClient.push(isBinary ? Buffer.from(data) : (typeof data === "string" ? data : data.toString()));
            return;
        }
        if (azureWs.readyState !== WebSocket.OPEN) return;
        try {
            azureWs.send(data, { binary: isBinary });
        } catch (e) {
            console.error("[Realtime] forward Client->Azure error:", e);
        }
    });

    clientWs.on("error", (err) => {
        console.error("[Realtime] Client WS ERROR:", err?.message || err);
        cleanupEL(); // AGGIUNTO 25/08
        closeBoth(1011, "client error");
    });

    clientWs.on("close", (code, reason) => {
        console.warn("[Realtime] Client WS CLOSED:", code, reason?.toString?.() || "");
        cleanupEL(); // AGGIUNTO 25/08
        closeBoth(code || 1000, "client closed");
    });

    // Safety: se Azure non apre entro 15s, chiudo tutto
    setTimeout(() => {
        if (!azureReady) {
            console.error("[Realtime] Timeout opening Azure WS");
            cleanupEL(); // AGGIUNTO 25/08
            closeBoth(1013, "azure connect timeout");
        }
    }, 15000);
});