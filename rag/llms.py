"""Three free LLM providers via plain REST.

Model names are AUTO-DISCOVERED from each provider's "list models" API, so the
app keeps working when a model is retired. If you set GROQ_MODEL / GEMINI_MODEL /
OPENROUTER_MODEL in .env they are tried first; on 404/400 we fall back to discovery.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import requests

TIMEOUT = 90
_cache = {}  # provider -> list of candidate model ids


def _check(r):
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r


# ---------------- model discovery ----------------
def _groq_models(key):
    r = _check(requests.get("https://api.groq.com/openai/v1/models",
                            headers={"Authorization": f"Bearer {key}"}, timeout=30))
    ids = [m["id"] for m in r.json()["data"]]
    bad = ("whisper", "tts", "orpheus", "guard", "embed", "compound", "safeguard")
    ids = [i for i in ids if not any(b in i.lower() for b in bad)]
    prefer = ("gpt-oss-120b", "llama-3.3-70b", "gpt-oss-20b", "llama", "qwen", "gemma")
    ids.sort(key=lambda i: next((n for n, p in enumerate(prefer) if p in i.lower()), 99))
    return ids


def _gemini_models(key):
    r = _check(requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                            headers={"x-goog-api-key": key}, timeout=30))
    names = []
    for m in r.json().get("models", []):
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        n = m["name"].replace("models/", "")
        if any(b in n for b in ("embedding", "tts", "image", "imagen", "veo", "aqa",
                                "live", "audio", "robotics", "computer", "learnlm")):
            continue
        names.append(n)
    stable = sorted([n for n in names if "flash" in n and not any(
        x in n for x in ("preview", "exp", "lite", "thinking", "latest"))], reverse=True)
    lite = sorted([n for n in names if "flash" in n and "lite" in n
                   and "preview" not in n and "exp" not in n], reverse=True)
    rest = [n for n in names if n not in stable + lite]
    return stable + lite + rest


def _openrouter_models(key):
    r = _check(requests.get("https://openrouter.ai/api/v1/models", timeout=30))
    ids = [m["id"] for m in r.json()["data"] if m["id"].endswith(":free")]
    prefer = ("gpt-oss", "llama", "gemma", "qwen", "mistral", "deepseek")
    ids.sort(key=lambda i: next((n for n, p in enumerate(prefer) if p in i.lower()), 99))
    return ids


def _candidates(provider, key, env_name, finder):
    if provider not in _cache:
        _cache[provider] = finder(key)
    cands = list(_cache[provider])
    configured = os.getenv(env_name, "").strip()
    return ([configured] if configured else []) + [c for c in cands if c != configured]


# ---------------- calls ----------------
def _openai_style(url, key, model, prompt):
    r = requests.post(url,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": model, "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.2},
                      timeout=TIMEOUT)
    return _check(r).json()["choices"][0]["message"]["content"]


def _gemini_call(key, model, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                      json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=TIMEOUT)
    return _check(r).json()["candidates"][0]["content"]["parts"][0]["text"]


def _try_models(models, call, max_tries=4):
    errors = []
    for m in models[:max_tries]:
        try:
            return f"{call(m)}\n\n_(model: `{m}`)_"
        except Exception as e:
            errors.append(f"{m}: {e}")
    raise RuntimeError(" | ".join(errors))


def ask_groq(prompt):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing in .env")
    models = _candidates("groq", key, "GROQ_MODEL", _groq_models)
    return _try_models(models, lambda m: _openai_style(
        "https://api.groq.com/openai/v1/chat/completions", key, m, prompt))


def ask_gemini(prompt):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in .env")
    models = _candidates("gemini", key, "GEMINI_MODEL", _gemini_models)
    return _try_models(models, lambda m: _gemini_call(key, m, prompt))


def ask_openrouter(prompt):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env")
    models = _candidates("openrouter", key, "OPENROUTER_MODEL", _openrouter_models)
    return _try_models(models, lambda m: _openai_style(
        "https://openrouter.ai/api/v1/chat/completions", key, m, prompt), max_tries=5)


PROVIDERS = {
    "Groq": ask_groq,
    "Google Gemini": ask_gemini,
    "OpenRouter (free)": ask_openrouter,
}


def build_prompt(question, contexts):
    ctx = "\n\n---\n\n".join(f"[Source: {s}]\n{t}" for s, t, _ in contexts)
    return (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I could not find this in the document.\"\n\n"
        f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nANSWER:"
    )


def ask_all(prompt):
    """Run all 3 models in parallel. Returns {name: answer_or_error}."""
    def run(item):
        name, fn = item
        try:
            return name, fn(prompt)
        except Exception as e:
            return name, f"⚠️ Error: {e}"

    with ThreadPoolExecutor(max_workers=3) as ex:
        return dict(ex.map(run, PROVIDERS.items()))