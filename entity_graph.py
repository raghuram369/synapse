"""Entity graph concept extraction and query expansion helpers."""

import re
from typing import List, Tuple


CONCEPT_MAP = {
    # hardware
    "hardware": {
        "category": "hardware",
        "aliases": [
            "hardware", "machine", "pc", "computer", "laptop", "desktop", "workstation", "server",
            "host", "cpu", "gpu", "processor", "ram", "memory", "storage", "ssd", "hdd",
            "nvme", "motherboard", "chip", "chipset", "network card", "peripheral", "router",
            "switch", "modem", "battery", "keyboard", "mouse", "display", "monitor", "screen",
        ],
    },
    "ai chip": {
        "category": "hardware",
        "aliases": ["ai chip", "neural chip", "accelerator", "tpu", "npu", "inference chip"],
    },
    "compute server": {
        "category": "hardware",
        "aliases": ["compute", "compute server", "gpu server", "blade", "rack", "node"],
    },
    "mobile device": {
        "category": "hardware",
        "aliases": ["mobile", "phone", "tablet", "smartphone", "iphone", "android phone"],
    },
    "mac mini": {
        "category": "hardware",
        "aliases": ["mac mini", "macmini", "mac mini m4", "mac mini m4 pro", "mac mini m3", "mac mini m2", "macbook", "macbook pro", "macbook air"],
    },
    "intel cpu": {
        "category": "hardware",
        "aliases": ["intel", "i5", "i7", "i9", "xeon", "core i5", "core i7"],
    },
    "amd cpu": {
        "category": "hardware",
        "aliases": ["amd", "ryzen", "epyc", "threadripper"],
    },
    "arm cpu": {
        "category": "hardware",
        "aliases": ["arm", "m1", "m2", "m3", "m4", "m1 pro", "m2 pro", "m4 pro", "apple silicon"],
    },

    # software
    "software": {
        "category": "software",
        "aliases": ["software", "app", "application", "platform", "stack", "package", "program", "binary", "firmware"],
    },
    "operating_system": {
        "category": "software",
        "aliases": ["operating system", "os", "linux", "ubuntu", "debian", "arch", "macos", "windows", "windows server", "ios", "android", "unix"],
    },
    "versioning": {
        "category": "software",
        "aliases": ["version", "versioning", "release", "tag", "patch", "rollback", "changelog", "semver", "rc"],
    },
    "database": {
        "category": "software",
        "aliases": ["database", "sql", "redis", "postgres", "postgresql", "sqlite", "mysql", "mariadb", "mongodb", "nosql", "timescale"],
    },
    "containerization": {
        "category": "software",
        "aliases": ["container", "containerization", "docker", "podman", "lxc", "virtual machine", "vm", "kvm", "hypervisor"],
    },
    "shell": {
        "category": "software",
        "aliases": ["shell", "bash", "zsh", "fish", "powershell", "command line", "terminal", "cli", "prompt"],
    },
    "api": {
        "category": "software",
        "aliases": ["api", "rest", "grpc", "graphql", "webhook", "endpoint", "http", "https"],
    },
    "runtime": {
        "category": "software",
        "aliases": ["runtime", "interpreter", "engine", "node", "bun", "deno", "python runtime"],
    },
    "framework": {
        "category": "software",
        "aliases": ["framework", "react", "vue", "angular", "svelte", "django", "flask", "fastapi", "spring", "nextjs", "nuxt"],
    },
    "version_control": {
        "category": "software",
        "aliases": ["version control", "git", "github", "gitlab", "svn", "mercurial", "bitbucket", "commit"],
    },
    "package_manager": {
        "category": "software",
        "aliases": ["pip", "npm", "yarn", "pnpm", "conda", "poetry", "cargo", "brew", "apt", "homebrew"],
    },

    # people
    "person": {
        "category": "people",
        "aliases": ["person", "people", "human", "user", "client", "partner", "individual", "manager", "director"],
    },
    "engineer": {
        "category": "people",
        "aliases": ["engineer", "engineering", "staff engineer", "senior engineer", "dev", "developer", "frontend engineer", "backend engineer"],
    },
    "maintainer": {
        "category": "people",
        "aliases": ["maintainer", "maintainers", "reviewer", "admin", "administrator", "sysadmin", "site reliability", "sre", "platform engineer"],
    },
    "alice": {
        "category": "people",
        "aliases": ["alice", "alice developer"],
    },
    "bob": {
        "category": "people",
        "aliases": ["bob", "bob smith"],
    },
    "charlie": {
        "category": "people",
        "aliases": ["charlie", "charlie chen"],
    },
    "dave": {
        "category": "people",
        "aliases": ["dave", "david"],
    },
    "erin": {
        "category": "people",
        "aliases": ["erin", "eric"],
    },
    "frank": {
        "category": "people",
        "aliases": ["frank", "frances"],
    },
    "project_lead": {
        "category": "people",
        "aliases": ["lead", "project lead", "tech lead", "owner", "owner"],
    },

    # projects
    "project": {
        "category": "projects",
        "aliases": ["project", "initiative", "initiative", "roadmap", "plan", "plan", "program"],
    },
    "synapse": {
        "category": "projects",
        "aliases": ["synapse", "synapse project", "memory project"],
    },
    "recall_project": {
        "category": "projects",
        "aliases": ["recall", "recall system", "graph recall", "search recall"],
    },
    "phoenix": {
        "category": "projects",
        "aliases": ["phoenix", "phoenix project"],
    },

    # scheduling/meetings
    "meeting": {
        "category": "scheduling",
        "aliases": ["meeting", "standup", "sync", "1on1", "one on one", "retro", "retrospective",
                     "sprint planning", "demo", "review meeting", "calendar", "schedule", "scheduled"],
    },

    # performance/benchmarks
    "performance": {
        "category": "engineering",
        "aliases": ["performance", "benchmark", "benchmarks", "latency", "throughput", "p99", "p95",
                     "optimization", "optimize", "profiling", "flame graph", "perf"],
    },

    # timezone/location
    "timezone": {
        "category": "preferences",
        "aliases": ["timezone", "time zone", "tz", "utc", "gmt", "cst", "est", "pst"],
    },

    # decisions
    "decision": {
        "category": "engineering",
        "aliases": ["decision", "decisions", "decided", "chose", "chosen", "picked", "selected",
                     "tech decision", "architecture decision", "adr"],
    },
    "compiler_project": {
        "category": "projects",
        "aliases": ["compiler", "compiler project", "bytecode"],
    },
    "infrastructure_project": {
        "category": "projects",
        "aliases": ["platform", "infra", "infrastructure", "migration"],
    },

    # tools
    "tool": {
        "category": "tools",
        "aliases": ["tool", "utilities", "utility", "script", "utility script", "assistant", "plugin", "extension"],
    },
    "docker": {
        "category": "tools",
        "aliases": ["docker", "dockerfile", "compose", "docker compose", "docker image"],
    },
    "kubernetes": {
        "category": "tools",
        "aliases": ["kubernetes", "k8s", "helm", "kubectl", "minikube", "kind"],
    },
    "terraform": {
        "category": "tools",
        "aliases": ["terraform", "tf", "tform", "terraformer"],
    },
    "ansible": {
        "category": "tools",
        "aliases": ["ansible", "playbook", "inventory", "ad-hoc"],
    },
    "pytest": {
        "category": "tools",
        "aliases": ["pytest", "unittest", "testing", "mock", "fixture"],
    },
    "ruff": {
        "category": "tools",
        "aliases": ["ruff", "lint", "linter", "formatter", "formatter"],
    },
    "make": {
        "category": "tools",
        "aliases": ["make", "makefile", "cmake", "bazel", "ninja", "gradle", "maven"],
    },
    "vscode": {
        "category": "tools",
        "aliases": ["vscode", "visual studio code", "code editor", "editor", "intellij", "pycharm", "neovim", "vim"],
    },
    "jupyter": {
        "category": "tools",
        "aliases": ["jupyter", "jupyter notebook", "jupyterlab", "notebook"],
    },
    "curl": {
        "category": "tools",
        "aliases": ["curl", "httpie", "wget", "postman", "insomnia"],
    },
    "ollama": {
        "category": "tools",
        "aliases": ["ollama", "llm tool", "local model runner"],
    },

    # AI / ML
    "ai_ml": {
        "category": "AI/ML",
        "aliases": [
            "ai", "artificial intelligence", "ml", "machine learning", "deep learning", "neural",
            "transformer", "embedding", "embedding model", "rag", "large language model", "llm", "fine tuning",
            "prompt", "chatgpt", "claude", "gpt", "llama", "mistral", "qwen", "ollama", "openai", "anthropic",
        ],
    },
    "model": {
        "category": "AI/ML",
        "aliases": ["model", "checkpoint", "weights", "pretrained model", "foundation model", "tokenizer", "bert", "bert model", "clip"],
    },
    "vector_search": {
        "category": "AI/ML",
        "aliases": ["vector search", "vector db", "vector database", "faiss", "qdrant", "weaviate", "pinecone", "chroma"],
    },
    "training": {
        "category": "AI/ML",
        "aliases": ["training", "train", "fine tune", "finetune", "sft", "rlhf", "inference", "serving"],
    },
    "mlops": {
        "category": "AI/ML",
        "aliases": ["mlops", "ml ops", "pipeline", "experiment tracking", "artifact", "wandb", "mlflow", "model registry"],
    },
    "computer_vision": {
        "category": "AI/ML",
        "aliases": ["computer vision", "cv", "image recognition", "object detection", "segmentation", "vision"],
    },
    "nlp": {
        "category": "AI/ML",
        "aliases": ["nlp", "natural language processing", "text classification", "token", "tokenization"],
    },
    "recommendation": {
        "category": "AI/ML",
        "aliases": ["recommendation", "recommender", "rank", "rerank", "retrieval", "search"],
    },

    # infrastructure
    "infrastructure": {
        "category": "infrastructure",
        "aliases": ["infrastructure", "platform", "system", "fleet", "architecture", "backend", "frontend"],
    },
    "cloud": {
        "category": "infrastructure",
        "aliases": ["cloud", "aws", "azure", "gcp", "heroku", "digitalocean", "linode", "oci", "vpc"],
    },
    "network": {
        "category": "infrastructure",
        "aliases": ["network", "vpn", "dns", "cdn", "load balancer", "reverse proxy", "proxy", "firewall"],
    },
    "observability": {
        "category": "infrastructure",
        "aliases": ["observability", "metrics", "monitoring", "logging", "tracing", "otel", "prometheus", "grafana", "sentry"],
    },
    "ci_cd": {
        "category": "infrastructure",
        "aliases": ["ci", "cd", "ci/cd", "pipeline", "github actions", "gitlab ci", "jenkins", "travis", "circleci", "argo"],
    },
    "queueing": {
        "category": "infrastructure",
        "aliases": ["queue", "rabbitmq", "kafka", "sns", "sqs", "bull", "redis queue", "celery"],
    },
    "cdn": {
        "category": "infrastructure",
        "aliases": ["cdn", "cloudfront", "cloudflare", "akamai"],
    },
    "serverless": {
        "category": "infrastructure",
        "aliases": ["serverless", "lambda", "functions", "edge", "cdn edge"],
    },

    # files/paths
    "path": {
        "category": "files/paths",
        "aliases": ["path", "directory", "dir", "folder", "pathname", "filepath", "file path", "mount"],
    },
    "file": {
        "category": "files/paths",
        "aliases": ["file", "files", "document", "artifact", "report", "record"],
    },
    "source_file": {
        "category": "files/paths",
        "aliases": ["source file", "src", "source", ".py", ".ts", ".js", ".md", ".toml", ".yaml", ".yml", ".json", ".cfg"],
    },
    "config": {
        "category": "files/paths",
        "aliases": ["config", "configuration", ".env", "settings", "config file", "ini", "properties", "properties file"],
    },
    "repository": {
        "category": "files/paths",
        "aliases": ["repo", "repository", "git repo", "git repository", "remote", "origin", "branch", "commit", "pull request"],
    },
    "log": {
        "category": "files/paths",
        "aliases": ["log", "logs", "log file", "trace", "stderr", "stdout", "event log"],
    },
    "script": {
        "category": "files/paths",
        "aliases": ["script", "automation", "batch", "entrypoint", "startup", "shell script", "build script"],
    },
    "temp_file": {
        "category": "files/paths",
        "aliases": ["tmp", "temp", "temporary", "/tmp", "cache", "scratch"],
    },

    # programming languages
    "python": {
        "category": "programming languages",
        "aliases": ["python", "py", "cpython", "pypy", "pip"],
    },
    "javascript": {
        "category": "programming languages",
        "aliases": ["javascript", "js", "nodejs", "es", "ecmascript", "node"],
    },
    "typescript": {
        "category": "programming languages",
        "aliases": ["typescript", "ts", "tsc"],
    },
    "go": {
        "category": "programming languages",
        "aliases": ["go", "golang"],
    },
    "rust": {
        "category": "programming languages",
        "aliases": ["rust", "rustc", "cargo"],
    },
    "java": {
        "category": "programming languages",
        "aliases": ["java", "jvm", "spring boot", "maven", "gradle"],
    },
    "csharp": {
        "category": "programming languages",
        "aliases": ["csharp", "c#", ".net", "dotnet"],
    },
    "cpp": {
        "category": "programming languages",
        "aliases": ["c++", "cpp", "cplusplus", "clang"],
    },
    "ruby": {
        "category": "programming languages",
        "aliases": ["ruby", "rails", "rake", "bundle"],
    },
    "swift": {
        "category": "programming languages",
        "aliases": ["swift", "ios dev", "swiftui"],
    },
    "kotlin": {
        "category": "programming languages",
        "aliases": ["kotlin", "kotlinx", "ktor"],
    },
    "dart": {
        "category": "programming languages",
        "aliases": ["dart", "flutter", "pub"],
    },
    "scala": {
        "category": "programming languages",
        "aliases": ["scala", "akka", "sbt"],
    },
    "r": {
        "category": "programming languages",
        "aliases": ["r", "r language", "rstudio"],
    },
    "sql": {
        "category": "programming languages",
        "aliases": ["sql", "postgresql", "mysql", "sqlite", "select", "insert", "update", "delete"],
    },
    "php": {
        "category": "programming languages",
        "aliases": ["php", "laravel", "composer", "composer"],
    },
}


def _normalize_alias(alias: str) -> str:
    """Normalize aliases for consistent lookups."""
    return re.sub(r"\s+", " ", alias.strip().lower())


_ALIASES_TO_CONCEPTS: dict[str, list[tuple[str, str]]] = {}
for _concept_name, payload in CONCEPT_MAP.items():
    aliases = set(payload.get("aliases", [])) | {_concept_name}
    for _alias in aliases:
        norm_alias = _normalize_alias(_alias)
        if not norm_alias:
            continue
        _ALIASES_TO_CONCEPTS.setdefault(norm_alias, []).append(
            (_concept_name, payload.get("category", "general")),
        )

# Fast alias scanning at import time.
# Instead of scanning ~400 aliases individually, we build:
#   1) a token-phrase lookup for purely word-ish aliases (space-separated a-z0-9)
#   2) a small regex for the remaining aliases (punctuation paths, etc.)
_WORD_ALIAS_TO_CONCEPTS: dict[tuple[str, ...], list[tuple[str, str]]] = {}
_NONWORD_ALIASES: list[str] = []
_MAX_ALIAS_WORDS = 1

for _alias_norm, _concepts in _ALIASES_TO_CONCEPTS.items():
    if re.fullmatch(r"[a-z0-9]+(?: [a-z0-9]+)*", _alias_norm):
        parts = tuple(_alias_norm.split(" "))
        _MAX_ALIAS_WORDS = max(_MAX_ALIAS_WORDS, len(parts))
        _WORD_ALIAS_TO_CONCEPTS[parts] = _concepts
    else:
        _NONWORD_ALIASES.append(_alias_norm)

# For non-word aliases, a single regex is still faster than per-alias scans.
_NONWORD_ALIASES.sort(key=len, reverse=True)
if _NONWORD_ALIASES:
    _NONWORD_ALIASES_RE = re.compile("|".join(re.escape(a) for a in _NONWORD_ALIASES))
else:
    _NONWORD_ALIASES_RE = None

_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s\"'`]+|/(?:[^\s\"'`]+/?)*)")
_VERSION_RE = re.compile(r"\bv\d+(?:\.\d+){1,3}\b|\b\d+(?:\.\d+){1,3}\b")
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*){1,4})\b"
)


def extract_concepts(text: str) -> List[Tuple[str, str]]:
    """Extract (concept_name, category) tuples from text using regex and CONCEPT_MAP lookups."""
    if not text:
        return []

    text_lower = text.lower()
    concepts: dict[tuple[str, str], float] = {}

    # Paths and filesystem references
    path_hits = _PATH_RE.findall(text)
    if path_hits:
        concepts[("path", "files/paths")] = concepts.get(("path", "files/paths"), 0.0) + len(path_hits)

    # Version numbers
    versions = _VERSION_RE.findall(text)
    if versions:
        concepts[("versioning", "software")] = concepts.get(("versioning", "software"), 0.0) + len(versions)

    # Capitalized phrases (people names, product names, model names, etc.)
    for match in _CAPITALIZED_PHRASE_RE.finditer(text):
        phrase = _normalize_alias(match.group(0))
        if phrase in _ALIASES_TO_CONCEPTS:
            for concept_name, category in _ALIASES_TO_CONCEPTS[phrase]:
                concepts[(concept_name, category)] = concepts.get((concept_name, category), 0.0) + 1.0

    # Known aliases and concept terms
    # 1) word-ish aliases: token phrase match (O(n * max_alias_words))
    words = re.findall(r"[a-z0-9]+", text_lower)
    n = len(words)
    for i in range(n):
        # Try longer phrases first (max 5-ish), then shorter.
        max_l = min(_MAX_ALIAS_WORDS, n - i)
        for l in range(max_l, 0, -1):
            key = tuple(words[i:i + l])
            hit = _WORD_ALIAS_TO_CONCEPTS.get(key)
            if not hit:
                continue
            for concept_name, category in hit:
                concepts[(concept_name, category)] = concepts.get((concept_name, category), 0.0) + 1.0
            break  # don't allow shorter overlapping aliases starting at i

    # 2) non-word aliases: one regex pass (fallback)
    if _NONWORD_ALIASES_RE is not None:
        for m in _NONWORD_ALIASES_RE.finditer(text_lower):
            alias = _normalize_alias(m.group(0))
            for concept_name, category in _ALIASES_TO_CONCEPTS.get(alias, []):
                concepts[(concept_name, category)] = concepts.get((concept_name, category), 0.0) + 1.0

    return [(name, category) for (name, category), _ in concepts.items()]


def expand_query(tokens: list[str]) -> list[str]:
    """Return concept names matched from query tokens via reverse CONCEPT_MAP lookup."""
    if not tokens:
        return []

    seen: set[str] = set()
    concept_names: list[str] = []

    norm_tokens = [_normalize_alias(str(t)) for t in tokens if t]
    for token in norm_tokens:
        for concept_name, _ in _ALIASES_TO_CONCEPTS.get(token, []):
            if concept_name not in seen:
                seen.add(concept_name)
                concept_names.append(concept_name)

    # Also try longer n-grams; many aliases are 4-5 tokens (e.g., "mac mini m4 pro").
    max_span = min(5, len(norm_tokens))
    for span in range(2, max_span + 1):
        for i in range(0, len(norm_tokens) - span + 1):
            phrase = " ".join(norm_tokens[i:i + span])
            for concept_name, _ in _ALIASES_TO_CONCEPTS.get(phrase, []):
                if concept_name not in seen:
                    seen.add(concept_name)
                    concept_names.append(concept_name)

    return concept_names
