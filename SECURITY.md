# SECURITY.md

# Synapse AI Memory Security Policy

Synapse AI Memory is designed for private, local-first operation. This policy summarizes current
security assumptions and recommended deployment practices.

## 1) Threat Model

- **Local-first default posture**: Memory operations are intended to run locally by default.
  Sensitive `.synapse` data is created, read, and used on the machine where Synapse AI Memory is running.
- **Federation trust boundary**:
  - Federation peers are treated as untrusted unless explicitly authenticated.
  - Remote peers may be compromised or malicious; never assume shared intent.
- **No cloud dependency**:
  - Synapse AI Memory does not require or use cloud APIs to store or synchronize memory content.
  - Federation sync uses direct peer-to-peer HTTP interactions between nodes.

## 2) Federation Security

- **Listen binding**:
  - Default behavior: federation binds only to localhost (`127.0.0.1`) unless explicitly configured otherwise.
  - Any public/network exposure must be an explicit configuration choice.
  - Treat non-loopback binding (`0.0.0.0` / external interface) as a separate trust and deployment boundary.
- **Authentication**:
  - Bearer token authentication is implemented and should be enabled for any deployment that is not purely local.
  - Configure tokens consistently for every exposed endpoint and peer interaction.
  - Keep tokens long, random, and rotate them periodically.
- **Payload limit protection**:
  - Federation HTTP handlers enforce a request payload cap of **50 MB**.
  - This cap applies to JSON and binary sync requests and helps protect against oversized payload abuse.
- **Rate limiting** (recommended):
  - Synapse AI Memory does not enforce per-peer rate quotas in the core library.
  - Add rate limiting at the edge (reverse proxy / API gateway / firewall) for exposed nodes.
  - Enforce conservative request-rate and connection limits, especially on public hosts.
- **Merkle-based sync minimization**:
  - Sync begins by exchanging Merkle metadata (`root` and `bucket_hashes`).
  - A peer learns only bucket hashes first, not memory bodies.
  - Full memory content is exchanged only in later phases after the hash negotiation identifies missing buckets.
  - Pair this with auth so metadata exposure is not meaningful for unauthorized peers.

## 3) Data at Rest

- `.synapse` exports/imports include **CRC32** checks.
  - CRC32 helps detect corruption and accidental tampering.
  - CRC32 is not encryption and is not a cryptographic integrity guarantee.
- **No encryption by default**:
  - Default behavior relies on operating system file permissions and process isolation.
  - Ensure storage directories are restricted to the service identity and trusted operators only.
- **Recommended enhancement for sensitive data**:
  - Enable full-disk encryption on the host.
  - Keep secrets and token storage in OS credential stores where available.

## 4) Reporting Vulnerabilities

- **Preferred security contact**: use GitHub's private vulnerability reporting / Security Advisory flow for this repository.
- **Fallback contact**: the maintainer on GitHub: **@raghuram369**.
- Responsible disclosure:
  1. Report privately first (direct message or GitHub private communication).
  2. Include version, affected component, reproducible steps, and impact.
  3. Do not access or exfiltrate additional data beyond what is needed to validate the issue.
  4. Allow time for a fix and coordinated disclosure before public posting.

## 5) Best Practices

- Use Bearer token auth for federation unless all peers are strictly localhost and ephemeral.
- Keep federation servers localhost-only during development.
- Regularly export and back up `.synapse` files and validate backups with `inspect` tooling.
- Apply strict OS-level permissions to memory/data directories and daemon users.
- Use short-lived tokens and rotate them after team changes, network movement, or suspected exposure.

## Policy Scope

- This document covers Synapse AI Memory core behavior and local/federated deployments.
- Third-party integrations and external infrastructure are outside repository security boundaries and must follow their own controls.
