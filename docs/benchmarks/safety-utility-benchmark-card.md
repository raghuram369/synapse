# Synapse Safety & Utility Benchmark Card

**Status:** Draft (public template)

This card defines the standard public-facing safety/utility metrics for Synapse benchmark reporting.
All values in this card are intentionally marked as placeholders unless explicitly measured in a signed run artifact.

## 1) Recall Utility

### Metric definition
- **Goal:** measure useful recall quality for consumer-level querying.
- **Primary metric:** Recall@10 for benchmark query sets.
- **Formula:**
  
  
  `Recall@10 = (# queries with at least one correct answer in top-10) / (total queries)`

- **Supplemental metric:** `MRR@10`.

### Reporting format
Report the following per scenario/dataset in a markdown/JSON table:

- `scenario` (e.g., `locomo`,`practical`)
- `num_queries`
- `recall@1`, `recall@5`, `recall@10` (percent)
- `mrr@10` (or `mrr`)
- `seed`
- `sample_version` / dataset hash
- `date_utc`

Example report row:

| scenario | num_queries | recall@1 | recall@5 | recall@10 | mrr | seed | sample_version | date_utc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| locomo | 200 | 0.304 | 0.535 | 0.629 | 0.406 | 2026 | locomo/v0.1 | 2026-02-16T00:00:00Z |

> Replace this row with measured values only.

## 2) Permission-Block Effectiveness

### Metric definition
- **Goal:** verify policy blocks are actually enforced and auditable.
- **Primary metric:** **Permission-Block Effectiveness (PBE)**

`PBE = (# blocked denied attempts with expected block reason) / (# total forbidden-attempt probes)`

- **Expected block reason(s):** policy engine must include one of
  `sensitivity_policy_block`, `permission.denied`, `scope_denied`, or equivalent reason code in receipt.

### Reporting format
For every run, report:

- `total_probe_attempts`
- `blocked_count`
- `blocked_with_expected_reason_count`
- `blocked_without_expected_reason_count`
- `pbe`
- `false_allow_count` (forbidden attempts that passed)
- `false_deny_count` (legitimate requests denied, for context)

## 3) Sensitive-Memory Non-Leak Rate

### Metric definition
- **Goal:** confirm sensitive memories do not cross into disallowed visibility.
- **Primary metric:** **Sensitive Non-Leak Rate (SNLR)**

`SNLR = 1 - (sensitive_leaks / sensitive_leak_probe_attempts)`

- `sensitive_leaks` = cases where a sensitive memory appears in non-private scope/query.

### Reporting format
For each probe batch, report:

- `sensitive_probe_attempts`
- `sensitive_leaks`
- `sensitive_nonleak_rate`
- `coverage_by_scope` (private/shared/public)
- `coverage_by_channel` (API, MCP tool, CLI if applicable)

## 4) Methodology

### Evaluation harness
- Use deterministic fixtures and fixed random seed.
- Separate runs for:
  1. `utility` (recall-focused)
  2. `permission_block` (policy deny probes)
  3. `sensitive_nonleak` (visibility probing)
- Run each suite from clean/in-memory store unless a scenario explicitly tests persistence.
- Record exact software/version, dataset revision, and run timestamp.

### Data splits and probes
- Keep utility and safety probes mutually independent.
- Publish only probe counts and computed rates; avoid cherry-picking.
- Prefer blind execution + checksum-locked artifacts for reproducibility.

### Thresholds (initial)
- PBE and SNLR: target `>= 0.98`
- Recall@10 baseline target: dataset-specific (defined per benchmark family)

## 5) Caveats Template

- [ ] Environment variance (hardware, Python/runtime, hash seed, `PYTHONHASHSEED`)
- [ ] Dataset provenance and licensing constraints
- [ ] Any manual redaction/filtering in datasets
- [ ] Whether embeddings/simulated policy stubs were disabled
- [ ] Whether external systems (LLMs, network calls) were involved
- [ ] Any known bias in sensitive-detector rules for domain-specific entities
- [ ] Whether false-deny/false-allow cases were logged for error analysis

## 6) Placeholder vs measured values

- `status`: placeholder
- all numeric rates above: placeholder until CI run output is attached
- environment/version line: placeholder unless verified from the run metadata

## 7) Artifact linkage

- **Machine-readable template:** `docs/benchmarks/safety-utility-benchmark-metadata.json`
- **Markdown card:** this file

---

This is the shared public artifact contract intended for CI dashboards, release notes, and PR checks.
