"""Scenario data for consumer-facing benchmarks."""

from __future__ import annotations

from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# Scenario 1: Long Thread Recall
# ---------------------------------------------------------------------------
# 50-message conversation thread simulating project planning.
# Messages at indices marked with RELEVANT=True are about the database migration.

RECALL_QUERY = "What did we decide about the database migration?"

# Indices of relevant messages (scattered early, middle, late)
RELEVANT_INDICES = {2, 8, 15, 23, 31, 39, 47}

CONVERSATION_THREAD: List[Dict[str, Any]] = [
    # 0-4
    {"role": "user", "text": "Hey, let's kick off the project planning meeting."},
    {"role": "assistant", "text": "Sure! I've got the agenda ready. We have backend, frontend, and infrastructure topics."},
    {"role": "user", "text": "First item: we need to migrate the database from MySQL to PostgreSQL by end of Q2.", "relevant": True},
    {"role": "assistant", "text": "Got it. Should we also discuss the API versioning strategy?"},
    {"role": "user", "text": "Yes, let's use semantic versioning for the REST API going forward."},
    # 5-9
    {"role": "assistant", "text": "Makes sense. What about the frontend framework — are we sticking with React?"},
    {"role": "user", "text": "Yes, React with TypeScript. No changes there."},
    {"role": "assistant", "text": "Good. And the CI/CD pipeline — Jenkins or GitHub Actions?"},
    {"role": "user", "text": "For the database migration, we decided to use pgloader for the data transfer and run it over a weekend maintenance window.", "relevant": True},
    {"role": "assistant", "text": "Smart. Weekend window minimizes user impact."},
    # 10-14
    {"role": "user", "text": "Let's talk about the mobile app — should we prioritize iOS or Android?"},
    {"role": "assistant", "text": "Our analytics show 60% iOS users, so iOS first makes sense."},
    {"role": "user", "text": "Agreed. We'll use Swift for iOS and Kotlin for Android."},
    {"role": "assistant", "text": "What about the notification system? Push or email?"},
    {"role": "user", "text": "Both. Push for urgent, email for weekly digests."},
    # 15-19
    {"role": "user", "text": "Back to the database migration — the schema changes need a compatibility layer so the old API keeps working during transition.", "relevant": True},
    {"role": "assistant", "text": "I'll set up a dual-write pattern during the migration period."},
    {"role": "user", "text": "Let's discuss the caching strategy. Redis or Memcached?"},
    {"role": "assistant", "text": "Redis gives us more flexibility with data structures."},
    {"role": "user", "text": "Agreed. Redis it is. Also, we need to set up monitoring with Grafana."},
    # 20-24
    {"role": "assistant", "text": "I'll configure Grafana dashboards for the key metrics."},
    {"role": "user", "text": "What about load testing? We need to handle 10k concurrent users."},
    {"role": "assistant", "text": "We can use k6 for load testing. It integrates well with our CI."},
    {"role": "user", "text": "For the database migration rollback plan, we'll keep MySQL running in read-only mode for two weeks after the switch.", "relevant": True},
    {"role": "assistant", "text": "Good safety net. Two weeks should be enough to catch issues."},
    # 25-29
    {"role": "user", "text": "Let's talk about the authentication system. OAuth2 or custom JWT?"},
    {"role": "assistant", "text": "OAuth2 with Auth0 would save us development time."},
    {"role": "user", "text": "Makes sense. Let's go with Auth0 for now."},
    {"role": "assistant", "text": "What about the search functionality? Elasticsearch or Algolia?"},
    {"role": "user", "text": "Elasticsearch for full-text search. We need the customization."},
    # 30-34
    {"role": "assistant", "text": "I'll set up the ES cluster. How many shards?"},
    {"role": "user", "text": "The database migration team confirmed that all stored procedures will be rewritten in PL/pgSQL before the switch.", "relevant": True},
    {"role": "assistant", "text": "That's a big task. How many stored procedures are there?"},
    {"role": "user", "text": "About 45. The team estimates two weeks for the rewrite."},
    {"role": "assistant", "text": "Let's discuss the file storage. S3 or local NFS?"},
    # 35-39
    {"role": "user", "text": "S3 for production, local for development environments."},
    {"role": "assistant", "text": "Good call. What about the logging strategy?"},
    {"role": "user", "text": "ELK stack — Elasticsearch, Logstash, Kibana for centralized logging."},
    {"role": "assistant", "text": "We need to decide on log retention. How long?"},
    {"role": "user", "text": "For the database migration testing, we decided to create a staging environment with a full copy of production data, anonymized.", "relevant": True},
    # 40-44
    {"role": "assistant", "text": "I'll handle the data anonymization pipeline."},
    {"role": "user", "text": "What about the deployment strategy? Blue-green or canary?"},
    {"role": "assistant", "text": "Canary deployments with 5% initial traffic routing."},
    {"role": "user", "text": "Sounds good. Let's also set up feature flags with LaunchDarkly."},
    {"role": "assistant", "text": "Already on it. I'll configure the SDK this week."},
    # 45-49
    {"role": "user", "text": "Quick note — the design team wants to redesign the settings page."},
    {"role": "assistant", "text": "I'll schedule a design review for next Tuesday."},
    {"role": "user", "text": "Final database migration note: the estimated total downtime during the switch is 4 hours, scheduled for Saturday 2am-6am.", "relevant": True},
    {"role": "assistant", "text": "I'll send out the maintenance notification to all users 48 hours in advance."},
    {"role": "user", "text": "Great meeting everyone. Let's reconvene next Wednesday for progress updates."},
]

assert len(CONVERSATION_THREAD) == 50
assert all(i < 50 for i in RELEVANT_INDICES)
assert len(RELEVANT_INDICES) == 7

# ---------------------------------------------------------------------------
# Scenario 2: Time Travel
# ---------------------------------------------------------------------------

TEMPORAL_FACTS: List[Dict[str, Any]] = [
    {
        "content": "User lives in Austin, Texas",
        "valid_from": "2024-01",
        "valid_to": "2024-06",
        "memory_type": "fact",
    },
    {
        "content": "User lives in Denver, Colorado",
        "valid_from": "2024-06",
        "valid_to": None,
        "memory_type": "fact",
    },
    {
        "content": "User works at Acme Corp as a software engineer",
        "valid_from": "2023-01",
        "valid_to": "2024-01",
        "memory_type": "fact",
    },
    {
        "content": "User works at Globex Corporation as a senior engineer",
        "valid_from": "2024-01",
        "valid_to": None,
        "memory_type": "fact",
    },
]

TEMPORAL_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "as_of:2024-03",
        "temporal": "as_of:2024-03",
        "expected_keywords": ["austin", "acme"],
        "unexpected_keywords": ["denver"],
        "description": "March 2024: should see Austin + Acme (before Globex switch in Jan was stored with valid_from=2024-01)",
    },
    {
        "name": "latest",
        "temporal": "latest",
        "expected_keywords": ["denver", "globex"],
        "unexpected_keywords": ["austin"],
        "description": "Latest: should see Denver + Globex (Acme may appear but Austin should not)",
    },
    {
        "name": "during:2024-05 to 2024-07",
        "temporal": "during:2024-05:2024-07",
        "expected_keywords": ["austin", "denver"],
        "unexpected_keywords": [],
        "description": "May-Jul 2024: should see both Austin and Denver (transition period)",
    },
]

# ---------------------------------------------------------------------------
# Scenario 3: Contradiction Resilience
# ---------------------------------------------------------------------------

CONTRADICTION_PAIRS: List[Dict[str, Any]] = [
    {
        "name": "Starbucks flip",
        "first": "User likes Starbucks coffee and goes there every morning",
        "second": "User hates Starbucks coffee and switched to a local roaster",
        "query": "coffee preferences",
        "expected_both": ["starbucks", "roaster"],
    },
    {
        "name": "Diet change",
        "first": "User is a committed vegetarian and avoids all meat",
        "second": "User ordered a steak dinner last night at the restaurant",
        "query": "dietary preferences",
        "expected_both": ["vegetarian", "steak"],
    },
]
