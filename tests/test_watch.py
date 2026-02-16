"""Tests for watch.py stream batching and routing."""

import os
import tempfile
import time
from unittest.mock import Mock, patch

from capture import IngestResult
from watch import MessageBatch, WatchConfig, BaseWatcher


class DummyWatcher(BaseWatcher):
    def start(self):
        self.running = True


def test_message_batch_by_size():
    cfg = WatchConfig(batch_size=2, batch_timeout=100)
    b = MessageBatch(cfg)
    assert b.add_message("a") is False
    assert b.add_message("b") is True
    assert b.get_batch() == ["a", "b"]


def test_message_batch_timeout():
    cfg = WatchConfig(batch_size=10, batch_timeout=0.01)
    b = MessageBatch(cfg)
    b.add_message("a")
    time.sleep(0.02)
    assert b.is_timeout_reached() is True


def test_message_batch_has_messages():
    b = MessageBatch(WatchConfig())
    assert b.has_messages() is False
    b.add_message("x")
    assert b.has_messages() is True


def test_message_batch_reset_after_get():
    b = MessageBatch(WatchConfig())
    b.add_message("x")
    got = b.get_batch()
    assert got == ["x"]
    assert b.has_messages() is False


def test_base_watcher_process_batch_stored():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    with patch("watch.ingest", return_value=IngestResult.STORED):
        w.process_batch(["hello"])
    assert w.stats.batches_processed == 1
    assert w.stats.memories_stored == 1


def test_base_watcher_process_batch_queued():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    with patch("watch.ingest", return_value=IngestResult.QUEUED_FOR_REVIEW):
        w.process_batch(["hello"])
    assert w.stats.memories_queued == 1


def test_base_watcher_process_batch_ignored():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    with patch("watch.ingest", return_value=IngestResult.IGNORED_FLUFF):
        w.process_batch(["ok"])
    assert w.stats.memories_ignored == 1


def test_base_watcher_process_batch_secret_rejected():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    with patch("watch.ingest", return_value=IngestResult.REJECTED_SECRET):
        w.process_batch(["password=123"])
    assert w.stats.secrets_rejected == 1


def test_base_watcher_empty_batch_noop():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    with patch("watch.ingest") as m:
        w.process_batch([])
    m.assert_not_called()


def test_base_watcher_joins_multi_message_batch():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    with patch("watch.ingest", return_value=IngestResult.STORED) as m:
        w.process_batch(["one", "two"])
    text = m.call_args.kwargs["text"]
    assert "[1] one" in text
    assert "[2] two" in text


def test_base_watcher_stop_flushes_remaining():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    w.batch.add_message("leftover")
    with patch("watch.ingest", return_value=IngestResult.STORED):
        w.stop()
    assert w.stats.batches_processed == 1


def test_watch_stats_uptime_positive():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    assert w.stats.uptime >= 0


def test_watch_stats_str_has_fields():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    s = str(w.stats)
    assert "messages" in s
    assert "batches" in s


def test_process_batch_passes_policy_source_meta():
    syn = Mock()
    cfg = WatchConfig(policy="minimal", source="stdin", metadata={"x": 1})
    w = DummyWatcher(synapse=syn, config=cfg)
    with patch("watch.ingest", return_value=IngestResult.STORED) as m:
        w.process_batch(["hello"])
    kwargs = m.call_args.kwargs
    assert kwargs["policy"] == "minimal"
    assert kwargs["source"] == "stdin"
    assert kwargs["meta"]["x"] == 1


def test_process_batch_adds_batch_metadata():
    syn = Mock()
    w = DummyWatcher(synapse=syn)
    with patch("watch.ingest", return_value=IngestResult.STORED) as m:
        w.process_batch(["hello"])
    meta = m.call_args.kwargs["meta"]
    assert meta["batch_size"] == 1
    assert "batch_timestamp" in meta
