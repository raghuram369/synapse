"""Stream watcher for Synapse AI Memory.

Watches text streams and feeds batched content to the memory router:
- stdin pipe watching
- file tail watching  
- callback function watching
- Configurable batching (message count or time window)
"""

from __future__ import annotations

import os
import select
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TextIO, TYPE_CHECKING

from capture import ingest, IngestResult

if TYPE_CHECKING:
    from synapse import Synapse
    from review_queue import ReviewQueue

# ---------------------------------------------------------------------------
# Configuration and Types
# ---------------------------------------------------------------------------

@dataclass
class WatchConfig:
    """Configuration for stream watching."""
    batch_size: int = 5  # Messages per batch
    batch_timeout: float = 30.0  # Seconds to wait before processing partial batch
    policy: str = "auto"  # Memory router policy
    source: str = "stream"  # Source identifier
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class WatchStats:
    """Statistics from stream watching."""
    messages_seen: int = 0
    batches_processed: int = 0
    memories_stored: int = 0
    memories_queued: int = 0
    memories_ignored: int = 0
    secrets_rejected: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time
    
    def __str__(self) -> str:
        return (f"📊 Stats: {self.messages_seen} messages, {self.batches_processed} batches, "
                f"{self.memories_stored} stored, {self.memories_queued} queued, "
                f"{self.memories_ignored} ignored, {self.secrets_rejected} secrets rejected, "
                f"uptime {self.uptime:.1f}s")


# ---------------------------------------------------------------------------
# Message Batching
# ---------------------------------------------------------------------------

class MessageBatch:
    """Collects messages into batches based on count or time."""
    
    def __init__(self, config: WatchConfig):
        self.config = config
        self.messages: List[str] = []
        self.first_message_time = None
        self.lock = threading.Lock()
    
    def add_message(self, message: str) -> bool:
        """Add message to batch. Returns True if batch is ready to process."""
        with self.lock:
            self.messages.append(message.strip())
            if self.first_message_time is None:
                self.first_message_time = time.time()
            
            # Check if batch is ready
            return (len(self.messages) >= self.config.batch_size or
                    time.time() - self.first_message_time >= self.config.batch_timeout)
    
    def get_batch(self) -> List[str]:
        """Get and clear current batch."""
        with self.lock:
            batch = self.messages.copy()
            self.messages.clear()
            self.first_message_time = None
            return batch
    
    def has_messages(self) -> bool:
        """Check if batch has any messages."""
        with self.lock:
            return len(self.messages) > 0
    
    def is_timeout_reached(self) -> bool:
        """Check if batch timeout has been reached."""
        with self.lock:
            if not self.messages or self.first_message_time is None:
                return False
            return time.time() - self.first_message_time >= self.config.batch_timeout


# ---------------------------------------------------------------------------
# Stream Watchers
# ---------------------------------------------------------------------------

class BaseWatcher:
    """Base class for stream watchers."""
    
    def __init__(self, synapse: "Synapse", review_queue: Optional["ReviewQueue"] = None, config: Optional[WatchConfig] = None):
        self.synapse = synapse
        self.review_queue = review_queue
        self.config = config or WatchConfig()
        self.stats = WatchStats()
        self.running = False
        self.batch = MessageBatch(self.config)
        
    def process_batch(self, messages: List[str]) -> None:
        """Process a batch of messages through the memory router."""
        if not messages:
            return
            
        # Combine messages into a single text block with context
        if len(messages) == 1:
            text = messages[0]
        else:
            text = "\n".join(f"[{i+1}] {msg}" for i, msg in enumerate(messages))
            
        # Add batch metadata
        metadata = self.config.metadata.copy()
        metadata.update({
            "batch_size": len(messages),
            "batch_timestamp": time.time(),
        })
        
        # Route through memory router
        result = ingest(
            text=text,
            synapse=self.synapse,
            review_queue=self.review_queue,
            source=self.config.source,
            meta=metadata,
            policy=self.config.policy
        )
        
        # Update stats
        self.stats.batches_processed += 1
        if result == IngestResult.STORED:
            self.stats.memories_stored += 1
        elif result == IngestResult.QUEUED_FOR_REVIEW:
            self.stats.memories_queued += 1
        elif result in (IngestResult.IGNORED_FLUFF, IngestResult.IGNORED_POLICY):
            self.stats.memories_ignored += 1
        elif result == IngestResult.REJECTED_SECRET:
            self.stats.secrets_rejected += 1
            
        # Print result
        self._print_result(messages, result)
    
    def _print_result(self, messages: List[str], result: IngestResult) -> None:
        """Print batch processing result."""
        preview = (messages[0] if len(messages) == 1 
                  else f"{len(messages)} messages: {messages[0][:50]}...")
        preview = preview.replace('\n', ' ')[:100]
        
        if result == IngestResult.STORED:
            print(f"✅ Stored batch: {preview}")
        elif result == IngestResult.QUEUED_FOR_REVIEW:
            print(f"📋 Queued batch: {preview}")
        elif result == IngestResult.IGNORED_FLUFF:
            print(f"🙄 Ignored batch (fluff): {preview}")
        elif result == IngestResult.REJECTED_SECRET:
            print(f"🔒 Rejected batch (secrets): {preview[:30]}...")
        elif result == IngestResult.IGNORED_POLICY:
            print(f"⏸️  Ignored batch (policy): {preview}")
    
    def start(self) -> None:
        """Start watching (implemented by subclasses)."""
        raise NotImplementedError
        
    def stop(self) -> None:
        """Stop watching."""
        self.running = False
        
        # Process any remaining messages
        if self.batch.has_messages():
            final_batch = self.batch.get_batch()
            self.process_batch(final_batch)


class StdinWatcher(BaseWatcher):
    """Watches stdin for incoming text."""
    
    def start(self) -> None:
        """Start watching stdin."""
        self.running = True
        print(f"👀 Watching stdin (batch_size={self.config.batch_size}, timeout={self.config.batch_timeout}s)")
        print("📝 Type messages (press Ctrl+C to stop):")
        
        try:
            while self.running:
                # Check for input with timeout
                if self._has_input(timeout=1.0):
                    try:
                        line = sys.stdin.readline()
                        if not line:  # EOF
                            break
                            
                        line = line.strip()
                        if not line:
                            continue
                            
                        self.stats.messages_seen += 1
                        
                        # Add to batch and process if ready
                        if self.batch.add_message(line):
                            batch = self.batch.get_batch()
                            self.process_batch(batch)
                            
                    except EOFError:
                        break
                        
                # Check for timeout
                elif self.batch.is_timeout_reached():
                    batch = self.batch.get_batch()
                    if batch:
                        self.process_batch(batch)
                        
        except KeyboardInterrupt:
            print("\n🛑 Stdin watch stopped.")
        finally:
            # Process any remaining messages
            if self.batch.has_messages():
                final_batch = self.batch.get_batch()
                self.process_batch(final_batch)
                
            print(self.stats)
    
    def _has_input(self, timeout: float = 0.1) -> bool:
        """Check if stdin has input available."""
        if sys.stdin.isatty():
            # Interactive mode - always check
            return True
        
        # Pipe mode - use select on Unix-like systems
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            return bool(ready)
        except:
            # Fallback for Windows or other systems
            return True


class FileWatcher(BaseWatcher):
    """Watches a file for new lines (tail -f style)."""
    
    def __init__(self, file_path: str, synapse: "Synapse", review_queue: Optional["ReviewQueue"] = None, config: Optional[WatchConfig] = None):
        super().__init__(synapse, review_queue, config)
        self.file_path = file_path
        self.last_size = 0
        
    def start(self) -> None:
        """Start watching the file."""
        if not os.path.exists(self.file_path):
            print(f"❌ File not found: {self.file_path}")
            return
            
        # Initialize with current file size
        self.last_size = os.path.getsize(self.file_path)
        
        self.running = True
        print(f"👀 Watching file: {self.file_path}")
        print(f"⚙️  Config: batch_size={self.config.batch_size}, timeout={self.config.batch_timeout}s")
        
        try:
            while self.running:
                self._check_file_changes()
                
                # Check for batch timeout
                if self.batch.is_timeout_reached():
                    batch = self.batch.get_batch()
                    if batch:
                        self.process_batch(batch)
                        
                time.sleep(1.0)  # Poll interval
                
        except KeyboardInterrupt:
            print(f"\n🛑 File watch stopped.")
        finally:
            # Process any remaining messages
            if self.batch.has_messages():
                final_batch = self.batch.get_batch()
                self.process_batch(final_batch)
                
            print(self.stats)
    
    def _check_file_changes(self) -> None:
        """Check for new content in the file."""
        try:
            current_size = os.path.getsize(self.file_path)
            if current_size > self.last_size:
                # Read new content
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self.last_size)
                    new_content = f.read()
                    
                # Process new lines
                for line in new_content.split('\n'):
                    line = line.strip()
                    if line:
                        self.stats.messages_seen += 1
                        
                        # Add to batch and process if ready
                        if self.batch.add_message(line):
                            batch = self.batch.get_batch()
                            self.process_batch(batch)
                            
                self.last_size = current_size
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")


class CallbackWatcher(BaseWatcher):
    """Watches for messages via callback function."""
    
    def __init__(self, synapse: "Synapse", review_queue: Optional["ReviewQueue"] = None, config: Optional[WatchConfig] = None):
        super().__init__(synapse, review_queue, config)
        self.message_queue = []
        self.queue_lock = threading.Lock()
        
    def on_message(self, message: str) -> None:
        """Callback function for external systems to send messages."""
        with self.queue_lock:
            self.message_queue.append(message.strip())
    
    def start(self) -> None:
        """Start the callback watcher."""
        self.running = True
        print("👀 Callback watcher started. Call on_message() to send messages.")
        
        try:
            while self.running:
                # Process queued messages
                messages_to_process = []
                with self.queue_lock:
                    messages_to_process = self.message_queue.copy()
                    self.message_queue.clear()
                
                for message in messages_to_process:
                    if message:
                        self.stats.messages_seen += 1
                        
                        # Add to batch and process if ready
                        if self.batch.add_message(message):
                            batch = self.batch.get_batch()
                            self.process_batch(batch)
                
                # Check for batch timeout
                if self.batch.is_timeout_reached():
                    batch = self.batch.get_batch()
                    if batch:
                        self.process_batch(batch)
                        
                time.sleep(0.5)  # Processing interval
                
        except KeyboardInterrupt:
            print(f"\n🛑 Callback watch stopped.")
        finally:
            # Process any remaining messages
            if self.batch.has_messages():
                final_batch = self.batch.get_batch()
                self.process_batch(final_batch)
                
            print(self.stats)


# ---------------------------------------------------------------------------
# High-level Interface for Synapse Integration
# ---------------------------------------------------------------------------

def watch_stdin(synapse: "Synapse", review_queue: Optional["ReviewQueue"] = None, policy: str = "auto", batch_size: int = 5, batch_timeout: float = 30.0) -> None:
    """Watch stdin and feed to memory router."""
    config = WatchConfig(
        batch_size=batch_size,
        batch_timeout=batch_timeout,
        policy=policy,
        source="stdin"
    )
    watcher = StdinWatcher(synapse, review_queue, config)
    watcher.start()


def watch_file(file_path: str, synapse: "Synapse", review_queue: Optional["ReviewQueue"] = None, policy: str = "auto", batch_size: int = 5, batch_timeout: float = 30.0) -> None:
    """Watch a file and feed new lines to memory router."""
    config = WatchConfig(
        batch_size=batch_size,
        batch_timeout=batch_timeout,
        policy=policy,
        source=f"file:{os.path.basename(file_path)}"
    )
    watcher = FileWatcher(file_path, synapse, review_queue, config)
    watcher.start()


def create_callback_watcher(synapse: "Synapse", review_queue: Optional["ReviewQueue"] = None, policy: str = "auto", batch_size: int = 5, batch_timeout: float = 30.0) -> CallbackWatcher:
    """Create a callback watcher for programmatic use."""
    config = WatchConfig(
        batch_size=batch_size,
        batch_timeout=batch_timeout,
        policy=policy,
        source="callback"
    )
    return CallbackWatcher(synapse, review_queue, config)


# ---------------------------------------------------------------------------
# Integration with Synapse class
# ---------------------------------------------------------------------------

def add_watch_methods(synapse_class):
    """Add watch methods to the Synapse class."""
    
    def watch_stream(self, stream_type: str = "stdin", **kwargs):
        """Watch a stream and feed to memory router.
        
        Args:
            stream_type: "stdin", "file", or "callback"
            **kwargs: Additional configuration (file_path, policy, batch_size, etc.)
        """
        review_queue = kwargs.get('review_queue')
        policy = kwargs.get('policy', 'auto')
        batch_size = kwargs.get('batch_size', 5)
        batch_timeout = kwargs.get('batch_timeout', 30.0)
        
        if stream_type == "stdin":
            watch_stdin(self, review_queue, policy, batch_size, batch_timeout)
        elif stream_type == "file":
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path required for file stream type")
            watch_file(file_path, self, review_queue, policy, batch_size, batch_timeout)
        elif stream_type == "callback":
            return create_callback_watcher(self, review_queue, policy, batch_size, batch_timeout)
        else:
            raise ValueError(f"Unknown stream type: {stream_type}")
    
    def ingest_text(self, text: str, **kwargs):
        """Ingest a single piece of text through the memory router."""
        review_queue = kwargs.get('review_queue')
        source = kwargs.get('source', 'direct')
        metadata = kwargs.get('metadata')
        policy = kwargs.get('policy', 'auto')
        
        return ingest(
            text=text,
            synapse=self,
            review_queue=review_queue,
            source=source,
            meta=metadata,
            policy=policy
        )
    
    # Add methods to class
    synapse_class.watch = watch_stream
    synapse_class.ingest = ingest_text
    
    return synapse_class