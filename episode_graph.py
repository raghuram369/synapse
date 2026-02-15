"""Episode graph helpers for temporal clustering of memories."""

from typing import List, Optional, Dict, Any


def find_or_create_episode(
    store,  # MemoryStore instance
    timestamp: float,
    window_secs: float = 1800.0,
) -> int:
    """Return an open episode within ``window_secs`` or create a new one."""
    # Find open episode within window
    for episode_id, episode_data in store.episodes.items():
        if (episode_data.get('ended_at') is None and 
            episode_data['started_at'] <= timestamp and
            (timestamp - episode_data['started_at']) < window_secs):
            return episode_id
    
    # Create new episode
    episode_data = {
        'name': '',
        'started_at': timestamp,
        'ended_at': None,
        'metadata': {}
    }
    return store.insert_episode(episode_data)


def get_episode_siblings(store, memory_id: int) -> List[int]:
    """Return all memory IDs that are linked to the same episode(s) as memory_id."""
    # This function is kept for compatibility but the main logic is in EpisodeIndex
    # Find episodes that contain this memory
    memory_episodes = []
    for episode_id, episode_data in store.episodes.items():
        if memory_id in episode_data.get('memory_ids', []):
            memory_episodes.append(episode_id)
    
    if not memory_episodes:
        return []
    
    # Find all other memories in those episodes
    siblings = set()
    for episode_id in memory_episodes:
        episode_data = store.episodes.get(episode_id, {})
        for sibling_id in episode_data.get('memory_ids', []):
            if sibling_id != memory_id:
                siblings.add(sibling_id)
    
    return list(siblings)


def close_stale_episodes(
    store,  # MemoryStore instance
    now: float,
    window: float = 1800.0,
):
    """Mark open episodes as ended when no memory arrived in ``window`` seconds."""
    cutoff = now - window
    
    for episode_id, episode_data in store.episodes.items():
        if episode_data.get('ended_at') is not None:
            continue  # Already closed
        
        # Find the last memory time in this episode
        last_memory_at = episode_data['started_at']  # Default to episode start
        
        for memory_id in episode_data.get('memory_ids', []):
            if memory_id in store.memories:
                memory_created = store.memories[memory_id]['created_at']
                last_memory_at = max(last_memory_at, memory_created)
        
        # Close episode if it's stale
        if last_memory_at < cutoff:
            store.episodes[episode_id]['ended_at'] = now
