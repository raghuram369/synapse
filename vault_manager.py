"""Vault Manager - Per-user isolated storage for Synapse memories."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from synapse import Synapse

VAULT_REGISTRY_FILE = "vault_registry.json"


class VaultRegistry:
    """Registry to track all available vaults."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.registry_path = os.path.join(base_path, VAULT_REGISTRY_FILE)
        self._vaults: Dict[str, Dict[str, Any]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load vault registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    self._vaults = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._vaults = {}
        else:
            self._vaults = {}
    
    def _save_registry(self):
        """Save vault registry to disk."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._vaults, f, indent=2, ensure_ascii=False)
    
    def register_vault(self, vault_id: str, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new vault."""
        if vault_id in self._vaults:
            return False
        
        vault_info = {
            'vault_id': vault_id,
            'user_id': user_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'metadata': metadata or {}
        }
        
        self._vaults[vault_id] = vault_info
        self._save_registry()
        return True
    
    def get_vault_info(self, vault_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a vault."""
        return self._vaults.get(vault_id)
    
    def list_vaults(self) -> List[Dict[str, Any]]:
        """List all registered vaults."""
        return list(self._vaults.values())
    
    def find_user_vault(self, user_id: str) -> Optional[str]:
        """Find the vault ID for a specific user."""
        for vault_id, info in self._vaults.items():
            if info.get('user_id') == user_id:
                return vault_id
        return None
    
    def update_access_time(self, vault_id: str):
        """Update the last accessed time for a vault."""
        if vault_id in self._vaults:
            self._vaults[vault_id]['last_accessed'] = time.time()
            self._save_registry()


def _validate_vault_id(vault_id: str) -> None:
    """Validate vault_id contains only safe characters."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', vault_id):
        raise ValueError(
            f"Invalid vault_id '{vault_id}': must contain only "
            "alphanumeric characters, underscores, and hyphens"
        )


class VaultManager:
    """Manages per-user vaults with automatic isolation."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        if base_path != ":memory:":
            os.makedirs(base_path, exist_ok=True)
            self.registry = VaultRegistry(base_path)
        else:
            self.registry = None
            self._memory_registry: Dict[str, Dict[str, Any]] = {}
        self._active_vaults: Dict[str, "Synapse"] = {}
    
    def get_vault_path(self, vault_id: str) -> str:
        """Get the storage path for a vault."""
        _validate_vault_id(vault_id)
        if self.base_path == ":memory:":
            return ":memory:"
        
        vault_dir = os.path.join(self.base_path, f"vault_{vault_id}")
        os.makedirs(vault_dir, exist_ok=True)
        return os.path.join(vault_dir, "synapse_store")
    
    def get_or_create_vault(self, vault_id: str, user_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> "Synapse":
        """Get existing vault or create new one."""
        _validate_vault_id(vault_id)
        
        # Check if vault is already loaded
        if vault_id in self._active_vaults:
            if self.registry:
                self.registry.update_access_time(vault_id)
            return self._active_vaults[vault_id]
        
        # Register vault if not exists
        if self.registry and not self.registry.get_vault_info(vault_id):
            self.registry.register_vault(vault_id, user_id, metadata)
        elif self.base_path == ":memory:" and vault_id not in self._memory_registry:
            self._memory_registry[vault_id] = {
                'vault_id': vault_id,
                'user_id': user_id,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'metadata': metadata or {}
            }
        
        # Create Synapse instance for vault
        from synapse import Synapse
        vault_path = self.get_vault_path(vault_id)
        vault_synapse = Synapse(path=vault_path)
        vault_synapse._vault_id = vault_id
        vault_synapse._user_id = user_id
        
        self._active_vaults[vault_id] = vault_synapse
        
        if self.registry:
            self.registry.update_access_time(vault_id)
        
        return vault_synapse
    
    def get_user_vault(self, user_id: str, auto_create: bool = True) -> Optional["Synapse"]:
        """Get vault for a specific user, optionally creating it."""
        if not self.registry and self.base_path != ":memory:":
            return None
        
        # Find existing vault for user
        if self.registry:
            existing_vault_id = self.registry.find_user_vault(user_id)
            if existing_vault_id:
                return self.get_or_create_vault(existing_vault_id, user_id)
        elif self.base_path == ":memory:":
            for vid, info in self._memory_registry.items():
                if info.get('user_id') == user_id:
                    return self.get_or_create_vault(vid, user_id)
        
        # Create new vault for user
        if auto_create:
            vault_id = f"user_{user_id}"
            return self.get_or_create_vault(vault_id, user_id)
        
        return None
    
    def list_vaults(self) -> List[Dict[str, Any]]:
        """List all registered vaults."""
        if self.registry:
            return self.registry.list_vaults()
        if self.base_path == ":memory:":
            return list(self._memory_registry.values())
        return []
    
    def close_vault(self, vault_id: str):
        """Close and remove vault from active cache."""
        if vault_id in self._active_vaults:
            vault = self._active_vaults.pop(vault_id)
            try:
                vault.close()
            except Exception:
                pass
    
    def close_all_vaults(self):
        """Close all active vaults."""
        for vault_id in list(self._active_vaults.keys()):
            self.close_vault(vault_id)

    def close(self):
        """Close all vaults and clean up."""
        self.close_all_vaults()