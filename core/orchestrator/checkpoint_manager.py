"""
Checkpoint Manager - Handles saving and restoring system state for continuous operation
"""

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
import hashlib
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class Checkpoint:
    """Represents a system checkpoint"""
    id: str
    iteration: int
    timestamp: datetime
    state: Dict[str, Any]
    metrics: Dict[str, float]
    agent_states: Dict[str, Dict]
    milestone_progress: Dict[str, float]
    error_count: int
    success_count: int
    hash: str
    description: str = ""
    is_recovery_point: bool = False


class CheckpointManager:
    """Manages system checkpoints for recovery and continuity"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.checkpoint_dir = Path(f"persistence/checkpoints/{project_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_checkpoint: Optional[Checkpoint] = None
        self.checkpoint_history: List[Checkpoint] = []
        self.max_checkpoints = 100
        self.auto_checkpoint_interval = 10  # iterations
        
        self._load_checkpoint_history()

    def _load_checkpoint_history(self):
        """Load existing checkpoint history"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                history_data = json.load(f)
                for cp_data in history_data:
                    checkpoint = self._dict_to_checkpoint(cp_data)
                    self.checkpoint_history.append(checkpoint)
                    
    def _dict_to_checkpoint(self, data: Dict) -> Checkpoint:
        """Convert dictionary to Checkpoint object"""
        return Checkpoint(
            id=data["id"],
            iteration=data["iteration"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state=data["state"],
            metrics=data["metrics"],
            agent_states=data["agent_states"],
            milestone_progress=data["milestone_progress"],
            error_count=data["error_count"],
            success_count=data["success_count"],
            hash=data["hash"],
            description=data.get("description", ""),
            is_recovery_point=data.get("is_recovery_point", False)
        )

    def create_checkpoint(
        self,
        iteration: int,
        state: Dict[str, Any],
        metrics: Dict[str, float],
        agent_states: Dict[str, Dict],
        milestone_progress: Dict[str, float],
        error_count: int,
        success_count: int,
        description: str = "",
        is_recovery_point: bool = False
    ) -> Checkpoint:
        """Create a new checkpoint"""
        
        # Generate checkpoint ID
        checkpoint_id = f"cp_{iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate state hash for verification
        state_str = json.dumps(state, sort_keys=True)
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        
        checkpoint = Checkpoint(
            id=checkpoint_id,
            iteration=iteration,
            timestamp=datetime.now(),
            state=state,
            metrics=metrics,
            agent_states=agent_states,
            milestone_progress=milestone_progress,
            error_count=error_count,
            success_count=success_count,
            hash=state_hash,
            description=description,
            is_recovery_point=is_recovery_point
        )
        
        # Save checkpoint to disk
        self._save_checkpoint(checkpoint)
        
        # Update history
        self.checkpoint_history.append(checkpoint)
        self.current_checkpoint = checkpoint
        
        # Cleanup old checkpoints
        if len(self.checkpoint_history) > self.max_checkpoints:
            self._cleanup_old_checkpoints()
            
        logger.info(f"Created checkpoint {checkpoint_id} at iteration {iteration}")
        return checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to disk"""
        # Save as JSON for readability
        json_file = self.checkpoint_dir / f"{checkpoint.id}.json"
        checkpoint_data = {
            "id": checkpoint.id,
            "iteration": checkpoint.iteration,
            "timestamp": checkpoint.timestamp.isoformat(),
            "state": checkpoint.state,
            "metrics": checkpoint.metrics,
            "agent_states": checkpoint.agent_states,
            "milestone_progress": checkpoint.milestone_progress,
            "error_count": checkpoint.error_count,
            "success_count": checkpoint.success_count,
            "hash": checkpoint.hash,
            "description": checkpoint.description,
            "is_recovery_point": checkpoint.is_recovery_point
        }
        
        with open(json_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
            
        # Also save as pickle for complex objects
        pickle_file = self.checkpoint_dir / f"{checkpoint.id}.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(checkpoint, f)
            
        # Update history file
        self._save_checkpoint_history()

    def _save_checkpoint_history(self):
        """Save checkpoint history to disk"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        history_data = []
        
        for checkpoint in self.checkpoint_history[-self.max_checkpoints:]:
            history_data.append({
                "id": checkpoint.id,
                "iteration": checkpoint.iteration,
                "timestamp": checkpoint.timestamp.isoformat(),
                "state": checkpoint.state,
                "metrics": checkpoint.metrics,
                "agent_states": checkpoint.agent_states,
                "milestone_progress": checkpoint.milestone_progress,
                "error_count": checkpoint.error_count,
                "success_count": checkpoint.success_count,
                "hash": checkpoint.hash,
                "description": checkpoint.description,
                "is_recovery_point": checkpoint.is_recovery_point
            })
            
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a specific checkpoint"""
        # Try loading from pickle first (preserves complex objects)
        pickle_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        if pickle_file.exists():
            try:
                with open(pickle_file, "rb") as f:
                    checkpoint = pickle.load(f)
                    logger.info(f"Loaded checkpoint {checkpoint_id} from pickle")
                    return checkpoint
            except Exception as e:
                logger.warning(f"Failed to load pickle checkpoint: {e}")
                
        # Fall back to JSON
        json_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        if json_file.exists():
            with open(json_file, "r") as f:
                data = json.load(f)
                checkpoint = self._dict_to_checkpoint(data)
                logger.info(f"Loaded checkpoint {checkpoint_id} from JSON")
                return checkpoint
                
        logger.error(f"Checkpoint {checkpoint_id} not found")
        return None

    def load_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Load the most recent checkpoint"""
        if self.checkpoint_history:
            latest = self.checkpoint_history[-1]
            return self.load_checkpoint(latest.id)
        return None

    def load_recovery_point(self) -> Optional[Checkpoint]:
        """Load the most recent recovery point"""
        for checkpoint in reversed(self.checkpoint_history):
            if checkpoint.is_recovery_point:
                return self.load_checkpoint(checkpoint.id)
        return self.load_latest_checkpoint()

    def should_checkpoint(self, iteration: int) -> bool:
        """Determine if a checkpoint should be created"""
        if not self.checkpoint_history:
            return True
            
        last_checkpoint = self.checkpoint_history[-1]
        iterations_since = iteration - last_checkpoint.iteration
        
        return iterations_since >= self.auto_checkpoint_interval

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        # Keep recovery points and recent checkpoints
        to_keep = set()
        
        # Keep all recovery points
        for cp in self.checkpoint_history:
            if cp.is_recovery_point:
                to_keep.add(cp.id)
                
        # Keep recent checkpoints
        for cp in self.checkpoint_history[-50:]:
            to_keep.add(cp.id)
            
        # Delete old files
        for file in self.checkpoint_dir.glob("cp_*"):
            checkpoint_id = file.stem
            if checkpoint_id not in to_keep:
                file.unlink()
                logger.info(f"Cleaned up old checkpoint: {checkpoint_id}")
                
        # Update history
        self.checkpoint_history = [
            cp for cp in self.checkpoint_history
            if cp.id in to_keep
        ]

    def get_checkpoint_summary(self) -> Dict:
        """Get summary of checkpoints"""
        recovery_points = sum(1 for cp in self.checkpoint_history if cp.is_recovery_point)
        
        return {
            "total_checkpoints": len(self.checkpoint_history),
            "recovery_points": recovery_points,
            "latest_checkpoint": self.current_checkpoint.id if self.current_checkpoint else None,
            "latest_iteration": self.current_checkpoint.iteration if self.current_checkpoint else 0,
            "checkpoint_dir": str(self.checkpoint_dir),
            "auto_interval": self.auto_checkpoint_interval
        }

    def validate_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Validate checkpoint integrity"""
        # Recalculate hash and compare
        state_str = json.dumps(checkpoint.state, sort_keys=True)
        calculated_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        
        is_valid = calculated_hash == checkpoint.hash
        
        if not is_valid:
            logger.error(f"Checkpoint {checkpoint.id} validation failed!")
            
        return is_valid

    def export_checkpoint(self, checkpoint_id: str, export_path: Path):
        """Export a checkpoint for backup or transfer"""
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return False
            
        export_dir = export_path / f"checkpoint_{checkpoint_id}"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint files
        for file in self.checkpoint_dir.glob(f"{checkpoint_id}.*"):
            shutil.copy2(file, export_dir)
            
        # Create metadata
        metadata = {
            "exported_at": datetime.now().isoformat(),
            "project_name": self.project_name,
            "checkpoint_id": checkpoint_id,
            "iteration": checkpoint.iteration,
            "metrics": checkpoint.metrics
        }
        
        with open(export_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Exported checkpoint {checkpoint_id} to {export_dir}")
        return True
