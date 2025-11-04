"""
Pattern Recognition Engine - Learns from successes and failures
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from loguru import logger
import pickle
from pathlib import Path


@dataclass
class Pattern:
    """Represents a recognized pattern"""
    id: str
    type: str
    context: Dict
    solution: Dict
    success_rate: float
    usage_count: int
    created_at: datetime
    last_used: datetime
    tags: List[str] = field(default_factory=list)


@dataclass
class LearningEvent:
    """Represents a learning event"""
    timestamp: datetime
    event_type: str  # success, failure, improvement
    pattern_id: Optional[str]
    context: Dict
    outcome: Dict
    confidence: float


class PatternRecognitionEngine:
    """Engine for recognizing and learning patterns"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.patterns: Dict[str, Pattern] = {}
        self.learning_events: List[LearningEvent] = []
        self.pattern_index = defaultdict(list)  # Type -> Pattern IDs
        self.similarity_threshold = 0.7
        
        # Persistence
        self.storage_path = Path(f"persistence/learning/{project_name}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Learning parameters
        self.min_confidence = 0.6
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
        # Statistics
        self.stats = {
            'patterns_recognized': 0,
            'successful_applications': 0,
            'failed_applications': 0,
            'average_success_rate': 0.0
        }
        
        self._load_patterns()
    
    def _load_patterns(self):
        """Load existing patterns from storage"""
        pattern_file = self.storage_path / "patterns.pkl"
        
        if pattern_file.exists():
            try:
                with open(pattern_file, 'rb') as f:
                    data = pickle.load(f)
                    self.patterns = data['patterns']
                    self.learning_events = data['events']
                    self.stats = data['stats']
                    
                    # Rebuild index
                    for pattern_id, pattern in self.patterns.items():
                        self.pattern_index[pattern.type].append(pattern_id)
                        
                logger.info(f"Loaded {len(self.patterns)} patterns")
            except Exception as e:
                logger.error(f"Failed to load patterns: {e}")
    
    def _save_patterns(self):
        """Save patterns to storage"""
        pattern_file = self.storage_path / "patterns.pkl"
        
        try:
            data = {
                'patterns': self.patterns,
                'events': self.learning_events[-1000:],  # Keep last 1000 events
                'stats': self.stats
            }
            
            with open(pattern_file, 'wb') as f:
                pickle.dump(data, f)
                
            logger.debug("Patterns saved successfully")
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    def recognize_pattern(self, context: Dict, problem_type: str) -> Optional[Pattern]:
        """Recognize a pattern from context"""
        # Look for similar patterns
        candidate_patterns = self.pattern_index.get(problem_type, [])
        
        best_match = None
        best_similarity = 0
        
        for pattern_id in candidate_patterns:
            pattern = self.patterns[pattern_id]
            similarity = self._calculate_similarity(context, pattern.context)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = pattern
        
        if best_match:
            logger.info(f"Recognized pattern {best_match.id} with similarity {best_similarity:.2f}")
            self.stats['patterns_recognized'] += 1
            
            # Update usage
            best_match.usage_count += 1
            best_match.last_used = datetime.now()
            
            self._save_patterns()
            
        return best_match
    
    def learn_from_success(self, context: Dict, solution: Dict, problem_type: str):
        """Learn from a successful solution"""
        # Check if similar pattern exists
        existing_pattern = self._find_similar_pattern(context, problem_type)
        
        if existing_pattern:
            # Update existing pattern
            self._update_pattern_success(existing_pattern, solution)
        else:
            # Create new pattern
            new_pattern = self._create_pattern(context, solution, problem_type)
            self.patterns[new_pattern.id] = new_pattern
            self.pattern_index[problem_type].append(new_pattern.id)
            
        # Record learning event
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type='success',
            pattern_id=existing_pattern.id if existing_pattern else new_pattern.id,
            context=context,
            outcome=solution,
            confidence=0.9
        )
        
        self.learning_events.append(event)
        self.stats['successful_applications'] += 1
        
        self._update_statistics()
        self._save_patterns()
        
        logger.info(f"Learned from success in {problem_type}")
    
    def learn_from_failure(self, context: Dict, attempted_solution: Dict, error: str, problem_type: str):
        """Learn from a failed solution"""
        # Find the pattern that led to failure
        failed_pattern = self._find_similar_pattern(context, problem_type)
        
        if failed_pattern:
            # Reduce confidence in this pattern
            self._update_pattern_failure(failed_pattern)
            
            # Create anti-pattern
            anti_pattern = self._create_anti_pattern(context, attempted_solution, error, problem_type)
            self.patterns[anti_pattern.id] = anti_pattern
        
        # Record learning event
        event = LearningEvent(
            timestamp=datetime.now(),
            event_type='failure',
            pattern_id=failed_pattern.id if failed_pattern else None,
            context=context,
            outcome={'error': error, 'attempted': attempted_solution},
            confidence=0.3
        )
        
        self.learning_events.append(event)
        self.stats['failed_applications'] += 1
        
        self._update_statistics()
        self._save_patterns()
        
        logger.info(f"Learned from failure in {problem_type}")
    
    def suggest_solution(self, context: Dict, problem_type: str) -> Optional[Dict]:
        """Suggest a solution based on learned patterns"""
        # Find matching pattern
        pattern = self.recognize_pattern(context, problem_type)
        
        if pattern and pattern.success_rate >= self.min_confidence:
            # Adapt solution to current context
            adapted_solution = self._adapt_solution(pattern.solution, context)
            
            return {
                'solution': adapted_solution,
                'confidence': pattern.success_rate,
                'pattern_id': pattern.id,
                'usage_count': pattern.usage_count,
                'explanation': f"Based on pattern with {pattern.success_rate:.1%} success rate"
            }
        
        return None
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics about learned patterns"""
        if not self.patterns:
            return {'status': 'no_patterns'}
        
        success_rates = [p.success_rate for p in self.patterns.values()]
        usage_counts = [p.usage_count for p in self.patterns.values()]
        
        return {
            'total_patterns': len(self.patterns),
            'average_success_rate': np.mean(success_rates),
            'highest_success_rate': max(success_rates),
            'lowest_success_rate': min(success_rates),
            'most_used_count': max(usage_counts),
            'average_usage': np.mean(usage_counts),
            'pattern_types': list(self.pattern_index.keys()),
            'recent_recognitions': self.stats['patterns_recognized'],
            'success_failure_ratio': (
                self.stats['successful_applications'] / 
                max(1, self.stats['failed_applications'])
            )
        }
    
    def export_patterns(self, output_path: Path) -> bool:
        """Export patterns for sharing or backup"""
        try:
            export_data = {
                'patterns': [
                    {
                        'id': p.id,
                        'type': p.type,
                        'context': p.context,
                        'solution': p.solution,
                        'success_rate': p.success_rate,
                        'usage_count': p.usage_count,
                        'tags': p.tags
                    }
                    for p in self.patterns.values()
                ],
                'statistics': self.get_pattern_statistics(),
                'export_date': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.patterns)} patterns to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export patterns: {e}")
            return False
    
    def import_patterns(self, input_path: Path, merge: bool = True) -> bool:
        """Import patterns from file"""
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for pattern_data in import_data['patterns']:
                pattern_id = pattern_data['id']
                
                # Skip if exists and not merging
                if pattern_id in self.patterns and not merge:
                    continue
                
                pattern = Pattern(
                    id=pattern_id,
                    type=pattern_data['type'],
                    context=pattern_data['context'],
                    solution=pattern_data['solution'],
                    success_rate=pattern_data['success_rate'],
                    usage_count=pattern_data.get('usage_count', 0),
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    tags=pattern_data.get('tags', [])
                )
                
                self.patterns[pattern_id] = pattern
                self.pattern_index[pattern.type].append(pattern_id)
                imported_count += 1
            
            self._save_patterns()
            
            logger.info(f"Imported {imported_count} patterns from {input_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import patterns: {e}")
            return False
    
    def _calculate_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two contexts"""
        # Convert to strings for comparison
        str1 = json.dumps(context1, sort_keys=True)
        str2 = json.dumps(context2, sort_keys=True)
        
        # Simple similarity based on common keys and values
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        # Key similarity
        key_similarity = len(keys1 & keys2) / len(keys1 | keys2)
        
        # Value similarity for common keys
        common_keys = keys1 & keys2
        if common_keys:
            value_matches = sum(
                1 for k in common_keys 
                if str(context1.get(k)) == str(context2.get(k))
            )
            value_similarity = value_matches / len(common_keys)
        else:
            value_similarity = 0
        
        # Combined similarity
        return 0.6 * key_similarity + 0.4 * value_similarity
    
    def _find_similar_pattern(self, context: Dict, problem_type: str) -> Optional[Pattern]:
        """Find pattern similar to given context"""
        return self.recognize_pattern(context, problem_type)
    
    def _update_pattern_success(self, pattern: Pattern, solution: Dict):
        """Update pattern after successful application"""
        # Update success rate with exponential moving average
        pattern.success_rate = (
            pattern.success_rate * (1 - self.learning_rate) + 
            1.0 * self.learning_rate
        )
        
        # Merge solution improvements
        for key, value in solution.items():
            if key not in pattern.solution:
                pattern.solution[key] = value
    
    def _update_pattern_failure(self, pattern: Pattern):
        """Update pattern after failed application"""
        # Reduce success rate
        pattern.success_rate = (
            pattern.success_rate * (1 - self.learning_rate) + 
            0.0 * self.learning_rate
        )
        
        # Apply decay to prevent pattern from being completely abandoned
        pattern.success_rate = max(
            0.1,  # Minimum success rate
            pattern.success_rate * self.decay_factor
        )
    
    def _create_pattern(self, context: Dict, solution: Dict, problem_type: str) -> Pattern:
        """Create a new pattern"""
        # Generate unique ID
        pattern_str = f"{problem_type}_{json.dumps(context, sort_keys=True)}_{datetime.now().isoformat()}"
        pattern_id = hashlib.md5(pattern_str.encode()).hexdigest()[:12]
        
        return Pattern(
            id=pattern_id,
            type=problem_type,
            context=context,
            solution=solution,
            success_rate=0.7,  # Initial confidence
            usage_count=1,
            created_at=datetime.now(),
            last_used=datetime.now(),
            tags=self._extract_tags(context, solution)
        )
    
    def _create_anti_pattern(self, context: Dict, solution: Dict, error: str, problem_type: str) -> Pattern:
        """Create an anti-pattern from failure"""
        pattern = self._create_pattern(context, solution, f"anti_{problem_type}")
        pattern.success_rate = 0.0  # This pattern should be avoided
        pattern.tags.append('anti_pattern')
        pattern.solution['error'] = error
        pattern.solution['avoid'] = True
        
        return pattern
    
    def _adapt_solution(self, solution: Dict, context: Dict) -> Dict:
        """Adapt a solution to current context"""
        adapted = solution.copy()
        
        # Replace template variables with context values
        for key, value in adapted.items():
            if isinstance(value, str) and '{' in value:
                try:
                    adapted[key] = value.format(**context)
                except KeyError:
                    pass  # Keep original if can't format
        
        return adapted
    
    def _extract_tags(self, context: Dict, solution: Dict) -> List[str]:
        """Extract tags from context and solution"""
        tags = []
        
        # Extract from context keys
        for key in context.keys():
            if any(keyword in key.lower() for keyword in ['error', 'bug', 'feature', 'performance']):
                tags.append(key.lower())
        
        # Extract from solution keys
        for key in solution.keys():
            if any(keyword in key.lower() for keyword in ['fix', 'optimize', 'refactor', 'improve']):
                tags.append(key.lower())
        
        return list(set(tags))  # Remove duplicates
    
    def _update_statistics(self):
        """Update overall statistics"""
        if self.patterns:
            success_rates = [p.success_rate for p in self.patterns.values()]
            self.stats['average_success_rate'] = np.mean(success_rates)
    
    def cleanup_old_patterns(self, days: int = 30) -> int:
        """Remove patterns not used in specified days"""
        cutoff_date = datetime.now().timestamp() - (days * 86400)
        removed = 0
        
        patterns_to_remove = []
        for pattern_id, pattern in self.patterns.items():
            if pattern.last_used.timestamp() < cutoff_date and pattern.usage_count < 5:
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
            # Remove from index
            for pattern_list in self.pattern_index.values():
                if pattern_id in pattern_list:
                    pattern_list.remove(pattern_id)
            removed += 1
        
        if removed > 0:
            self._save_patterns()
            logger.info(f"Cleaned up {removed} old patterns")
        
        return removed
    
    def get_most_successful_patterns(self, limit: int = 10) -> List[Pattern]:
        """Get the most successful patterns"""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: (p.success_rate * p.usage_count),
            reverse=True
        )
        
        return sorted_patterns[:limit]
    
    def get_pattern_by_type(self, problem_type: str) -> List[Pattern]:
        """Get all patterns for a specific problem type"""
        pattern_ids = self.pattern_index.get(problem_type, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]