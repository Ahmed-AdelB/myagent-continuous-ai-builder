"""
Tests for Pattern Recognition Engine

Validates that the system correctly identifies:
- Code patterns from successful implementations
- Error patterns and solutions
- Performance optimization patterns
"""

import pytest
import asyncio
from datetime import datetime
from core.learning.pattern_recognition import PatternRecognizer, Pattern, PatternType


class TestPatternRecognizer:
    """Test suite for PatternRecognizer"""

    @pytest.fixture
    def recognizer(self):
        """Create a fresh PatternRecognizer instance"""
        return PatternRecognizer()

    @pytest.fixture
    def sample_code_pattern(self):
        """Sample code pattern for testing"""
        return {
            'code': 'def process_data(data):\n    return [x * 2 for x in data]',
            'context': 'data transformation',
            'success_rate': 0.95,
            'occurrences': 10
        }

    @pytest.fixture
    def sample_error_pattern(self):
        """Sample error pattern for testing"""
        return {
            'error_type': 'TypeError',
            'error_message': 'unsupported operand type',
            'solution': 'Add type checking before operation',
            'occurrences': 3
        }

    def test_initialization(self, recognizer):
        """Test PatternRecognizer initializes correctly"""
        assert recognizer is not None
        assert hasattr(recognizer, 'patterns')
        assert len(recognizer.patterns) == 0

    @pytest.mark.asyncio
    async def test_add_pattern(self, recognizer, sample_code_pattern):
        """Test adding a new pattern"""
        pattern_id = await recognizer.add_pattern(
            pattern_type=PatternType.CODE,
            content=sample_code_pattern['code'],
            context=sample_code_pattern['context']
        )

        assert pattern_id is not None
        assert len(recognizer.patterns) == 1

    @pytest.mark.asyncio
    async def test_find_similar_patterns(self, recognizer, sample_code_pattern):
        """Test finding similar patterns"""
        # Add initial pattern
        await recognizer.add_pattern(
            pattern_type=PatternType.CODE,
            content=sample_code_pattern['code'],
            context=sample_code_pattern['context']
        )

        # Find similar
        similar = await recognizer.find_similar_patterns(
            content='def transform(items):\n    return [i * 2 for i in items]',
            pattern_type=PatternType.CODE
        )

        assert len(similar) > 0
        assert similar[0]['similarity'] > 0.5

    @pytest.mark.asyncio
    async def test_pattern_evolution(self, recognizer, sample_code_pattern):
        """Test that patterns evolve with more occurrences"""
        pattern_id = await recognizer.add_pattern(
            pattern_type=PatternType.CODE,
            content=sample_code_pattern['code'],
            context=sample_code_pattern['context']
        )

        # Record success
        await recognizer.record_pattern_success(pattern_id)
        await recognizer.record_pattern_success(pattern_id)

        pattern = recognizer.patterns[pattern_id]
        assert pattern['occurrences'] == 3  # Initial + 2
        assert pattern['success_count'] == 2

    @pytest.mark.asyncio
    async def test_error_pattern_detection(self, recognizer, sample_error_pattern):
        """Test error pattern detection and solution suggestion"""
        # Add error pattern
        pattern_id = await recognizer.add_pattern(
            pattern_type=PatternType.ERROR,
            content=f"{sample_error_pattern['error_type']}: {sample_error_pattern['error_message']}",
            context={'solution': sample_error_pattern['solution']}
        )

        # Find solution for similar error
        solutions = await recognizer.suggest_solutions(
            error_type=sample_error_pattern['error_type'],
            error_message='unsupported operand type for int and str'
        )

        assert len(solutions) > 0
        assert sample_error_pattern['solution'] in solutions[0]['solution']

    def test_pattern_persistence(self, recognizer, tmp_path):
        """Test saving and loading patterns"""
        # Add pattern
        recognizer.add_pattern_sync(
            pattern_type=PatternType.CODE,
            content='test code',
            context='test'
        )

        # Save
        save_path = tmp_path / "patterns.json"
        recognizer.save_patterns(str(save_path))

        # Load into new recognizer
        new_recognizer = PatternRecognizer()
        new_recognizer.load_patterns(str(save_path))

        assert len(new_recognizer.patterns) == 1

    @pytest.mark.asyncio
    async def test_pattern_ranking(self, recognizer):
        """Test patterns are ranked by success rate"""
        # Add multiple patterns with different success rates
        await recognizer.add_pattern(
            pattern_type=PatternType.CODE,
            content='pattern 1',
            context='low success'
        )
        await recognizer.add_pattern(
            pattern_type=PatternType.CODE,
            content='pattern 2',
            context='high success'
        )

        # Record different success rates
        await recognizer.record_pattern_success('pattern_0')  # 1/1 = 100%
        await recognizer.record_pattern_failure('pattern_0')  # 1/2 = 50%

        await recognizer.record_pattern_success('pattern_1')  # 1/1 = 100%
        await recognizer.record_pattern_success('pattern_1')  # 2/2 = 100%

        # Get top patterns
        top_patterns = recognizer.get_top_patterns(n=2)

        assert top_patterns[0]['id'] == 'pattern_1'  # Higher success rate
        assert top_patterns[0]['success_rate'] > top_patterns[1]['success_rate']

    @pytest.mark.asyncio
    async def test_pattern_lifecycle(self, recognizer):
        """Test complete pattern lifecycle: create → use → evolve → deprecate"""
        # Create
        pattern_id = await recognizer.add_pattern(
            pattern_type=PatternType.CODE,
            content='old_pattern',
            context='legacy'
        )

        # Use successfully multiple times
        for _ in range(10):
            await recognizer.record_pattern_success(pattern_id)

        # Pattern becomes established
        pattern = recognizer.patterns[pattern_id]
        assert pattern['occurrences'] >= 10
        assert pattern['status'] == 'established'

        # Deprecate after failures
        for _ in range(5):
            await recognizer.record_pattern_failure(pattern_id)

        # Check if flagged for review
        assert recognizer.needs_review(pattern_id)
