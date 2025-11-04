#!/usr/bin/env python3
"""
Quick test to verify CoderAgent works with API keys from .env file
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.agents.coder_agent import CoderAgent
from config.settings import settings

async def test_coder_agent():
    """Test that CoderAgent can initialize and make a simple LLM call"""

    print("=" * 60)
    print("Testing CoderAgent with API Key Integration")
    print("=" * 60)

    # Verify API key is loaded
    if settings.OPENAI_API_KEY:
        print(f"✅ OpenAI API Key loaded: {settings.OPENAI_API_KEY[:20]}...")
    else:
        print("❌ OpenAI API Key NOT found in settings")
        return False

    # Initialize CoderAgent
    print("\n📦 Initializing CoderAgent...")
    try:
        agent = CoderAgent()
        print(f"✅ CoderAgent initialized: {agent.name}")
        print(f"   Agent ID: {agent.id}")
        print(f"   Role: {agent.role}")
    except Exception as e:
        print(f"❌ Failed to initialize CoderAgent: {e}")
        return False

    # Test simple code generation
    print("\n🤖 Testing code generation with LLM...")
    try:
        from core.agents.base_agent import AgentTask
        from datetime import datetime

        task = AgentTask(
            id="test_001",
            type="implement_feature",
            description="Create a simple add function",
            priority=1,
            data={
                "feature_name": "add_function",
                "description": "A Python function that adds two numbers and returns the result",
                "requirements": [
                    "Accept two parameters (a and b)",
                    "Return the sum of a and b",
                    "Include type hints",
                    "Include a docstring"
                ],
                "context": {},
                "code_structure": {}
            },
            created_at=datetime.now()
        )

        print(f"   Task: {task.description}")
        print("   Calling LLM... (this may take 10-30 seconds)")

        # Process the task
        result = await agent.process_task(task)

        # Result is a Dict, not an object with .success attribute
        if isinstance(result, dict):
            print(f"✅ Feature implementation successful!")
            print(f"   Result keys: {list(result.keys())}")

            # Show generated code if available
            if 'generated_code' in result:
                code = result['generated_code']
                print(f"\n   Generated code ({len(str(code))} chars):")
                print("   " + "-" * 50)
                print(str(code)[:800])  # Show first 800 chars
                print("   " + "-" * 50)
            elif 'files' in result:
                print(f"\n   Generated {len(result['files'])} file(s)")
                for filename, content in list(result['files'].items())[:1]:  # Show first file
                    print(f"   File: {filename}")
                    print("   " + "-" * 50)
                    print(str(content)[:800])  # Show first 800 chars
                    print("   " + "-" * 50)
        else:
            print(f"❌ Feature implementation failed: {result}")
            return False

    except Exception as e:
        print(f"❌ Error during code generation: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED - CoderAgent fully operational with API!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_coder_agent())
    sys.exit(0 if success else 1)
