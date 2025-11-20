#!/usr/bin/env python3
"""
Gemini Session Wrapper - Test Script
Emulates interactive mode with conversation history
"""
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime

HISTORY_PATH = Path("gemini_test_session.json")

def load_history():
    """Load existing conversation history"""
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())
    return []

def save_history(history):
    """Save conversation history to disk"""
    HISTORY_PATH.write_text(json.dumps(history, indent=2))

def build_conversation_prompt(history):
    """Build conversation prompt with history context"""
    if len(history) <= 1:
        # First message, no history
        return history[0]["content"]

    # Multi-turn conversation
    prompt_parts = ["[Conversation History]"]

    # Include all previous messages
    for msg in history[:-1]:  # Exclude the last message (current)
        role = msg["role"].capitalize()
        content = msg["content"]
        prompt_parts.append(f"{role}: {content}")

    # Add current message
    prompt_parts.append(f"\nUser: {history[-1]['content']}")
    prompt_parts.append("\nAssistant:")

    return "\n\n".join(prompt_parts)

def call_gemini(conversation_prompt):
    """Call Gemini CLI with conversation context"""
    result = subprocess.run(
        ["gemini", "-p", conversation_prompt],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        error = result.stderr.strip()
        raise RuntimeError(f"Gemini CLI failed: {error}")

    return result.stdout.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_session_test.py 'your message here'")
        print("   or: python gemini_session_test.py --clear  (to clear history)")
        sys.exit(1)

    # Handle clear command
    if sys.argv[1] == "--clear":
        if HISTORY_PATH.exists():
            HISTORY_PATH.unlink()
        print("✅ Session history cleared")
        sys.exit(0)

    # Get user message
    user_message = " ".join(sys.argv[1:])

    # Load history
    history = load_history()

    # Add user message
    history.append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })

    # Build conversation prompt
    conversation_prompt = build_conversation_prompt(history)

    print(f"\n🤖 Calling Gemini (session with {len(history)} messages)...")
    print("=" * 60)

    try:
        # Call Gemini with full conversation context
        reply = call_gemini(conversation_prompt)

        # Add assistant response to history
        history.append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })

        # Save updated history
        save_history(history)

        # Print only the latest model answer
        print(reply)
        print("=" * 60)
        print(f"✅ Session saved ({len(history)} messages total)")

    except RuntimeError as e:
        # Don't save failed messages
        history.pop()  # Remove the user message we just added
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        history.pop()
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
