"""
GPT-5 Direct Client
Direct integration with GPT-5-chat-latest using curl commands
"""

import json
import subprocess
import os
from typing import Dict, Any, Optional, List


class GPT5DirectClient:
    """Direct client for GPT-5-chat-latest using curl"""

    def __init__(self, api_key: str, model: str = "gpt-5-chat-latest"):
        """
        Initialize GPT-5 Direct Client

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5-chat-latest)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.timeout = 60  # seconds

    def make_request(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
        """
        Make a direct request to GPT-5 using curl

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)

        Returns:
            Response content or None if failed
        """
        try:
            # Prepare request data
            request_data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            # Prepare curl command
            curl_cmd = [
                "curl",
                "-X", "POST",
                f"{self.base_url}/chat/completions",
                "-H", "Content-Type: application/json",
                "-H", f"Authorization: Bearer {self.api_key}",
                "-d", json.dumps(request_data),
                "--silent",
                "--show-error"
            ]

            # Execute curl command
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                response_data = json.loads(result.stdout)

                if "choices" in response_data and response_data["choices"]:
                    return response_data["choices"][0]["message"]["content"]
                elif "error" in response_data:
                    print(f"❌ GPT-5 API Error: {response_data['error']}")
                    return None
                else:
                    print(f"❌ Unexpected response format: {response_data}")
                    return None
            else:
                print(f"❌ Curl command failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("❌ Request timed out")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"❌ Request failed: {str(e)}")
            return None

    def analyze_system_requirements(self, project_path: str) -> Optional[str]:
        """
        Analyze system requirements from project files

        Args:
            project_path: Path to the project directory

        Returns:
            Analysis report or None if failed
        """
        try:
            # Read key project files
            files_to_analyze = [
                "CLAUDE.md",
                "README.md",
                "requirements.txt",
                "core/orchestrator/continuous_director.py"
            ]

            project_info = []
            for file_path in files_to_analyze:
                full_path = os.path.join(project_path, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()[:2000]  # Limit content size
                        project_info.append(f"=== {file_path} ===\n{content}\n")

            # Create analysis prompt
            prompt = f"""
Analyze this AI system project and provide insights:

{chr(10).join(project_info)}

Please provide:
1. System architecture overview
2. Current capabilities
3. Missing or incomplete components
4. Potential improvements
5. Priority recommendations

Focus on the continuous AI development philosophy and multi-agent architecture.
"""

            return self.make_request(prompt, max_tokens=2000, temperature=0.3)

        except Exception as e:
            print(f"❌ System analysis failed: {str(e)}")
            return None

    def get_improvement_recommendations(self, system_description: str) -> Optional[str]:
        """
        Get improvement recommendations from GPT-5

        Args:
            system_description: Description of the system to improve

        Returns:
            Improvement recommendations or None if failed
        """
        prompt = f"""
You are an expert AI systems architect. Analyze this system and provide specific improvement recommendations:

System: {system_description}

Please provide:
1. Architecture improvements
2. Performance optimizations
3. Reliability enhancements
4. Scalability recommendations
5. Security considerations
6. Development workflow improvements
7. Monitoring and observability enhancements

Prioritize recommendations by impact and implementation difficulty.
Focus on practical, actionable improvements.
"""

        return self.make_request(prompt, max_tokens=2000, temperature=0.4)

    def generate_code_improvements(self, code_snippet: str, context: str = "") -> Optional[str]:
        """
        Generate code improvements using GPT-5

        Args:
            code_snippet: The code to improve
            context: Additional context about the code

        Returns:
            Improved code or None if failed
        """
        prompt = f"""
Review and improve this code:

Context: {context}

Code:
```python
{code_snippet}
```

Please provide:
1. Code review findings
2. Improved version of the code
3. Explanation of improvements
4. Best practices applied

Focus on performance, reliability, and maintainability.
"""

        return self.make_request(prompt, max_tokens=2000, temperature=0.2)

    def design_system_architecture(self, requirements: str) -> Optional[str]:
        """
        Design system architecture using GPT-5

        Args:
            requirements: System requirements

        Returns:
            Architecture design or None if failed
        """
        prompt = f"""
Design a system architecture based on these requirements:

{requirements}

Please provide:
1. High-level architecture diagram (text description)
2. Component breakdown
3. Data flow design
4. Technology stack recommendations
5. Deployment strategy
6. Scalability considerations
7. Error handling approach

Focus on production-ready, scalable solutions.
"""

        return self.make_request(prompt, max_tokens=2500, temperature=0.3)

    def test_connection(self) -> bool:
        """
        Test the connection to GPT-5

        Returns:
            True if connection successful, False otherwise
        """
        response = self.make_request(
            "Respond with exactly: 'Connection successful'",
            max_tokens=10,
            temperature=0.0
        )

        return response is not None and "Connection successful" in response