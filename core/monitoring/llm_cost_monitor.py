"""
LLM Cost Monitoring and Budget Controls
Implements comprehensive cost tracking and budget management for LLM usage
Based on 2024 best practices for enterprise AI cost management
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    COHERE = "cohere"

@dataclass
class UsageRecord:
    """Individual LLM usage record"""
    id: str
    timestamp: datetime
    provider: str
    model: str
    agent_name: str
    task_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    response_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class CostSummary:
    """Cost summary for a time period"""
    period_start: datetime
    period_end: datetime
    total_cost: Decimal
    total_tokens: int
    total_requests: int
    cost_by_provider: Dict[str, Decimal]
    cost_by_model: Dict[str, Decimal]
    cost_by_agent: Dict[str, Decimal]
    average_cost_per_request: Decimal
    average_tokens_per_request: int

class LLMCostMonitor:
    """Comprehensive LLM cost monitoring and budget management"""
    
    # Current pricing as of 2024 (update regularly)
    PRICING_TABLE = {
        "openai": {
            "gpt-5-chat-latest": {
                "input_cost_per_1k_tokens": Decimal("0.005"),
                "output_cost_per_1k_tokens": Decimal("0.020"),
                "context_window": 200000
            },
            "gpt-4o": {
                "input_cost_per_1k_tokens": Decimal("0.0025"),
                "output_cost_per_1k_tokens": Decimal("0.0100"),
                "context_window": 128000
            },
            "gpt-4o-mini": {
                "input_cost_per_1k_tokens": Decimal("0.00015"),
                "output_cost_per_1k_tokens": Decimal("0.0006"),
                "context_window": 128000
            }
        },
        "anthropic": {
            "claude-3-5-sonnet-20241022": {
                "input_cost_per_1k_tokens": Decimal("0.003"),
                "output_cost_per_1k_tokens": Decimal("0.015"),
                "context_window": 200000
            },
            "claude-3-5-haiku-20241022": {
                "input_cost_per_1k_tokens": Decimal("0.00025"),
                "output_cost_per_1k_tokens": Decimal("0.00125"),
                "context_window": 200000
            }
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.usage_records: List[UsageRecord] = []
        self.daily_budget = Decimal("100.00")
        self.monthly_budget = Decimal("2000.00")
        self.alert_thresholds = [50, 75, 90, 95]
    
    async def record_llm_usage(self, 
                              provider: str,
                              model: str,
                              agent_name: str,
                              task_type: str,
                              input_tokens: int,
                              output_tokens: int,
                              response_time: float,
                              success: bool = True,
                              error_message: Optional[str] = None) -> UsageRecord:
        """Record LLM usage and calculate costs"""
        
        # Calculate costs
        total_tokens = input_tokens + output_tokens
        pricing = self.PRICING_TABLE.get(provider, {}).get(model, {})
        
        if pricing:
            input_cost = (Decimal(input_tokens) / Decimal(1000)) * pricing["input_cost_per_1k_tokens"]
            output_cost = (Decimal(output_tokens) / Decimal(1000)) * pricing["output_cost_per_1k_tokens"]
            total_cost = input_cost + output_cost
        else:
            # Fallback pricing if model not found
            self.logger.warning(f"Pricing not found for {provider}/{model}, using fallback")
            input_cost = (Decimal(input_tokens) / Decimal(1000)) * Decimal("0.001")
            output_cost = (Decimal(output_tokens) / Decimal(1000)) * Decimal("0.003")
            total_cost = input_cost + output_cost
        
        # Create usage record
        record = UsageRecord(
            id=f"{datetime.utcnow().isoformat()}_{agent_name}_{task_type}",
            timestamp=datetime.utcnow(),
            provider=provider,
            model=model,
            agent_name=agent_name,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=input_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP),
            output_cost=output_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP),
            total_cost=total_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP),
            response_time=response_time,
            success=success,
            error_message=error_message
        )
        
        self.usage_records.append(record)
        self.logger.info(f"Recorded LLM usage: {agent_name} - {provider}/{model} - ")
        
        return record
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        now = datetime.utcnow()
        
        # Daily budget status
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_records = [r for r in self.usage_records if r.timestamp >= today_start]
        today_cost = sum(r.total_cost for r in today_records)
        
        # Monthly budget status
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_records = [r for r in self.usage_records if r.timestamp >= month_start]
        month_cost = sum(r.total_cost for r in month_records)
        
        daily_used_percent = (today_cost / self.daily_budget * 100) if self.daily_budget > 0 else 0
        monthly_used_percent = (month_cost / self.monthly_budget * 100) if self.monthly_budget > 0 else 0
        
        return {
            "daily_budget": {
                "budget": float(self.daily_budget),
                "spent": float(today_cost),
                "remaining": float(self.daily_budget - today_cost),
                "used_percent": float(daily_used_percent),
                "requests": len(today_records),
                "status": self._get_budget_status_color(daily_used_percent)
            },
            "monthly_budget": {
                "budget": float(self.monthly_budget),
                "spent": float(month_cost),
                "remaining": float(self.monthly_budget - month_cost),
                "used_percent": float(monthly_used_percent),
                "requests": len(month_records),
                "status": self._get_budget_status_color(monthly_used_percent)
            }
        }
    
    def _get_budget_status_color(self, used_percent: float) -> str:
        """Get status color based on budget usage"""
        if used_percent >= 95:
            return "critical"
        elif used_percent >= 75:
            return "warning"
        elif used_percent >= 50:
            return "caution"
        else:
            return "healthy"

# Global instance for cost monitoring
_cost_monitor = None

def get_cost_monitor() -> LLMCostMonitor:
    """Get or create global cost monitor instance"""
    global _cost_monitor
    if _cost_monitor is None:
        _cost_monitor = LLMCostMonitor()
    return _cost_monitor

# Usage tracking decorator
def track_llm_usage(provider: str, model: str, agent_name: str, task_type: str):
    """Decorator to track LLM usage"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_message = None
            input_tokens = 0
            output_tokens = 0
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract token counts from result if available
                if isinstance(result, dict) and 'usage' in result:
                    usage = result['usage']
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                response_time = time.time() - start_time
                
                # Record usage
                cost_monitor = get_cost_monitor()
                await cost_monitor.record_llm_usage(
                    provider=provider,
                    model=model,
                    agent_name=agent_name,
                    task_type=task_type,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time=response_time,
                    success=success,
                    error_message=error_message
                )
        
        return wrapper
    return decorator
