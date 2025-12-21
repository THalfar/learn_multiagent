import time
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

@dataclass
class AgentTiming:
    """Track timing for a single agent call"""
    agent: str
    iteration: int
    start_time: float = 0
    end_time: float = 0
    duration: float = 0
    tokens_in: int = 0
    tokens_out: int = 0

@dataclass 
class RunStatistics:
    """Track statistics for entire run"""
    run_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    timings: List[AgentTiming] = field(default_factory=list)
    
    def add_timing(self, timing: AgentTiming):
        self.timings.append(timing)
    
    def get_agent_stats(self, agent: str) -> dict:
        agent_timings = [t for t in self.timings if t.agent == agent]
        if not agent_timings:
            return {}
        durations = [t.duration for t in agent_timings]
        return {
            "calls": len(agent_timings),
            "total_time": sum(durations),
            "avg_time": sum(durations) / len(agent_timings),
            "min_time": min(durations),
            "max_time": max(durations)
        }
    
    def get_iteration_stats(self, iteration: int) -> dict:
        iter_timings = [t for t in self.timings if t.iteration == iteration]
        return {
            "total_time": sum(t.duration for t in iter_timings),
            "agents": {t.agent: t.duration for t in iter_timings}
        }
    
    def print_summary(self):
        total_duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        print("\n" + "="*60)
        print("📊 RUN STATISTICS")
        print("="*60)
        print(f"Run ID: {self.run_id}")
        print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"Iterations: {max(t.iteration for t in self.timings) if self.timings else 0}")
        
        print("\n📈 PER-AGENT BREAKDOWN:")
        print("-"*60)
        print(f"{'Agent':<12} {'Calls':<8} {'Total':<10} {'Avg':<10} {'Min':<10} {'Max':<10}")
        print("-"*60)
        
        for agent in ["manager", "coder", "tester", "reviewer"]:
            stats = self.get_agent_stats(agent)
            if stats:
                print(f"{agent:<12} {stats['calls']:<8} {stats['total_time']:<10.1f}s {stats['avg_time']:<10.1f}s {stats['min_time']:<10.1f}s {stats['max_time']:<10.1f}s")
        
        print("\n⏱️  PER-ITERATION BREAKDOWN:")
        print("-"*60)
        max_iter = max(t.iteration for t in self.timings) if self.timings else 0
        for i in range(1, max_iter + 1):
            iter_stats = self.get_iteration_stats(i)
            agents_str = " | ".join(f"{k}: {v:.1f}s" for k, v in iter_stats["agents"].items())
            print(f"Iter {i}: {iter_stats['total_time']:.1f}s total [{agents_str}]")
        
        print("="*60)
    
    def save_to_file(self, filepath: str):
        import json
        data = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_s": (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            "timings": [
                {
                    "agent": t.agent,
                    "iteration": t.iteration,
                    "duration": t.duration
                } for t in self.timings
            ],
            "agent_stats": {
                agent: self.get_agent_stats(agent) 
                for agent in ["manager", "coder", "tester", "reviewer"]
            }
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)