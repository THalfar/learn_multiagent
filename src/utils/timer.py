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
    code_lines_per_iteration: Dict[int, int] = field(default_factory=dict)  # iteration -> lines of code

    def add_timing(self, timing: AgentTiming):
        self.timings.append(timing)

    def add_code_stats(self, iteration: int, lines_of_code: int):
        """Track lines of code generated per iteration"""
        self.code_lines_per_iteration[iteration] = lines_of_code

    def get_code_stats(self) -> dict:
        """Get statistics about generated code"""
        if not self.code_lines_per_iteration:
            return {}

        lines = list(self.code_lines_per_iteration.values())
        return {
            "total_iterations": len(lines),
            "total_lines": sum(lines),
            "avg_lines_per_iteration": sum(lines) / len(lines),
            "min_lines": min(lines),
            "max_lines": max(lines),
            "lines_per_iteration": self.code_lines_per_iteration
        }
    
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
    
    def get_agent_token_stats(self, agent: str) -> dict:
        """Get token statistics for a specific agent"""
        agent_timings = [t for t in self.timings if t.agent == agent]
        if not agent_timings:
            return {}
        
        tokens_in = [t.tokens_in for t in agent_timings if t.tokens_in > 0]
        tokens_out = [t.tokens_out for t in agent_timings if t.tokens_out > 0]
        
        def calc_stats(values):
            if not values:
                return {}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
            return {
                "total": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "median": median,
                "count": len(values)
            }
        
        return {
            "tokens_in": calc_stats(tokens_in),
            "tokens_out": calc_stats(tokens_out),
            "total_tokens": calc_stats([t.tokens_in + t.tokens_out for t in agent_timings if t.tokens_in > 0 or t.tokens_out > 0])
        }
    
    def get_all_agents_token_stats(self) -> dict:
        """Get token statistics for all agents"""
        return {
            agent: self.get_agent_token_stats(agent)
            for agent in ["manager", "coder", "tester", "reviewer"]
        }
    
    def get_iteration_stats(self, iteration: int) -> dict:
        iter_timings = [t for t in self.timings if t.iteration == iteration]
        agent_stats = {}
        for t in iter_timings:
            if t.agent not in agent_stats:
                agent_stats[t.agent] = {"duration": 0, "tokens_in": 0, "tokens_out": 0}
            agent_stats[t.agent]["duration"] = t.duration
            agent_stats[t.agent]["tokens_in"] = t.tokens_in
            agent_stats[t.agent]["tokens_out"] = t.tokens_out
        
        return {
            "total_time": sum(t.duration for t in iter_timings),
            "agents": {agent: stats["duration"] for agent, stats in agent_stats.items()},
            "agent_tokens": {agent: {"tokens_in": stats["tokens_in"], "tokens_out": stats["tokens_out"]} 
                           for agent, stats in agent_stats.items()}
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
        
        print("\n🔢 TOKEN STATISTICS BY AGENT:")
        print("-"*60)
        print(f"{'Agent':<12} {'Calls':<8} {'Tokens In':<20} {'Tokens Out':<20} {'Total':<20}")
        print(f"{'':<12} {'':<8} {'(total/avg/min/max)':<20} {'(total/avg/min/max)':<20} {'(total/avg/min/max)':<20}")
        print("-"*60)
        
        for agent in ["manager", "coder", "tester", "reviewer"]:
            token_stats = self.get_agent_token_stats(agent)
            if token_stats:
                tokens_in = token_stats.get("tokens_in", {})
                tokens_out = token_stats.get("tokens_out", {})
                total_tokens = token_stats.get("total_tokens", {})
                
                if tokens_in.get("count", 0) > 0:
                    in_str = f"{tokens_in['total']:,}/{tokens_in['avg']:.0f}/{tokens_in['min']:,}/{tokens_in['max']:,}"
                else:
                    in_str = "N/A"
                
                if tokens_out.get("count", 0) > 0:
                    out_str = f"{tokens_out['total']:,}/{tokens_out['avg']:.0f}/{tokens_out['min']:,}/{tokens_out['max']:,}"
                else:
                    out_str = "N/A"
                
                if total_tokens.get("count", 0) > 0:
                    total_str = f"{total_tokens['total']:,}/{total_tokens['avg']:.0f}/{total_tokens['min']:,}/{total_tokens['max']:,}"
                else:
                    total_str = "N/A"
                
                calls = len([t for t in self.timings if t.agent == agent])
                print(f"{agent:<12} {calls:<8} {in_str:<20} {out_str:<20} {total_str:<20}")
        
        # Code statistics
        code_stats = self.get_code_stats()
        if code_stats:
            print("\n💻 CODE GENERATION STATISTICS:")
            print("-"*60)
            print(f"Total iterations with code: {code_stats['total_iterations']}")
            print(f"Total lines of code generated: {code_stats['total_lines']:,}")
            print(f"Average lines per iteration: {code_stats['avg_lines_per_iteration']:.1f}")
            print(f"Min/Max lines per iteration: {code_stats['min_lines']}/{code_stats['max_lines']}")

        print("\n⏱️  PER-ITERATION BREAKDOWN:")
        print("-"*60)
        max_iter = max(t.iteration for t in self.timings) if self.timings else 0
        for i in range(1, max_iter + 1):
            iter_stats = self.get_iteration_stats(i)
            agents_str = " | ".join(f"{k}: {v:.1f}s" for k, v in iter_stats["agents"].items())
            code_lines = self.code_lines_per_iteration.get(i, 0)
            code_info = f" [{code_lines} lines]" if code_lines > 0 else ""
            print(f"Iter {i}: {iter_stats['total_time']:.1f}s total [{agents_str}]{code_info}")

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
                    "duration": t.duration,
                    "tokens_in": t.tokens_in,
                    "tokens_out": t.tokens_out
                } for t in self.timings
            ],
            "agent_stats": {
                agent: self.get_agent_stats(agent)
                for agent in ["manager", "coder", "tester", "reviewer"]
            },
            "agent_token_stats": {
                agent: self.get_agent_token_stats(agent)
                for agent in ["manager", "coder", "tester", "reviewer"]
            },
            "code_stats": self.get_code_stats()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)