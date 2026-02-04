"""
Adaptiivinen mallien vaihto - vaihtaa LLM-mallia automaattisesti kun agentti jumittuu.

Triggerit:
- Sama virhe toistuu N kertaa
- Reward ei parane N iteraatiossa
- Repetition loop havaittu (Coderilla)
- Timeout toistuu

Transhumanistinen kokeilu: abliterated vs normaali - miten "vapautettu ajattelu"
vaikuttaa ongelmanratkaisuun?
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import random
import time
from rich.console import Console

console = Console()


class SwitchTrigger(Enum):
    """Syyt mallin vaihdolle"""
    REPETITION_LOOP = "repetition_loop"
    REPEATED_ERROR = "repeated_error"
    NO_REWARD_IMPROVEMENT = "no_reward_improvement"
    TIMEOUT = "timeout"
    MANUAL = "manual"


@dataclass
class AgentStuckState:
    """Yksittäisen agentin jumiutumistila"""
    agent_name: str
    current_model: str = ""

    # Virheseuranta
    recent_errors: List[str] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)

    # Reward-seuranta (Testerille)
    recent_rewards: List[float] = field(default_factory=list)
    reward_history_window: int = 5

    # Repetition-seuranta (Coderille)
    repetition_loop_count: int = 0

    # Timeout-seuranta
    consecutive_timeouts: int = 0

    # Mallinvaihtohistoria
    switch_history: List[Dict] = field(default_factory=list)

    def add_error(self, error_msg: str, threshold: int = 2) -> bool:
        """Lisää virhe. Palauttaa True jos sama virhe N+ kertaa."""
        normalized = self._normalize_error(error_msg)
        self.recent_errors.append(normalized)
        self.error_counts[normalized] = self.error_counts.get(normalized, 0) + 1

        # Pidä vain viimeiset 10 virhettä
        if len(self.recent_errors) > 10:
            old = self.recent_errors.pop(0)
            self.error_counts[old] = max(0, self.error_counts.get(old, 1) - 1)

        return self.error_counts[normalized] >= threshold

    def add_reward(self, reward: float) -> bool:
        """Lisää reward. Palauttaa True jos ei parannusta N iteraatiossa."""
        self.recent_rewards.append(reward)

        if len(self.recent_rewards) > self.reward_history_window:
            self.recent_rewards.pop(0)

        if len(self.recent_rewards) >= self.reward_history_window:
            # Vertaa viimeistä keskiarvoon
            avg = sum(self.recent_rewards[:-1]) / (len(self.recent_rewards) - 1)
            latest = self.recent_rewards[-1]
            # Ei 5% parannusta -> jumissa
            if latest <= avg * 1.05:
                return True
        return False

    def record_repetition_loop(self, threshold: int = 2) -> bool:
        """Kirjaa repetition loop. Palauttaa True jos N+ peräkkäin."""
        self.repetition_loop_count += 1
        return self.repetition_loop_count >= threshold

    def clear_repetition_loop(self):
        """Nollaa repetition loop laskuri"""
        self.repetition_loop_count = 0

    def record_timeout(self, threshold: int = 2) -> bool:
        """Kirjaa timeout. Palauttaa True jos N+ peräkkäin."""
        self.consecutive_timeouts += 1
        return self.consecutive_timeouts >= threshold

    def clear_timeout(self):
        """Nollaa timeout laskuri"""
        self.consecutive_timeouts = 0

    def reset_on_success(self):
        """Nollaa kaikki laskurit onnistumisen jälkeen"""
        self.recent_errors.clear()
        self.error_counts.clear()
        self.repetition_loop_count = 0
        self.consecutive_timeouts = 0

    def _normalize_error(self, error: str) -> str:
        """Normalisoi virheviesti vertailua varten"""
        import re
        # Poista rivinumerot ja polut
        error = re.sub(r'line \d+', 'line X', error)
        error = re.sub(r'[A-Za-z]:[\\\/][^\s"\']+', 'PATH', error)
        error = re.sub(r'/[^\s"\']+', 'PATH', error)
        return error[:200].strip()


class ModelSwitcher:
    """
    Pääluokka adaptiiviselle mallinvaihdolle.

    Valitsee SATUNNAISESTI uuden mallin poolista kun jumi havaitaan.
    Tämä tuo vaihtelua ja estää "echo chamber" -efektin.

    CHAOS MODE: Kun chaos_mode=true, arpoo mallin JOKA kutsulla!
    Maksimi emergenssi - eri malli joka kerta, historia pysyy mutta ajattelija vaihtuu.
    """

    def __init__(self, config):
        self.config = config
        self.agent_states: Dict[str, AgentStuckState] = {}
        self.model_pools: Dict[str, List[str]] = {}
        self.triggers_config: Dict[str, int] = {}
        self.enabled = False
        self.chaos_mode = False  # 🎲 CHAOS MODE
        self._load_config()

    def _load_config(self):
        """Lataa konfiguraatio project.yaml:sta"""
        # Tarkista onko adaptive_model_switching olemassa
        if not hasattr(self.config, 'adaptive_model_switching'):
            console.print("[dim]Adaptive model switching not configured[/dim]")
            return

        adaptive_config = self.config.adaptive_model_switching

        if not adaptive_config.enabled:
            console.print("[dim]Adaptive model switching disabled[/dim]")
            return

        self.enabled = True

        # 🎲 CHAOS MODE - arpoo mallin JOKA kutsulla
        self.chaos_mode = getattr(adaptive_config, 'chaos_mode', False)

        # Lataa triggerit
        self.triggers_config = {
            'repeated_error_threshold': adaptive_config.triggers.repeated_error_threshold,
            'no_improvement_iterations': adaptive_config.triggers.no_improvement_iterations,
            'repetition_loop_threshold': adaptive_config.triggers.repetition_loop_threshold,
            'timeout_threshold': adaptive_config.triggers.timeout_threshold,
        }

        # Lataa mallipoolit per agentti
        for agent_name, pool_config in adaptive_config.model_pools.items():
            if pool_config.models:
                self.model_pools[agent_name] = pool_config.models
                # Alusta tila - aloita ensimmäisellä mallilla
                self.agent_states[agent_name] = AgentStuckState(
                    agent_name=agent_name,
                    current_model=pool_config.models[0],
                    reward_history_window=self.triggers_config['no_improvement_iterations']
                )

        # Lokita konfiguraatio
        console.print("\n[bold cyan]🎲 ADAPTIVE MODEL SWITCHING ENABLED[/bold cyan]")
        if self.chaos_mode:
            console.print("[bold magenta]   💀 CHAOS MODE ACTIVE - random model EVERY call![/bold magenta]")
        for agent, models in self.model_pools.items():
            unique_models = set(models)
            if len(unique_models) <= 1:
                console.print(f"  [yellow]{agent}:[/yellow] {len(models)} model(s) in pool [dim red](⚠ switching disabled - only one unique model)[/dim red]")
            else:
                console.print(f"  [yellow]{agent}:[/yellow] {len(models)} models in pool")
            for m in models:
                abliterated = "🔓" if "abliterated" in m.lower() else "🔒"
                console.print(f"    {abliterated} {m}")
        console.print()

    def get_current_model(self, agent_name: str) -> Optional[str]:
        """Hae agentin nykyinen malli"""
        if agent_name in self.agent_states:
            return self.agent_states[agent_name].current_model
        return None

    def get_chaos_model(self, agent_name: str) -> Optional[str]:
        """
        🎲 CHAOS MODE: Arvo satunnainen malli poolista.
        Kutsutaan JOKA LLM-kutsulla jos chaos_mode=true.

        Palauttaa uuden mallin tai None jos chaos mode ei päällä / ei poolia.
        """
        if not self.enabled or not self.chaos_mode:
            return None

        if agent_name not in self.model_pools:
            return None

        pool = self.model_pools[agent_name]
        if not pool:
            return None

        # Arvo satunnainen malli
        old_model = self.agent_states[agent_name].current_model if agent_name in self.agent_states else "unknown"

        # Jos poolissa vain yksi malli (tai kaikki samoja), ei vaihtoa
        unique_models = set(pool)
        if len(unique_models) <= 1 and (not unique_models or old_model in unique_models):
            return None

        new_model = random.choice(pool)

        # Jos arvottiin sama malli, ei tehdä turhaa "vaihtoa"
        if new_model == old_model:
            return None

        # Päivitä tila
        if agent_name in self.agent_states:
            self.agent_states[agent_name].current_model = new_model

        # Kompakti loki (ei haluta spämmiä joka kutsulla)
        old_abl = "🔓" if "abliterated" in old_model.lower() else "🔒"
        new_abl = "🔓" if "abliterated" in new_model.lower() else "🔒"
        console.print(f"[dim magenta]🎲 CHAOS: {agent_name} {old_abl}{old_model.split(':')[0]} → {new_abl}{new_model.split(':')[0]}[/dim magenta]")

        return new_model

    def check_and_switch(
        self,
        agent_name: str,
        trigger: SwitchTrigger,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Tarkista pitääkö vaihtaa mallia ja vaihda tarvittaessa.

        Palauttaa uuden mallin nimen jos vaihto tapahtui, muuten None.
        """
        if not self.enabled:
            return None

        if agent_name not in self.model_pools:
            return None

        state = self.agent_states[agent_name]
        pool = self.model_pools[agent_name]
        should_switch = False
        context = context or {}

        # Tarkista triggeri
        if trigger == SwitchTrigger.REPEATED_ERROR:
            threshold = self.triggers_config['repeated_error_threshold']
            should_switch = state.add_error(context.get("error", ""), threshold)

        elif trigger == SwitchTrigger.NO_REWARD_IMPROVEMENT:
            should_switch = state.add_reward(context.get("reward", 0))

        elif trigger == SwitchTrigger.REPETITION_LOOP:
            threshold = self.triggers_config['repetition_loop_threshold']
            should_switch = state.record_repetition_loop(threshold)

        elif trigger == SwitchTrigger.TIMEOUT:
            threshold = self.triggers_config['timeout_threshold']
            should_switch = state.record_timeout(threshold)

        elif trigger == SwitchTrigger.MANUAL:
            should_switch = True

        if should_switch:
            return self._do_switch(agent_name, trigger, context)

        return None

    def _do_switch(
        self,
        agent_name: str,
        trigger: SwitchTrigger,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Suorita mallinvaihto - valitse SATUNNAISESTI poolista. Palauttaa None jos vaihtoa ei tapahdu."""
        state = self.agent_states[agent_name]
        pool = self.model_pools[agent_name]
        old_model = state.current_model

        # Valitse SATUNNAISESTI eri malli kuin nykyinen
        available = [m for m in pool if m != old_model]
        if not available:
            # Vain yksi malli poolissa - ei vaihtoa, turha "vaihtaa" samaan
            return None

        new_model = random.choice(available)

        # Päivitä tila
        state.current_model = new_model
        state.switch_history.append({
            "timestamp": time.time(),
            "trigger": trigger.value,
            "from_model": old_model,
            "to_model": new_model,
            "context": context
        })

        # Nollaa laskurit vaihdon jälkeen
        state.reset_on_success()

        # Lokita vaihto
        self._log_switch(agent_name, old_model, new_model, trigger, context)

        return new_model

    def _log_switch(
        self,
        agent_name: str,
        old_model: str,
        new_model: str,
        trigger: SwitchTrigger,
        context: Dict[str, Any]
    ):
        """Tulosta näyttävä lokiviesti mallinvaihdosta"""
        emoji_map = {
            SwitchTrigger.REPETITION_LOOP: "🔄",
            SwitchTrigger.REPEATED_ERROR: "❌",
            SwitchTrigger.NO_REWARD_IMPROVEMENT: "📉",
            SwitchTrigger.TIMEOUT: "⏰",
            SwitchTrigger.MANUAL: "👆",
        }
        emoji = emoji_map.get(trigger, "🔀")

        # Tarkista onko abliterated
        old_abliterated = "🔓" if "abliterated" in old_model.lower() else "🔒"
        new_abliterated = "🔓" if "abliterated" in new_model.lower() else "🔒"

        console.print()
        console.print("═" * 70)
        console.print(f"[bold yellow]{emoji} ADAPTIVE MODEL SWITCH - {agent_name.upper()}[/bold yellow]")
        console.print("═" * 70)
        console.print(f"[yellow]Trigger:[/yellow] {trigger.value}")
        console.print(f"[red]From:[/red] {old_abliterated} {old_model}")
        console.print(f"[green]To:[/green]   {new_abliterated} {new_model}")

        # Lisätietoa kontekstista
        if trigger == SwitchTrigger.REPEATED_ERROR and "error" in context:
            error_preview = context["error"][:100]
            console.print(f"[dim]Error: {error_preview}...[/dim]")
        elif trigger == SwitchTrigger.REPETITION_LOOP:
            console.print(f"[dim]Detected code repetition loop[/dim]")
        elif trigger == SwitchTrigger.NO_REWARD_IMPROVEMENT:
            console.print(f"[dim]Reward stagnated at {context.get('reward', '?')}[/dim]")
        elif trigger == SwitchTrigger.TIMEOUT:
            console.print(f"[dim]Execution timeout[/dim]")

        console.print("═" * 70)
        console.print()

    def report_success(self, agent_name: str):
        """Raportoi onnistuminen - nollaa laskurit"""
        if agent_name in self.agent_states:
            self.agent_states[agent_name].reset_on_success()

    def get_switch_history(self, agent_name: str = None) -> List[Dict]:
        """Hae mallinvaihtohistoria"""
        if agent_name:
            if agent_name in self.agent_states:
                return self.agent_states[agent_name].switch_history
            return []

        # Kaikki agentit
        all_history = []
        for state in self.agent_states.values():
            all_history.extend(state.switch_history)
        return sorted(all_history, key=lambda x: x["timestamp"])

    def print_summary(self):
        """Tulosta yhteenveto mallinvaihdoista"""
        history = self.get_switch_history()
        if not history:
            console.print("[dim]No model switches occurred[/dim]")
            return

        console.print("\n[bold cyan]📊 MODEL SWITCH SUMMARY[/bold cyan]")
        console.print("─" * 50)

        # Laske per agentti
        agent_counts: Dict[str, int] = {}
        trigger_counts: Dict[str, int] = {}

        for entry in history:
            # Parsitaan agentti from/to mallista
            for agent_name, state in self.agent_states.items():
                if entry in state.switch_history:
                    agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
                    break

            trigger_counts[entry["trigger"]] = trigger_counts.get(entry["trigger"], 0) + 1

        console.print(f"[yellow]Total switches:[/yellow] {len(history)}")
        console.print("\n[yellow]By agent:[/yellow]")
        for agent, count in agent_counts.items():
            console.print(f"  {agent}: {count}")

        console.print("\n[yellow]By trigger:[/yellow]")
        for trigger, count in trigger_counts.items():
            console.print(f"  {trigger}: {count}")
        console.print("─" * 50)
