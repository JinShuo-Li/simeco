"""Curses observer for SSH-friendly interactive runs."""

from __future__ import annotations

import curses
import time
from pathlib import Path

from .simulation import Simulation
from .snapshot import load_snapshot, save_snapshot

SPARKS = "._-~=+*#%@"
PLANTS = " .,:;ox"


def sparkline(values: list[float], width: int) -> str:
    if width <= 0 or not values:
        return ""
    values = values[-width:]
    low, high = min(values), max(values)
    if high == low:
        return SPARKS[len(SPARKS) // 2] * len(values)
    return "".join(SPARKS[min(len(SPARKS) - 1, int((value - low) / (high - low) * (len(SPARKS) - 1)))] for value in values)


class TerminalUI:
    def __init__(self, simulation: Simulation, snapshot_path: str | Path, max_steps: int | None = None):
        self.simulation = simulation
        self.snapshot_path = Path(snapshot_path)
        self.max_steps = max_steps
        self.paused = False
        self.speeds = [1, 4, 15, 60, 240]
        self.speed_index = 2
        self.selected_id: int | None = None
        self.message = ""

    @property
    def speed(self) -> int:
        return self.speeds[self.speed_index]

    def run(self, screen) -> None:
        curses.curs_set(0)
        screen.nodelay(True)
        screen.timeout(40)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
        running = True
        while running:
            key = screen.getch()
            running = self._key(key)
            if not self.paused and running:
                for _ in range(self.speed):
                    self.simulation.step()
                    if self.max_steps is not None and self.simulation.step_count >= self.max_steps:
                        running = False
                        break
            self._draw(screen)
            if self.speed < 100:
                time.sleep(0.025)

    def _key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self.paused = not self.paused
        elif key in (ord("+"), ord("=")):
            self.speed_index = min(len(self.speeds) - 1, self.speed_index + 1)
        elif key == ord("-"):
            self.speed_index = max(0, self.speed_index - 1)
        elif key == ord("s"):
            save_snapshot(self.simulation, self.snapshot_path)
            self.message = f"saved {self.snapshot_path}"
        elif key == ord("l"):
            if self.snapshot_path.exists():
                self.simulation = load_snapshot(self.snapshot_path)
                self.message = f"loaded step {self.simulation.step_count}"
            else:
                self.message = f"no snapshot at {self.snapshot_path}"
        elif key in (ord("i"), ord("n")):
            ids = sorted(self.simulation.organisms)
            if ids:
                if self.selected_id not in ids:
                    self.selected_id = ids[0]
                else:
                    self.selected_id = ids[(ids.index(self.selected_id) + 1) % len(ids)]
        return True

    @staticmethod
    def _put(screen, y: int, x: int, text: str, style: int = 0) -> None:
        height, width = screen.getmaxyx()
        if y < 0 or y >= height or x >= width:
            return
        try:
            screen.addnstr(y, max(0, x), text, max(0, width - max(0, x) - 1), style)
        except curses.error:
            pass

    def _draw(self, screen) -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        sim = self.simulation
        mode = "INSTINCT+LEARNING" if sim.learning else "INSTINCT ONLY"
        header = f" LIVING ECOSYSTEM V2  step {sim.step_count:,}  {'PAUSED' if self.paused else f'{self.speed}x'}  {mode} "
        self._put(screen, 0, 0, header, curses.A_REVERSE)
        map_height = min(sim.config.height, max(3, height - 5))
        side_width = 30 if width >= 76 else 0
        map_width = min(sim.config.width, max(8, width - side_width - 2))
        cells: dict[tuple[int, int], list[str]] = {}
        for animal in sim.organisms.values():
            if animal.x < map_width and animal.y < map_height:
                cells.setdefault((animal.x, animal.y), []).append(animal.species)
        for y in range(map_height):
            line = []
            styles = []
            for x in range(map_width):
                animals = cells.get((x, y), [])
                if len(animals) > 1:
                    line.append("*")
                    styles.append(3)
                elif animals and animals[0] == "predator":
                    line.append("P")
                    styles.append(3)
                elif animals:
                    line.append("h")
                    styles.append(2)
                else:
                    fraction = sim.resources[y][x] / sim.config.plant_capacity
                    line.append(PLANTS[min(len(PLANTS) - 1, int(fraction * len(PLANTS)))])
                    styles.append(1)
            for x, (character, color) in enumerate(zip(line, styles)):
                style = curses.color_pair(color) if curses.has_colors() else 0
                self._put(screen, y + 1, x, character, style)

        herbs = sim.species("herbivore")
        predators = sim.species("predator")
        history = sim.metrics.history
        if side_width:
            x = map_width + 2
            plant_mean = sum(map(sum, sim.resources)) / (sim.config.width * sim.config.height)
            hunt_rate = sim.metrics.hunts / max(1, sim.metrics.hunt_attempts)
            updates = sum(a.adaptive_policy.updates for a in sim.organisms.values())
            rewards = sum(a.lifetime_reward for a in sim.organisms.values()) / max(1, len(sim.organisms))
            lines = [
                ("POPULATION", curses.A_BOLD),
                (f"herbivores  {len(herbs):7d}", curses.color_pair(2) if curses.has_colors() else 0),
                (f"predators   {len(predators):7d}", curses.color_pair(3) if curses.has_colors() else 0),
                (f"plants/cell {plant_mean:7.2f}", curses.color_pair(1) if curses.has_colors() else 0),
                ("", 0),
                ("EVENTS (total / last tick)", curses.A_BOLD),
                (f"births {sim.metrics.births_herbivore + sim.metrics.births_predator:7d} / {sim.last_events['births']}", 0),
                (f"hunts  {sim.metrics.hunts:7d} / {sim.last_events['hunts']}", 0),
                (f"hunt success {hunt_rate:7.2%}", 0),
                (f"starved {sim.metrics.deaths_starvation:7d}", 0),
                ("", 0),
                ("LEARNING / EVOLUTION", curses.A_BOLD),
                (f"updates     {updates:7d}", 0),
                (f"mean reward {rewards:7.2f}", 0),
                (f"generation  {max((a.generation for a in sim.organisms.values()), default=0):7d}", 0),
            ]
            for index, (line, style) in enumerate(lines):
                self._put(screen, index + 1, x, line, style)

            chart_width = min(27, width - x - 1)
            chart_y = len(lines) + 2
            if chart_y + 2 <= map_height:
                self._put(screen, chart_y, x, "H " + sparkline([row["herbivores"] for row in history], chart_width - 2))
                self._put(screen, chart_y + 1, x, "P " + sparkline([row["predators"] for row in history], chart_width - 2))
                self._put(screen, chart_y + 2, x, "+ " + sparkline([row["plants"] for row in history], chart_width - 2))

        footer_y = min(height - 3, map_height + 1)
        self._put(screen, footer_y, 0, "q quit  space pause  +/- speed  s save  l load  i inspect", curses.A_REVERSE)
        selected = sim.organisms.get(self.selected_id) if self.selected_id is not None else None
        if selected:
            detail = (
                f"#{selected.id} {selected.species} E={selected.energy:.1f} age={selected.age} "
                f"gen={selected.generation} meals={selected.meals} children={selected.offspring_count} "
                f"reward={selected.lifetime_reward:.1f} updates={selected.adaptive_policy.updates}"
            )
            self._put(screen, footer_y + 1, 0, detail, curses.color_pair(4) if curses.has_colors() else 0)
        elif self.message:
            self._put(screen, footer_y + 1, 0, self.message)
        screen.refresh()


def run_tui(simulation: Simulation, snapshot_path: str | Path, max_steps: int | None = None) -> Simulation:
    ui = TerminalUI(simulation, snapshot_path, max_steps=max_steps)
    curses.wrapper(ui.run)
    return ui.simulation
