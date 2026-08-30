"""Centralized critic for evaluating multi-agent trajectories."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base import BaseCritic, Trajectory
from .optimizer import PolicyGradientOptimizer


class PromptLoader:
    """Loader for evaluation prompts from external JSON files."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            # Default: maskills/envs/language/prompts/evaluation
            prompts_dir = Path(__file__).parent.parent / "envs" / "language" / "prompts" / "evaluation"
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}

    def _load_json(self, filepath: Path) -> dict:
        str_path = str(filepath)
        if str_path not in self._cache:
            with open(filepath, 'r') as f:
                self._cache[str_path] = json.load(f)
        return self._cache[str_path]

    def load_game_context(self, game_name: str = "language_task") -> str:
        game_file = self.prompts_dir / "game_contexts" / f"{game_name}.json"
        if not game_file.exists():
            game_file = self.prompts_dir / "game_contexts" / "default.json"
        if not game_file.exists():
            return ""
        data = self._load_json(game_file)
        return data.get("game_context", "")

    def load_evaluation_template(self, paradigm: str, task_type: str = "language") -> str:
        template_file = self.prompts_dir / "templates" / f"{task_type}_{paradigm}.json"
        if not template_file.exists():
            template_file = self.prompts_dir / "templates" / f"{paradigm}.json"
        if not template_file.exists():
            raise ValueError(f"No template file found for paradigm '{paradigm}'")
        data = self._load_json(template_file)
        return data.get("template", "")

    def clear_cache(self):
        self._cache.clear()


class CentralizedCritic(BaseCritic):
    """Centralized critic supporting central_global and central_credit paradigms.

    Evaluates N-agent sequential collaboration trajectories using LLM-as-judge.
    """

    def __init__(self, config, prompts_dir: Optional[Path] = None, tool_library: str = ""):
        """
        Args:
            config: BaseConfig instance with paradigm, num_agents, critic_llm/llm fields.
            prompts_dir: Optional custom path to prompts directory.
            tool_library: Optional markdown describing tools the agents
                can invoke. Prepended to the evaluation prompt so the
                critic can judge whether tools were used appropriately.
        """
        self.paradigm = config.paradigm
        self.num_agents = config.num_agents
        self.task_type = getattr(config, 'task_type', 'qa')
        self.all_agents = [f"agent_{i + 1}" for i in range(self.num_agents)]

        self.prompt_loader = PromptLoader(prompts_dir)
        self._load_prompts()
        self.tool_library = tool_library or ""

        llm = getattr(config, 'critic_llm', None) or config.llm
        api_key = llm.get_api_key()
        if llm.base_url:
            self._client = OpenAI(base_url=llm.base_url, api_key=api_key)
        else:
            self._client = OpenAI(api_key=api_key)
        self._model = llm.model_string
        self.logger = logging.getLogger(__name__)

    def _tool_library_block(self) -> str:
        if not self.tool_library.strip():
            return ""
        return (
            "## Tool Library (available to the agents in this environment —\n"
            "## judge whether they invoked tools when they should have, and\n"
            "## whether their queries/usage were effective)\n"
            + self.tool_library.strip()
            + "\n\n"
        )

    def _load_prompts(self):
        self.game_context = self.prompt_loader.load_game_context("language_task")
        self.evaluation_template = self.prompt_loader.load_evaluation_template(
            self.paradigm, "language"
        )

    def evaluate(self, trajectory: Trajectory, policies: dict) -> dict:
        """Evaluate a trajectory. Returns dict with 'raw_response' and 'per_agent' credits.

        Args:
            trajectory: Episode trajectory.
            policies: dict mapping agent_name -> AgentPolicy (or legacy str).
        """
        from .trajectory import TrajectoryFormatter

        # Convert Trajectory to episode dict for formatting
        episode = self._trajectory_to_episode(trajectory)

        if self.paradigm == "central_global":
            traj_str = TrajectoryFormatter.format_trajectory(episode)
            eval_prompt = self._create_global_prompt(traj_str)
        else:
            traj_str = TrajectoryFormatter.format_for_credit_assignment(episode)
            eval_prompt = self._create_credit_prompt(traj_str)

        eval_response = self._call_llm(eval_prompt)

        result = {"raw_response": eval_response, "paradigm": self.paradigm}

        if self.paradigm == "central_credit":
            agent_names = list(policies.keys())
            result["per_agent"] = PolicyGradientOptimizer.parse_credit_response(
                eval_response, agent_names
            )
        else:
            result["per_agent"] = {agent: eval_response for agent in policies}

        return result

    def _trajectory_to_episode(self, trajectory: Trajectory) -> dict:
        """Convert a Trajectory dataclass to legacy episode dict for formatting."""
        episode = {
            "task": trajectory.task,
            "task_type": trajectory.metadata.get("task_type", self.task_type),
            "transitions": trajectory.steps,
            "final_answer": trajectory.steps[-1]["output"] if trajectory.steps else "",
            "ground_truth": trajectory.task.get("ground_truth", ""),
            "reward": trajectory.reward,
            "episode_id": trajectory.metadata.get("episode_id", 0),
        }
        if "verified_reward" in trajectory.metadata:
            episode["verified_reward"] = trajectory.metadata["verified_reward"]
        if "verification_details" in trajectory.metadata:
            episode["verification_details"] = trajectory.metadata["verification_details"]
        if "evaluation_feedback" in trajectory.metadata:
            episode["evaluation_feedback"] = trajectory.metadata["evaluation_feedback"]
        return episode

    def _get_agent_role(self, agent_name: str) -> tuple[str, str]:
        try:
            idx = int(agent_name.split("_")[1]) - 1
        except (IndexError, ValueError):
            idx = 0

        n = self.num_agents
        position = idx + 1
        is_first = (idx == 0)
        is_last = (idx == n - 1)

        if n == 1:
            short = "Sole Agent"
            detailed = (
                f"Agent {position} is the only agent. It receives the task directly and "
                "produces the final evaluated answer."
            )
        elif is_first:
            short = "First Responder"
            later = ", ".join([f"Agent {j + 1}" for j in range(1, n)])
            detailed = (
                f"Agent {position} receives the task first and provides an initial response. "
                f"This response is added to the shared message pool and seen by {later}. "
                f"Agent {n}'s output will be the final answer."
            )
        elif is_last:
            prev = ", ".join([f"Agent {j + 1}" for j in range(idx)])
            short = "Final Responder"
            detailed = (
                f"Agent {position} sees the original task and all previous responses from "
                f"{prev} in the shared message pool. "
                f"Agent {position} produces the FINAL answer that will be evaluated."
            )
        else:
            prev = ", ".join([f"Agent {j + 1}" for j in range(idx)])
            later = ", ".join([f"Agent {j + 1}" for j in range(position, n)])
            short = f"Intermediate Agent {position}"
            detailed = (
                f"Agent {position} sees the original task and all previous responses from "
                f"{prev} in the shared message pool. "
                f"Its response is in turn seen by {later}. "
                f"Agent {n}'s output will be the final answer."
            )
        return short, detailed

    def _get_agent_criteria(self, agent_name: str) -> str:
        try:
            idx = int(agent_name.split("_")[1]) - 1
        except (IndexError, ValueError):
            idx = 0

        n = self.num_agents
        position = idx + 1
        is_first = (idx == 0)
        is_last = (idx == n - 1)
        short, _ = self._get_agent_role(agent_name)

        lines = [f"**For Agent {position} ({short}):**"]
        if is_first:
            lines += [
                "- Did it correctly understand the task?",
                "- Did it provide useful initial analysis or reasoning?",
                "- Did it set up subsequent agents for success?",
                "- Was the level of detail appropriate?",
                "- Did it identify key aspects of the problem?",
            ]
        elif is_last:
            lines += [
                "- Did it effectively use previous agents' responses?",
                "- Did it produce an accurate/high-quality final answer?",
                "- Did it appropriately refine or extend earlier agents' work?",
                "- Did it catch and correct any errors from previous agents?",
                "- Did it add value beyond earlier responses?",
            ]
        else:
            lines += [
                "- Did it correctly process the task and previous agents' responses?",
                "- Did it add meaningful value to the collaborative chain?",
                "- Did it effectively bridge earlier and later agents?",
                "- Did it correct any errors from earlier agents?",
                "- Was its response at an appropriate level of detail for subsequent agents?",
            ]
        return "\n".join(lines)

    def _create_global_prompt(self, trajectory: str) -> str:
        agent_lines = []
        for agent_name in self.all_agents:
            short, detailed = self._get_agent_role(agent_name)
            agent_lines.append(f"- {agent_name} ({short}): {detailed}")

        role_context = (
            f"This is a {self.num_agents}-agent sequential collaborative system:\n"
            + "\n".join(agent_lines)
        )

        prompt = f"{self.game_context}\n\n{self.evaluation_template}".format(
            role_context=role_context,
            task_type=self.task_type,
            trajectory=trajectory,
            num_agents=self.num_agents,
        )
        return self._tool_library_block() + prompt

    def _create_credit_prompt(self, trajectory: str) -> str:
        agent_keys_lines = ",\n".join(
            f'  "{name}": "(contribution assessment, what worked well, what could improve, specific policy suggestions)"'
            for name in self.all_agents
        )
        agent_evaluation_sections = (
            "Your response MUST be a JSON dictionary with exactly these keys:\n"
            "```json\n{\n"
            + agent_keys_lines
            + "\n}\n```"
        )

        criteria_blocks = [self._get_agent_criteria(name) for name in self.all_agents]
        agent_specific_criteria = "\n\n".join(criteria_blocks)

        prompt = f"{self.game_context}\n\n{self.evaluation_template}".format(
            task_type=self.task_type,
            trajectory=trajectory,
            num_agents=self.num_agents,
            agent_evaluation_sections=agent_evaluation_sections,
            agent_specific_criteria=agent_specific_criteria,
        )
        return self._tool_library_block() + prompt

    def _call_llm(self, prompt: str, max_tokens: int = 1500) -> str:
        params = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        m = self._model.lower()
        if "o1" in m or "o3" in m:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content.strip()
