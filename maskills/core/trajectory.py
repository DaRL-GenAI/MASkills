"""Trajectory formatting for LLM evaluation."""

from typing import Dict


class TrajectoryFormatter:
    """Format episode trajectories for LLM evaluation."""

    @staticmethod
    def format_trajectory(episode: Dict) -> str:
        task = episode.get('task', {})
        task_type = episode.get('task_type', 'unknown')
        transitions = episode.get('transitions', [])
        final_answer = episode.get('final_answer', '')
        ground_truth = episode.get('ground_truth', '')
        reward = episode.get('reward', 0.0)
        verified_reward = episode.get('verified_reward', None)
        evaluation_feedback = episode.get('evaluation_feedback', '')
        verification_details = episode.get('verification_details', '')

        lines = []
        lines.append("=== EPISODE TRAJECTORY ===")
        lines.append(f"Task Type: {task_type}")
        lines.append(f"Task ID: {task.get('task_id', 'unknown')}")
        lines.append("")

        lines.append("[TASK]")
        question = task.get('question', '')
        context = task.get('context', '')
        if context:
            lines.append(f"Context: {context[:500]}..." if len(context) > 500 else f"Context: {context}")
            lines.append("")
        lines.append(f"Question: {question}")
        lines.append("")

        num_transitions = len(transitions)
        for t_idx, transition in enumerate(transitions):
            agent_name = transition.get('agent', transition.get('agent_id', 'unknown'))
            system_prompt = transition.get('system_prompt', '')
            agent_output = transition.get('output', transition.get('action', ''))

            label = agent_name.replace("_", " ").upper()
            if t_idx == 0:
                label += " - First Response"
            elif t_idx == num_transitions - 1:
                label += " - Final Response"
            lines.append(f"[{label}]")

            lines.append(f"Policy/System Prompt: {system_prompt[:200]}..." if len(system_prompt) > 200 else f"Policy/System Prompt: {system_prompt}")
            lines.append("")
            lines.append("Response:")
            lines.append(agent_output)
            lines.append("")

        lines.append("[EVALUATION]")
        lines.append(f"Final Answer: {final_answer[:500]}..." if len(final_answer) > 500 else f"Final Answer: {final_answer}")
        lines.append("")

        if ground_truth:
            lines.append(f"Ground Truth: {ground_truth[:200]}..." if len(ground_truth) > 200 else f"Ground Truth: {ground_truth}")
            lines.append("")

        lines.append(f"Score: {reward:.2f} (heuristic)")
        if verified_reward is not None:
            lines.append(f"Verified Score: {verified_reward:.2f} (LLM judge / test execution)")
        if evaluation_feedback:
            lines.append(f"Feedback: {evaluation_feedback}")
        if verification_details:
            lines.append(f"Verification Details: {verification_details}")
        lines.append("")

        score_display = (
            f"verified={verified_reward:.2f}" if verified_reward is not None
            else f"heuristic={reward:.2f}"
        )
        lines.append(f"=== END TRAJECTORY (Score: {score_display}) ===")

        return "\n".join(lines)

    @staticmethod
    def format_trajectory_minimal(episode: Dict) -> str:
        task = episode.get('task', {})
        transitions = episode.get('transitions', [])
        reward = episode.get('reward', 0.0)

        lines = []
        lines.append(f"Task: {task.get('question', 'N/A')[:100]}...")

        for transition in transitions:
            agent = transition.get('agent', transition.get('agent_id', 'unknown'))
            output = transition.get('output', transition.get('action', ''))[:200]
            lines.append(f"{agent}: {output}...")

        lines.append(f"Score: {reward:.2f}")
        return "\n".join(lines)

    @staticmethod
    def format_for_credit_assignment(episode: Dict) -> str:
        task = episode.get('task', {})
        task_type = episode.get('task_type', 'unknown')
        transitions = episode.get('transitions', [])
        final_answer = episode.get('final_answer', '')
        ground_truth = episode.get('ground_truth', '')
        reward = episode.get('reward', 0.0)
        verified_reward = episode.get('verified_reward', None)
        verification_details = episode.get('verification_details', '')

        lines = []
        lines.append("=== CREDIT ASSIGNMENT TRAJECTORY ===")
        lines.append(f"Task Type: {task_type}")
        lines.append(f"Overall Score: {reward:.2f} (heuristic)")
        if verified_reward is not None:
            lines.append(f"Verified Score: {verified_reward:.2f} (LLM judge / test execution)")
        lines.append("")

        lines.append("[TASK]")
        # Include the supporting context (HotpotQA gold passages) when
        # present — critic and downstream optimizer use it as the basis for
        # judging whether the agent's reasoning matched the available
        # evidence.  Math / coding tasks have empty ``context`` so this is
        # a no-op for them.
        context = task.get('context', '')
        if context:
            lines.append("Context:")
            lines.append(context)
            lines.append("")
        lines.append(f"Question: {task.get('question', 'N/A')}")
        lines.append("")

        if ground_truth:
            lines.append("[GROUND TRUTH]")
            lines.append(ground_truth)
            lines.append("")

        num_transitions = len(transitions)
        for i, transition in enumerate(transitions):
            agent_name = transition.get('agent', transition.get('agent_id', f'agent_{i+1}'))
            system_prompt = transition.get('system_prompt', '')
            output = transition.get('output', transition.get('action', ''))
            position = i + 1

            lines.append(f"=== {agent_name.upper()} CONTRIBUTION ===")
            lines.append(f"Policy: {system_prompt}")
            lines.append("")
            lines.append("Output:")
            lines.append(output)
            lines.append("")

            if i == 0:
                lines.append(f"Role Context: Agent {position} provides the initial response to the task.")
                if num_transitions > 1:
                    lines.append("Subsequent agents will see this response in the shared message pool.")
            elif i == num_transitions - 1:
                lines.append(f"Role Context: Agent {position} provides the FINAL answer that is evaluated.")
                lines.append("It sees all previous agents' responses in the shared message pool.")
            else:
                lines.append(f"Role Context: Agent {position} is an intermediate agent.")
                lines.append("It sees all previous responses and contributes to the shared message pool.")
            lines.append("")

        lines.append("=== FINAL EVALUATION ===")
        lines.append(f"Final Answer: {final_answer}")
        lines.append(f"Score: {reward:.2f} (heuristic)")
        if verified_reward is not None:
            lines.append(f"Verified Score: {verified_reward:.2f} (LLM judge / test execution)")
        if verification_details:
            lines.append(f"Verification Details: {verification_details}")
        lines.append("")
        lines.append("Evaluate each agent's contribution to the final score:")
        for i in range(num_transitions):
            position = i + 1
            if i == 0:
                lines.append(f"- Did Agent {position} provide a useful initial response?")
            elif i == num_transitions - 1:
                lines.append(f"- Did Agent {position} effectively produce a high-quality final answer?")
            else:
                lines.append(f"- Did Agent {position} make a meaningful intermediate contribution?")
        lines.append("- How did each agent contribute to the final quality?")
        lines.append("")

        return "\n".join(lines)
