from __future__ import annotations

from harbor.agents.installed.mini_swe_agent import MiniSweAgent


class LlamaTextBasedMiniSweAgent(MiniSweAgent):
    """Mini-SWE-Agent configured for text-parsed shell commands.

    The model config uses Mini-SWE's ``litellm_textbased`` model class, so the
    model never has to satisfy provider-native function-calling syntax. This is
    intentional for Llama-family routes that are otherwise prone to structured
    tool-call parser failures.
    """

    pass
