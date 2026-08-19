import os

from custom_agents.llama_textbased_mini_swe import LlamaTextBasedMiniSweAgent


class BootstrapCompatibleLlamaTextBasedMiniSweAgent(LlamaTextBasedMiniSweAgent):
    """Mini-SWE 2.4.5 with a reproducible client-only LiteLLM installation.

    Solver/run behavior remains inherited from the benchmark's existing
    LlamaTextBasedMiniSweAgent. Only installation is overridden.

    Harbor's stock Mini-SWE installer adds ``litellm[proxy]``. The benchmark
    uses LiteLLM as a client to an external gateway and does not run a local
    LiteLLM proxy, so the proxy-server extra is unnecessary here. We pin the
    LiteLLM core version for reproducibility.
    """

    async def install(self, environment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get &>/dev/null; then"
                "  apt-get update && apt-get install -y curl build-essential git;"
                " elif command -v apk &>/dev/null; then"
                "  apk add --no-cache curl bash build-base git python3 py3-pip;"
                " elif command -v yum &>/dev/null; then"
                "  yum install -y curl git gcc make;"
                " elif command -v dnf &>/dev/null; then"
                "  dnf install -y curl git gcc make;"
                " else"
                '  echo "Warning: No known package manager found, assuming build tools are available" >&2;'
                " fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        version = getattr(self, "_version", None)
        version_spec = f"=={version}" if version else ""
        litellm_version = os.environ.get("MINI_SWE_LITELLM_VERSION", "1.83.0")

        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "if ! command -v uv >/dev/null 2>&1; then"
                "  curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh;"
                " fi && "
                'if ! grep -q \'export PATH="$HOME/.local/bin:$PATH"\' "$HOME/.bashrc" 2>/dev/null; then'
                '  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> "$HOME/.bashrc";'
                " fi && "
                'if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv tool install mini-swe-agent{version_spec} "
                f"--with 'litellm=={litellm_version}' && "
                "mini-swe-agent --help"
            ),
        )
