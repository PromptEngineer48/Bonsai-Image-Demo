#!/usr/bin/env python3
"""
Bonsai Image MCP Server

4 tools that wrap the local GPU backend running at localhost:8000:
  - health_check     → GET  /healthz
  - list_backends    → GET  /backends
  - generate_image   → POST /generate
  - compare_backends → POST /generate/compare

Install deps (run once, inside the project venv):
    pip install "mcp[cli]" httpx

Run as HTTP server (default — for Claude Desktop Connectors):
    python mcp_server.py
    → Then add  http://localhost:8001/mcp  in Claude Desktop Connectors

Run as stdio (for Claude Code / .mcp.json):
    python mcp_server.py --stdio

Claude Desktop — add via Connectors UI  (Settings → Connectors → +):
    http://localhost:8001/mcp

Claude Code config  (.mcp.json in project root):

    {
      "mcpServers": {
        "bonsai-image": {
          "command": "wsl",
          "args": ["-e", "/home/palas/Bonsai-Image-Demo/.venv/bin/python",
                   "/home/palas/Bonsai-Image-Demo/mcp_server.py", "--stdio"],
          "type": "stdio"
        }
      }
    }
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import httpx
from mcp import types
from mcp.server.fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

mcp = FastMCP("bonsai-image", host="0.0.0.0", port=8001)


# ── Tool 1: health_check ──────────────────────────────────────────────────
@mcp.tool()
def health_check() -> str:
    """Check whether the Bonsai GPU backend is running and ready.

    Calls GET /healthz. Returns a plain-text status message. Use this
    before generating images to confirm the server is up.
    """
    try:
        resp = httpx.get(f"{BACKEND_URL}/healthz", timeout=5)
        resp.raise_for_status()
        status = resp.json().get("status", "unknown")
        return f"Backend is healthy. Status: {status}"
    except httpx.ConnectError:
        return (
            "ERROR: Cannot connect to localhost:8000. "
            "Make sure you ran ./scripts/serve.sh first."
        )
    except Exception as exc:
        return f"ERROR: {exc}"


# ── Tool 2: list_backends ─────────────────────────────────────────────────
@mcp.tool()
def list_backends() -> str:
    """List the available model families and inference backends on this machine.

    Calls GET /backends. Returns the loaded backend kind (gemlite / mlx),
    the supported model families (bonsai-ternary, bonsai-binary), and
    which family is currently loaded.
    """
    resp = httpx.get(f"{BACKEND_URL}/backends", timeout=5)
    resp.raise_for_status()
    d = resp.json()
    lines = [
        f"Inference kind  : {d['kind']}",
        f"Loaded families : {', '.join(d['supported_families'])}",
        f"Default family  : {d['default_family']}",
        f"Healthy         : {d['healthy']}",
    ]
    if d.get("reason"):
        lines.append(f"Reason          : {d['reason']}")
    return "\n".join(lines)


# ── Tool 3: generate_image ────────────────────────────────────────────────
@mcp.tool()
def generate_image(
    prompt: str,
    steps: int = 4,
    width: int = 512,
    height: int = 512,
    guidance: float = 3.5,
    seed: int | None = None,
    backend: str = "bonsai-ternary-gemlite",
) -> list[types.TextContent | types.ImageContent]:
    """Generate a single image from a text prompt on the local GPU.

    Args:
        prompt:   Text description of the image to generate.
        steps:    Diffusion steps (4 = fast draft; 20-30 = high quality).
        width:    Output width in pixels  (default 512, min 16).
        height:   Output height in pixels (default 512, min 16).
        guidance: CFG guidance scale — higher = more literal (default 3.5).
        seed:     Integer seed for reproducibility. Omit for a random result.
        backend:  Backend name, e.g. "bonsai-ternary-gemlite" (default).

    Returns the generated image inline plus a text summary with timing and
    the path it was saved to under outputs/.
    """
    payload: dict = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "guidance": guidance,
        "backend": backend,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = httpx.post(f"{BACKEND_URL}/generate", json=payload, timeout=300)
    resp.raise_for_status()

    # Persist the PNG so the user has it after the conversation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUTS_DIR / f"bonsai_{timestamp}.png"
    out_path.write_bytes(resp.content)

    wall_s = resp.headers.get("x-wall-seconds", "?")
    peak_mb = resp.headers.get("x-peak-memory-mb", "?")
    b64 = base64.b64encode(resp.content).decode("ascii")

    return [
        types.TextContent(
            type="text",
            text=(
                f"Generated in {wall_s}s | Peak GPU memory: {peak_mb} MB\n"
                f"Saved to: {out_path}"
            ),
        ),
        types.ImageContent(type="image", data=b64, mimeType="image/png"),
    ]


# ── Tool 4: compare_backends ──────────────────────────────────────────────
@mcp.tool()
def compare_backends(
    prompt: str,
    steps: int = 4,
    width: int = 512,
    height: int = 512,
    guidance: float = 3.5,
    seed: int = 42,
    backends: list[str] | None = None,
) -> list[types.TextContent | types.ImageContent]:
    """Run the same prompt on multiple backends and return all images side-by-side.

    Useful for comparing bonsai-ternary vs bonsai-binary quality at the
    same seed, or benchmarking generation speed across configs.

    Args:
        prompt:   Text prompt shared across all backends.
        steps:    Diffusion steps (default 4).
        width:    Image width in pixels (default 512).
        height:   Image height in pixels (default 512).
        guidance: CFG guidance scale (default 3.5).
        seed:     Fixed seed so all backends get an identical noise start (default 42).
        backends: List of backend names to compare. Defaults to all available.

    Returns one text block + one image per backend, in order.
    """
    payload: dict = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "guidance": guidance,
        "seed": seed,
    }
    if backends:
        payload["backends"] = backends

    resp = httpx.post(f"{BACKEND_URL}/generate/compare", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    content: list[types.TextContent | types.ImageContent] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for result in data["results"]:
        backend_name = result["backend"]
        wall_s = result["wall_seconds"]
        swap_s = result["swap_seconds"]
        png_bytes = base64.b64decode(result["png_b64"])

        out_path = OUTPUTS_DIR / f"compare_{timestamp}_{backend_name}.png"
        out_path.write_bytes(png_bytes)

        content.append(types.TextContent(
            type="text",
            text=(
                f"[{backend_name}]  gen: {wall_s:.2f}s | "
                f"swap: {swap_s:.2f}s | saved: {out_path.name}"
            ),
        ))
        content.append(types.ImageContent(
            type="image",
            data=result["png_b64"],
            mimeType="image/png",
        ))

    return content


if __name__ == "__main__":
    import sys
    if "--stdio" in sys.argv:
        # Stdio mode — used by Claude Code via .mcp.json
        mcp.run(transport="stdio")
    else:
        # HTTP mode (default) — start on port 8001, exposes /mcp endpoint.
        # Add  http://localhost:8001/mcp  in Claude Desktop Connectors.
        mcp.run(transport="streamable-http")
