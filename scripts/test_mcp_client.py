"""Smoke-test the MCP server over real stdio: initialize, list tools, call tool."""

from __future__ import annotations

import asyncio
import os
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    env = dict(os.environ)
    env["ANGLICAN_DB"] = "test_library.db"
    env["ANGLICAN_INDEX"] = "test_index.faiss"
    env["PYTHONUTF8"] = "1"

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "anglican_search.server"], env=env
    )
    errlog = open("mcp_server.err", "w", encoding="utf-8")
    t0 = time.time()
    print("spawning server...", flush=True)
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            print(f"[{time.time()-t0:.1f}s] initializing...", flush=True)
            await asyncio.wait_for(session.initialize(), timeout=60)
            print(f"[{time.time()-t0:.1f}s] initialized OK", flush=True)

            tools = await asyncio.wait_for(session.list_tools(), timeout=15)
            print("TOOLS:", [(t.name, t.description.splitlines()[0]) for t in tools.tools])

            print(f"[{time.time()-t0:.1f}s] calling tool (loads model)...", flush=True)
            res = await asyncio.wait_for(
                session.call_tool(
                    "search_anglican_library",
                    {"query": "the divinity and eternal generation of the Son", "top_k": 2},
                ),
                timeout=120,
            )
            print(f"[{time.time()-t0:.1f}s] --- semantic tool result ---")
            print(res.content[0].text[:1200])

            res2 = await asyncio.wait_for(
                session.call_tool(
                    "search_anglican_library",
                    {"query": "Athanasian Creed", "top_k": 2, "mode": "literal"},
                ),
                timeout=60,
            )
            print("\n--- literal tool result (first 350 chars) ---")
            print(res2.content[0].text[:350])
    errlog.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.TimeoutError:
        print("\nTIMEOUT — see mcp_server.err for the server's stderr.", flush=True)
        sys.exit(2)
