# Hosting the search server on a CPU-only box

This serves `search_anglican_library` as an MCP tool from a cheap rented Linux
host. Embedding was done on the GPU (H100); serving is CPU-only — query
embedding (Qwen3-0.6B @ 512-d), FAISS search, and a light cross-encoder rerank.

## Footprint (measured)

| Item | Size |
|---|---|
| FAISS index (512-d, 1.23M vectors) — held in RAM | **2.5 GB** |
| Serving DB (`library-serve.db`, content dropped) | **3.7 GB** disk |
| Embedder + light reranker + torch runtime | ~2 GB RAM |
| **RAM working set** | **~6 GB** |

➡ **Target spec: 4 vCPU · 8 GB RAM · 40 GB SSD** (comfortable: 8 vCPU · 16 GB).
Dedicated (non-shared) vCPUs matter — the reranker is the latency-sensitive part.

## Where to host (best value, 2026)

| Provider | Plan (≈) | vCPU / RAM | ~Price/mo | Notes |
|---|---|---|---|---|
| **Oracle Cloud Free** | Ampere A1 (Always Free) | 2 ARM / 12 GB | **$0** | Fits our 6 GB footprint. ARM (wheels work). Free tier dropped to 2/12 in Jun 2026; provisioning can be capacity-limited. Best value if you tolerate ARM. |
| **Netcup** ⭐ | VPS / RS G12 | 4–8 **dedicated** / 8–16 GB | **€8–13** | Best paid value: dedicated AMD EPYC "Turin" cores, no throttling → consistent rerank latency. x86_64 (everything just works). |
| **Contabo** | VPS M/L | 6–9 / 16–24 GB | **€8–12** | Cheapest RAM/core, but slower disk and oversubscribed CPU (variable latency). |
| **Hetzner** | CPX31 / CPX41 | 4–8 shared / 8–16 GB | **$25+** | Simplest, reliable, hourly billing, but pricier after the 2026 hikes and shared-CPU throttling. |

**Recommendation:** **Oracle's Always-Free Ampere A1** if you want $0 and don't
mind ARM + the provisioning lottery. Otherwise **Netcup** (x86, dedicated cores)
is the best paid value at ~€8–13/mo and the least fuss. Pick Ubuntu 24.04 LTS.

(Prices move — verify on the provider's page. Sources: Hetzner pricing,
Netcup vs Hetzner 2026 comparisons, Oracle free-tier breakdown.)

## Steps

### 0. Prerequisite — build the index (once, on the H100)
Per the [notebook](../notebooks/embed_on_pcai.ipynb) you produce `index.faiss`.
Keep `library.db` on your build machine (you need its `content` only if you ever
re-chunk).

### 1. Make the slim serving DB (on the build machine)
```bash
uv run python scripts/slim_db.py library.db library-serve.db   # ~3.7 GB
```

### 2. Provision the box
Create the instance (4 vCPU / 8 GB / Ubuntu 24.04), add your SSH public key,
then create a user (or use the default), e.g. `anglican@SERVER_IP`.

### 3. Install the server
```bash
ssh anglican@SERVER_IP
git clone https://github.com/Timon-Stacy/anglican-search-starter
cd anglican-search-starter
bash deploy/serve_setup.sh          # installs CPU torch + deps, caches models
```

### 4. Upload the data
From your build machine:
```bash
scp library-serve.db index.faiss anglican@SERVER_IP:~/anglican-search-starter/data/
```

### 5. Smoke-test on the box
```bash
cd ~/anglican-search-starter
ANGLICAN_DB=$PWD/data/library-serve.db ANGLICAN_INDEX=$PWD/data/index.faiss \
ANGLICAN_RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2 ANGLICAN_RERANK_POOL=30 \
uv run python -m anglican_search.search --q "the doctrine of the Holy Trinity" --k 3
```

### 6. Wire your MCP client (over SSH — no open ports)
The client launches the remote server and pipes its stdio over SSH. Add to your
client's MCP servers config (replace the host):
```json
{
  "mcpServers": {
    "anglican-library": {
      "command": "ssh",
      "args": [
        "-T", "anglican@SERVER_IP",
        "cd ~/anglican-search-starter && ANGLICAN_DB=$HOME/anglican-search-starter/data/library-serve.db ANGLICAN_INDEX=$HOME/anglican-search-starter/data/index.faiss ANGLICAN_RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2 ANGLICAN_RERANK_POOL=30 HF_HUB_OFFLINE=1 ~/.local/bin/uv run anglican-search-mcp"
      ]
    }
  }
}
```
The server warms the models in the main thread at startup (~20–40 s on CPU on
the first connect of a session), then queries are fast. Use SSH keys so the
launch is non-interactive.

## Always-on HTTPS service (recommended for a real deployment)

Runs the server persistently (models loaded **once**, so queries are always
fast) over Streamable HTTP, behind Caddy for automatic TLS + bearer-token auth.
The MCP server binds `127.0.0.1:8000` only; Caddy is the sole public listener.

**Prereqs:** a domain (or a free DDNS like DuckDNS) with an A record pointing at
the server IP, and ports 80 + 443 open.

> Firewall: open 80/443 in the provider's panel/security list. On **Oracle**,
> also open them in the instance: `sudo iptables -I INPUT -p tcp -m multiport --dports 80,443 -j ACCEPT` and persist, or use `ufw`. Keep 8000 closed (localhost-only).

### 1. Run the MCP server under systemd
```bash
sudo cp deploy/systemd/anglican-mcp.service /etc/systemd/system/
# edit User= / paths if your user isn't "anglican"
sudo systemctl daemon-reload
sudo systemctl enable --now anglican-mcp
journalctl -u anglican-mcp -f          # wait for: serving on http://127.0.0.1:8000/mcp (~30s warmup)
```
Verify locally:
```bash
curl -sN -XPOST localhost:8000/mcp -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"c","version":"0"}}}' | head -c 200
# -> ...{"result":{"serverInfo":{"name":"anglican-library"...
```

### 2. Auth token
```bash
openssl rand -hex 32           # copy this
```

### 3. Caddy (automatic HTTPS + token check)
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile     # set your domain + paste the token
sudo systemctl restart caddy       # fetches a Let's Encrypt cert automatically
```

### 4. Point your MCP client at the HTTPS endpoint
```json
{
  "mcpServers": {
    "anglican-library": {
      "type": "http",
      "url": "https://your.domain.example/mcp",
      "headers": { "Authorization": "Bearer YOUR_TOKEN" }
    }
  }
}
```
(Adapt to your client's remote-server schema — some use `"transport": "http"`.)

### Managing it
```bash
sudo systemctl restart anglican-mcp     # after swapping in a new index/DB
journalctl -u anglican-mcp -f           # logs
```

## Cost summary
- **$0/mo** — Oracle Always-Free Ampere A1 (2 ARM / 12 GB).
- **~€8–13/mo** — Netcup x86 dedicated (recommended paid).
- One-time GPU embedding cost is separate (the H100 run).
