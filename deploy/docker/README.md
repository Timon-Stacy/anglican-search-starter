# Docker deployment: Intel Arc home box + cloud VPS edge (WireGuard)

The heavy work (Arc A380 GPU, FAISS index, the SQLite DBs, the whole app) runs on
your **home server**. A cheap **cloud VPS** is a thin TLS edge that reverse-proxies
HTTPS to the home app over a **WireGuard** tunnel. The home box dials out, so it
needs no public IP and no port-forwarding.

```
Internet ──443──▶  VPS (public IP, domain)            Home server (Intel Arc A380)
                   caddy: TLS + reverse_proxy         bilson: engine + /mcp + /v1 + website
                   wireguard: 51820/udp  ◀── tunnel ──  wireguard (dials out)
                          │                            listens on 10.8.0.2:8001
                          └─ proxies 443 ─▶ 10.8.0.2:8001 over wg0
```

Layout:

| Path | Runs on | What it is |
|---|---|---|
| `home/Dockerfile.serve` | home | Arc (XPU) image: app + engine, torch from Intel's IPEX base |
| `home/docker-compose.yml` | home | `bilson` (Arc GPU) + `wireguard` (dials the VPS) |
| `vps/docker-compose.yml` | VPS | `caddy` (TLS) + `wireguard` (tunnel endpoint) |
| `vps/Caddyfile` | VPS | HTTPS + `reverse_proxy 10.8.0.2:8001` |

---

## Prerequisites

**Home server (must be Linux for Arc-in-Docker):**
- Ubuntu 24.04 (or similar) on the box with the Arc A380. **Windows/WSL2 will not
  pass the Arc compute device into containers** — use bare-metal Linux.
- Recent kernel with the Intel `i915` driver + Intel **compute runtime** installed
  on the host (so `/dev/dri/renderD128` exists and works). Follow Intel's
  "Installing client GPUs" guide for 24.04. Verify on the host: `clinfo` lists the
  Arc, and `ls -l /dev/dri` shows `renderD128`.
- Docker Engine + the compose plugin.

**VPS:** any small Linux box with a public IP (1 vCPU / 1 GB is plenty), Docker +
compose, and these ports open in the provider firewall: **80/tcp, 443/tcp,
443/udp, 51820/udp**.

**DNS:** an `A` record for your domain pointing at the **VPS** public IP.

---

**Order:** Steps 1 and 4 (stage the DB, build the image, verify the Arc GPU, build
the index, optional local smoke test) don't need the tunnel — do them first to
prove the GPU and app work. Steps 2, 3, 5, 6 then wire up public access through
the VPS. Home's `wg0.conf` needs the VPS's public key, which is why the VPS comes
up (Step 3) before the home stack (Step 5).

## Step 1 — Put the repaired DB in place (home)

You already produced `library-clean.db` (the repaired copy). Stage it as the
serving DB:

```bash
# from the repo root on the home box
cp library-clean.db deploy/docker/home/data/library.db
```

(`index.faiss` doesn't exist yet — Step 4 builds it.)

## Step 2 — WireGuard keys + configs

On **each** box generate a keypair:

```bash
wg genkey | tee privatekey | wg pubkey > publickey   # do this on home AND on the VPS
```

Then fill the two configs (swap in the keys + the VPS public IP):

- `vps/wireguard/wg0.conf`  — from `wg0.conf.example`; needs the VPS private key
  and the **home** public key.
- `home/wireguard/wg0.conf` — from `wg0.conf.example`; needs the home private key,
  the **VPS** public key, and `Endpoint = <VPS_PUBLIC_IP>:51820`.

The tunnel subnet is `10.8.0.0/24` (VPS = `.1`, home = `.2`). If you change it,
update `Caddyfile` (`reverse_proxy 10.8.0.2:8001`) to match.

## Step 3 — Bring up the VPS edge

```bash
cd deploy/docker/vps
cp .env.example .env && nano .env        # set DOMAIN + ACME_EMAIL
docker compose up -d
docker compose logs -f wireguard          # confirm the interface comes up
```

Caddy won't get a cert until the home app is reachable, which is fine for now.

## Step 4 — Build the image, verify the GPU, build the index (home)

These run on the default network via the `engine` one-off service — no tunnel
needed yet, so you can do all of this before touching WireGuard.

```bash
cd deploy/docker/home
cp .env.example .env && nano .env         # set BILSON_SECRET, BILSON_ADMIN_EMAIL, BILSON_PUBLIC_URL, IPEX_TAG

# Build the Arc image (tagged anglican-bilson-xpu:latest, reused by `engine`).
docker compose build bilson

# PROVE the Arc GPU works inside the container before the long embed run:
docker compose run --rm engine python scripts/check_env.py
#   -> must print: backend : XPU (Intel Arc ...) and "GPU matmul OK"
#   If it prints CPU instead, fix the host Intel driver / /dev/dri passthrough first.

# Build the index once (writes /data/index.faiss). The A380 has only ~6 GB, and
# XPU uses eager attention (it materializes the seq×seq score matrix), so keep the
# batch small: 16 is a safe default, 8 if it still OOMs, 24 if you want it faster.
docker compose run --rm engine \
  python -m anglican_search.embed_library --phase all --encode-batch 16
```

The embed is resumable — if it stops (or you lower the batch and re-run), it
continues from the last saved window. ~1.23M chunks; watch the
`embedded N/1230281 ... ETA` log. On `XPU out of memory`, halve `--encode-batch`.
If fragmentation bites near the limit, add `-e PYTORCH_XPU_ALLOC_CONF=expandable_segments:True`.

### Optional — smoke-test the app locally before wiring the tunnel

```bash
docker compose run --rm -p 8001:8001 engine python -m bilson_ai.app &
sleep 60 && curl -s http://127.0.0.1:8001/health    # {"status":"ok","search_engine":"ready"}
# Ctrl-C / docker stop the run when done.
```

## Step 5 — Start the home stack (with the tunnel)

This needs `home/wireguard/wg0.conf` filled in (Step 2) and the VPS up (Step 3).

```bash
cd deploy/docker/home
docker compose up -d                      # starts wireguard + bilson
docker compose logs -f bilson             # wait for: "[bilson] ready: MCP (/mcp) + REST (/v1) + website"
```

The `wireguard` container dials the VPS; the `bilson` container shares its netns,
so it's now reachable from the VPS at `10.8.0.2:8001`.

## Step 6 — Verify end to end

```bash
# On the home box (loopback publish): the app answers.
curl -s http://127.0.0.1:8001/health        # {"status":"ok","search_engine":"ready"}

# On the VPS: it can reach the home app through the tunnel.
docker compose -f vps/docker-compose.yml exec caddy wget -qO- http://10.8.0.2:8001/health

# From anywhere: HTTPS via the VPS.
curl -s https://YOUR_DOMAIN/health
```

Then create your admin user and an API key:

```bash
# on the home box
cd deploy/docker/home
docker compose exec bilson python -m bilson_ai.admin create-admin you@example.com 'a-strong-password'
```

Log in at `https://YOUR_DOMAIN/login`, open the dashboard, create an API key, and
point an MCP client at `https://YOUR_DOMAIN/mcp` with `Authorization: Bearer <key>`.

---

## Operating it

```bash
# Update the app after a code change (home):
cd deploy/docker/home && docker compose build bilson && docker compose up -d bilson

# Swap in a rebuilt index: replace data/index.faiss, then:
docker compose restart bilson

# Logs / status
docker compose logs -f bilson
docker compose ps
```

## Notes & trade-offs

- **Why share the wireguard netns?** It keeps everything in Docker: the app is
  reachable at the WG IP with no host-level routing. If you'd rather run WireGuard
  on the host with `wg-quick` instead, drop the `wireguard` service and the
  `network_mode: "service:wireguard"` line, publish `8001` from `bilson`, and the
  VPS reaches `10.8.0.2:8001` via the host interface — same result, less Docker.
- **The home app is never public.** It only listens inside the tunnel (plus a
  loopback publish on the home box for debugging). All public traffic goes through
  Caddy's TLS on the VPS.
- **Latency:** every request takes one home↔VPS tunnel hop. Pick a VPS region near
  home; for search (not chatty) it's negligible next to the rerank compute.
- **`slim_db.py` is not used here** — that's for a remote CPU-only serve box. The
  home box has the GPU and the full DB, so keep the full (repaired) `library.db`.
- **Nvidia path is unchanged:** the root `Dockerfile` (cu128) + `deploy/k8s/` still
  build the index on Nvidia. This Arc setup is additive.
