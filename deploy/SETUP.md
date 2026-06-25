# Set up Bilson AI on the Dell 7920 + RTX 3060 (full runbook)

End state: a public HTTPS site where people sign up, browse the library, run
manual semantic/literal searches, get API keys, and call `POST /v1/search`; an
admin panel for you. Everything runs in one GPU process on your workstation.

**Assumptions:** Ubuntu 24.04 on the Dell 7920, the RTX 3060 12 GB installed, and
a domain you control. The 3060 easily holds the embedder (~1.2 GB) + the full
`bge-reranker-v2-m3` reranker (~1.1 GB) with room to spare.

---

## 1. NVIDIA driver + uv
```bash
sudo ubuntu-drivers autoinstall      # or: sudo apt install -y nvidia-driver-560
sudo reboot
nvidia-smi                           # confirm the 3060 shows up
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Get the code + GPU environment (one command)
```bash
git clone https://github.com/Timon-Stacy/anglican-search-starter
cd anglican-search-starter
uv sync --extra web                  # CUDA 12.8 PyTorch + search + web deps + the package
uv run python scripts/check_env.py   # should print the RTX 3060 and sm_86, cuda available
```

## 3. Get the index + data into `data/`
You need `library.db` (the SQLite store with cleaned chunks) and `index.faiss`.
Two ways:

- **Build on the 3060** (simplest, run it overnight — ~hours for 1.23M chunks):
  ```bash
  mkdir -p data && cp /path/to/library.db data/
  ANGLICAN_DB=data/library.db ANGLICAN_INDEX=data/index.faiss \
    uv run python -m anglican_search.embed_library --phase all --encode-batch 128
  ```
- **Or build on the H100** (faster) via [../notebooks/embed_on_pcai.ipynb](../notebooks/embed_on_pcai.ipynb),
  then copy both files into `data/`:
  ```bash
  scp library.db index.faiss <workstation>:~/anglican-search-starter/data/
  ```

## 4. Run the service
```bash
openssl rand -hex 32                  # copy -> BILSON_SECRET
sudo cp deploy/systemd/bilson-ai.service /etc/systemd/system/
sudo nano /etc/systemd/system/bilson-ai.service
#   set User= to your username, fix the /home/<user>/... paths,
#   set BILSON_SECRET=<value>, BILSON_ADMIN_EMAIL=<your email>
sudo systemctl daemon-reload
sudo systemctl enable --now bilson-ai
journalctl -u bilson-ai -f           # wait for "search engine ready."
curl -s localhost:8001/health        # {"status":"ok","search_engine":"ready"}
```

## 5. Put it on the internet (pick one)

### A. Cloudflare Tunnel — recommended for a home/office box
No port forwarding, no public IP, hides your address, free, TLS at the edge.
Requires your domain on a (free) Cloudflare account.
```bash
# install cloudflared (see https://pkg.cloudflare.com), then:
cloudflared tunnel login
cloudflared tunnel create bilson
cloudflared tunnel route dns bilson your.domain.example
# ~/.cloudflared/config.yml:
#   tunnel: <tunnel-id>
#   credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
#   ingress:
#     - hostname: your.domain.example
#       service: http://localhost:8001
#     - service: http_status:404
sudo cloudflared service install     # runs the tunnel on boot
```

### B. Caddy + router port-forward — if you have a public IP
Point your domain's A record at your IP, forward router ports 80/443 to the
workstation, then:
```bash
sudo apt install -y caddy            # (official repo cmds: caddyserver.com/docs/install)
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile       # set your domain
sudo systemctl restart caddy         # auto Let's Encrypt cert
```
(Many home ISPs use CGNAT, which blocks inbound — if so, use Cloudflare Tunnel.)

## 6. Become admin + verify
1. Visit `https://your.domain.example` → **Sign up** with the email you set as
   `BILSON_ADMIN_EMAIL` → you get the **Admin** link.
2. As a user: **Search** (manual semantic/literal), **Library** (browse all books,
   read passages), **Dashboard** (create an API key).
3. Test the API:
```bash
curl -s https://your.domain.example/v1/search \
  -H "Authorization: Bearer bk_your_key" -H "Content-Type: application/json" \
  -d '{"query":"the eternal generation of the Son","top_k":3}'
```

---

## What people can do on the site
- **Browse** the whole library (`/library`): filter by title/author/category, open
  any book and page through its cleaned passages.
- **Search** manually (`/search`): semantic (meaning) or literal (keywords).
- **API** (`/dashboard` → key → `/docs`): programmatic `POST /v1/search`, metered
  by a per-user monthly quota you set in the admin panel.

## Day-2 operations
```bash
git pull && uv sync --extra web && sudo systemctl restart bilson-ai   # update
journalctl -u bilson-ai -f                                            # logs
cp data/accounts.db ~/backups/                                        # back up accounts
```
- Re-embed after a model/data change: `uv run python -m anglican_search.embed_library --phase all`, then restart the service.
- Want MCP clients too? Run the `anglican-mcp` service (see [SERVE.md](SERVE.md)) and uncomment the `/mcp` block in the Caddyfile, or add a Cloudflare ingress rule for `/mcp` → `localhost:8000`.
