# Set up Bilson AI on a new server (full runbook)

End state: a public HTTPS site at `https://your.domain` where people sign up,
get API keys, and call `POST /v1/search`; an admin panel for you; everything
served by one CPU process that loads the search model once.

**Prereqs**
- The index is already built on the GPU (see [../notebooks/embed_on_pcai.ipynb](../notebooks/embed_on_pcai.ipynb)) → you have `index.faiss`.
- A domain name with an **A record → your server's IP** (needed for the TLS cert).
- Spec: 4 vCPU · 8 GB RAM · 40 GB SSD (see [SERVE.md](SERVE.md) for host picks — Oracle Free or Netcup).

---

## 1. Prep the data (on your build machine)
```bash
uv run python scripts/slim_db.py library.db library-serve.db    # ~3.7 GB
```

## 2. Provision the box
Create the instance (Ubuntu 24.04), add your SSH key. Open **80 and 443** in the
provider firewall (on **Oracle**, also in the instance: see [SERVE.md](SERVE.md)).
Use a non-root user named `anglican` (or edit the unit files to match yours):
```bash
ssh ubuntu@SERVER_IP
sudo adduser --disabled-password --gecos "" anglican
sudo usermod -aG sudo anglican
sudo rsync -a ~/.ssh/ /home/anglican/.ssh/ && sudo chown -R anglican:anglican /home/anglican/.ssh
```

## 3. Install the app
```bash
ssh anglican@SERVER_IP
git clone https://github.com/Timon-Stacy/anglican-search-starter
cd anglican-search-starter
bash deploy/serve_setup.sh          # CPU torch + web deps + caches the models (~few min)
```

## 4. Upload the data (from your build machine)
```bash
scp library-serve.db index.faiss anglican@SERVER_IP:~/anglican-search-starter/data/
```

## 5. Configure + start the service
```bash
openssl rand -hex 32                 # copy this — your session secret
sudo cp deploy/systemd/bilson-ai.service /etc/systemd/system/
sudo nano /etc/systemd/system/bilson-ai.service
#   set BILSON_SECRET=<the value above>
#   set BILSON_ADMIN_EMAIL=<your email>   (this account becomes admin on signup)
#   check User= and paths match your username
sudo systemctl daemon-reload
sudo systemctl enable --now bilson-ai
journalctl -u bilson-ai -f           # wait for: "search engine ready." (~30s warmup)
curl -s localhost:8001/health        # {"status":"ok","search_engine":"ready"}
```

## 6. HTTPS with Caddy
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile        # set your.domain.example -> your domain
sudo systemctl restart caddy          # fetches the Let's Encrypt cert automatically
```

## 7. Become admin + verify
1. Open `https://your.domain` → **Sign up** with the email you set as `BILSON_ADMIN_EMAIL`.
2. You now have an **Admin** link (manage users, quotas, enable/disable).
   (Alternative, no env needed: `cd ~/anglican-search-starter && BILSON_DB=$PWD/data/accounts.db uv run python -m bilson_ai.admin create-admin you@example.com 'password'`.)
3. On your **Dashboard**, create an API key, then test it:
```bash
curl -s https://your.domain/v1/search \
  -H "Authorization: Bearer bk_your_key" -H "Content-Type: application/json" \
  -d '{"query":"the eternal generation of the Son","top_k":3}'
```

Done — that's a complete service: marketing page, accounts, API keys, usage
quotas, admin panel, documented REST API, TLS, and a health check.

---

## Day-2 operations
```bash
sudo systemctl restart bilson-ai       # after swapping in a new index/DB
journalctl -u bilson-ai -f             # app logs
sudo systemctl status caddy            # TLS / proxy
```
- **Update code:** `git pull` in `~/anglican-search-starter`, then restart the service.
- **Back up accounts:** copy `data/accounts.db` (users, hashed passwords, hashed keys, usage).
- **Quotas:** default is `BILSON_MONTHLY_LIMIT`; override per user in the admin panel.

## Optional: also expose the MCP endpoint
For MCP clients (in addition to the REST API), run the `anglican-mcp` service
(see [SERVE.md](SERVE.md)) and uncomment the `/mcp` block in the Caddyfile — it
gates MCP with the same Bilson API keys via `forward_auth`. Note this loads a
second copy of the model (~6 GB extra RAM), so use a 16 GB box for both.
