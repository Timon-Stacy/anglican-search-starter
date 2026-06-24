# Deploying the embed job on HPE Private Cloud AI

This runs the FAISS index build (Qwen3-Embedding-0.6B @ 512-d) as a GPU batch
job on HPE Private Cloud AI (PCAI / AI Essentials), which is Kubernetes-based
with NVIDIA GPUs. Input is `library.db`; output is `index.faiss` on a persistent
volume you can pull back down.

Two paths — the **Kubernetes Job** is the production route; the **notebook** is
the fastest one-off.

---

## What's in here

| File | Purpose |
|---|---|
| `../Dockerfile` | GPU image; torch cu128 wheels + pre-baked Qwen3 embedder (no runtime model egress) |
| `entrypoint.sh` | Ensures the DB is present (optionally downloads it once), runs the resumable embed |
| `k8s/pvc.yaml` | 20 Gi PersistentVolumeClaim for `library.db` + `index.faiss` |
| `k8s/embed-job.yaml` | The GPU Job (`nvidia.com/gpu: 1`) that builds the index |

The job is **resumable** — if it's interrupted, re-applying it continues from
`embeddings_status` and the last saved index window.

---

## Option A — Kubernetes Job (recommended)

### 1. Build and push the image
On a build host with Docker and internet (the build downloads torch + the Qwen3
model and bakes them in, so the cluster needs no egress):

```bash
docker build -t <REGISTRY>/anglican-embed:latest .
docker push <REGISTRY>/anglican-embed:latest
```
Use your PCAI-accessible registry. Then set that image in `k8s/embed-job.yaml`
(the `image:` field).

### 2. Get cluster access
From the AI Essentials UI, download your kubeconfig (or use the provided
`kubectl` access), then:
```bash
export KUBECONFIG=~/Downloads/pcai-kubeconfig
kubectl get nodes        # confirm GPU nodes are visible
kubectl get storageclass # pick one for pvc.yaml if the default isn't right
```

### 3. Create the volume
```bash
kubectl apply -f deploy/k8s/pvc.yaml
```

### 4. Stage `library.db` onto the volume
Pick one:

- **Object storage (simplest, self-staging):** put `library.db` in your MinIO/S3
  bucket and uncomment `ANGLICAN_DB_URL` in `embed-job.yaml`. The job downloads
  it once onto the PVC.
- **Copy directly:** start a tiny helper pod that mounts the PVC and `kubectl cp`
  into it:
  ```bash
  kubectl run stage --image=busybox --restart=Never --overrides='
  {"spec":{"containers":[{"name":"stage","image":"busybox","command":["sleep","3600"],
  "volumeMounts":[{"name":"d","mountPath":"/data"}]}],
  "volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"anglican-data"}}]}}'
  kubectl cp library.db stage:/data/library.db
  kubectl delete pod stage
  ```

### 5. Run the job and watch it
```bash
kubectl apply -f deploy/k8s/embed-job.yaml
kubectl logs -f job/anglican-embed
```
You'll see `embedded N/1230281 (… chunks/s, ETA … min)` progress. On an H100 NVL
expect this to be far faster than a workstation GPU.

### 6. Retrieve the index
```bash
# re-use a helper pod mounting the PVC, then:
kubectl cp stage:/data/index.faiss ./index.faiss
```
Ship `index.faiss` (and `library.db`) to the CPU serving box.

---

## Option B — Jupyter notebook (fastest one-off)

In AI Essentials, open a **GPU notebook** workspace, then in a terminal:
```bash
git clone <your-repo> && cd anglican_search_starter
pip install uv && uv sync
# upload library.db into the workspace (file browser or object storage), then:
ANGLICAN_DB=./library.db ANGLICAN_INDEX=./index.faiss \
  uv run python -m anglican_search.embed_library --phase all --encode-batch 512
```
Download `index.faiss` from the file browser when it finishes.

---

## Tuning & notes

- **Throughput:** Qwen3-0.6B is a causal decoder; PyTorch SDPA gives FlashAttention-2
  speed on H100 automatically. If a run is slow, raise `--encode-batch` (the H100
  NVL has 94 GB) and confirm the GPU is actually attached (`nvidia-smi` prints in
  the logs at startup).
- **Re-chunking:** not needed for the Qwen3 swap — `--phase all` skips re-chunking
  if `rechunk_status` is already populated and goes straight to embedding.
- **Air-gap:** the embedder is baked into the image, so the job needs no Hugging
  Face access. (The reranker is only needed at serve time, not for embedding.)
- **Switching to Qwen3-Embedding-4B later:** change `EMBEDDING_MODEL` (and
  `EMBEDDING_TRUNCATE_DIM`) in `config.py`, rebuild the image, clear
  `embeddings_status`, and re-run — the H100 handles the heavier model easily.
- **GPU scheduling:** if pods stay `Pending`, check the node taint and adjust the
  `tolerations`/`nodeSelector` in `embed-job.yaml` to match your GPU pool.
