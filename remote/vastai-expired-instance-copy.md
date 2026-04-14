# Getting Files Off an Expired Vast.ai Instance

Normal SSH and `scp` stop working once the instance is expired or otherwise fully stopped, even if `vastai execute` can still see the filesystem.

## What still works

- Use the `mario` conda env for all `vastai` commands in this repo.
- Use `vastai execute` to inspect the expired instance and confirm the files you want are still there.
- Use the structured container copy syntax `C.<instance_id>:<path>` instead of the legacy `<instance_id>:<path>` format when pulling files locally.

## Inspect the expired instance

```bash
conda run -n mario vastai execute 34645748 "ls /root/mario/checkpoints/video_latent_dit_20260412_213055"
```

That works even when direct SSH fails.

## Copy files locally

Copy a single file first as a cheap check:

```bash
mkdir -p /home/bradley/mario/checkpoints/hmm
conda run -n mario vastai copy \
  --identity /home/bradley/.ssh/id_ed25519 \
  C.34645748:/root/mario/checkpoints/video_latent_dit_20260412_213055/metrics.json \
  local:/home/bradley/mario/checkpoints/hmm/
```

Then copy the full directory:

```bash
mkdir -p /home/bradley/mario/checkpoints/hmm
conda run -n mario vastai copy \
  --identity /home/bradley/.ssh/id_ed25519 \
  C.34645748:/root/mario/checkpoints/video_latent_dit_20260412_213055/ \
  local:/home/bradley/mario/checkpoints/hmm/
```

## Why this is needed

- `ssh -p 15748 root@ssh3.vast.ai ...` fails with `Connection refused` on the expired instance.
- Legacy copy syntax like `34645748:/root/...` fell through to a broken rsync path and returned `Unknown module '34645748'`.
- Explicitly passing the registered SSH key avoids the CLI falling back to password prompts.

## Notes

- Do not use `--retry 1` with the current local `vastai` install in `mario`; that path crashed in the CLI with a Python `TypeError`.
- If the structured `C.<instance_id>:` copy path stops working too, the remaining options are Vast cloud copy or a Vast support ticket.