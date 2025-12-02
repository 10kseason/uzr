# Security Guide (EXAONE × UZR Memory Gateway)

This document summarizes the hardening knobs implemented in `mem_server.py` and the matching client headers.

## Server-side (env vars)

- `MEM_READ_KEY` / `MEM_WRITE_KEY`: if set, requires Bearer token (Authorization) or `X-API-Key` for reads/writes.
- `MEM_HMAC_KEY`: if set, requires `X-Signature: <sha256>` HMAC over request body.
- `MEM_IP_ALLOW_READ` / `MEM_IP_ALLOW_WRITE`: comma-separated allowlist of IPs per method-class.
- `MEM_CORS_ORIGIN`: CORS `Access-Control-Allow-Origin` (default `*`).
- `MEM_MAX_BODY_BYTES`: max payload size (default `1048576`).
- `MEM_MAX_OPS`: max operations in a write envelope (default `256`).
- `MEM_MAX_SSE`: max concurrent SSE clients (default `64`).
- `MEM_MAX_SSE_PER_IP`: max SSE clients per IP (default `16`).
- `MEM_PROJECT_ALLOW`: comma-separated project allowlist (e.g., `uzr,exaone`). Enforced on UPSERT and `/mem/search` (when `MEM_PROJECT_REQUIRE=1`).
- `MEM_PROJECT_REQUIRE`: if truthy, `/mem/search` requires a `filters.project` subset of the allowlist.

## Client-side (env vars → headers)

- `UZR_MEM_TOKEN`: sent as `Authorization: Bearer <token>`.
- `UZR_MEM_HMAC_KEY`: if set, `X-Signature` is computed as HMAC-SHA256 over JSON body.
- `UZR_CLIENT_ID`: included in tags for writes as `client:<id>`.

## TLS / Reverse proxy (example sketch)

```nginx
server {
  listen 443 ssl;
  server_name mem.example.com;

  ssl_certificate     /etc/letsencrypt/live/mem.example.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/mem.example.com/privkey.pem;

  client_max_body_size 1m;
  proxy_read_timeout  60s;
  proxy_connect_timeout 5s;

  location / {
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_pass http://127.0.0.1:8088;
  }
}
```

Recommended: enable rate limiting and WAF at the edge, and pin TLS.

