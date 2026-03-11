# iox_src_client

Small Rust CLI publisher for the daemon's Iceoryx2 src ingress path.

## What it sends

- Data topic: `<service>_query_src`
- Event topic: `<service>_query_src_query_event`

By default, `<service>` is `HymekoFastState`, so the client sends to:

- `HymekoFastState_query_src`
- `HymekoFastState_query_src_query_event`

## File

- `hymeko_daemon/src/bin/iox_src_client.rs`

## Quick run

```powershell
Set-Location "D:\Hakiko\hymeko_framework"
cargo run -p hymeko_daemon -- --service HymekoFastState
```

```powershell
Set-Location "D:\Hakiko\hymeko_framework"
cargo run -p hymeko_daemon --bin iox_src_client -- --service HymekoFastState --max-slice-len 65536 --message "BenchmarkCase`n{}`ncontext`n{`n    n0{}`n}" --repeat 1
```

## Common options

- `--service <name>`: base daemon service name
- `--message <text>`: UTF-8 payload body to send
- `--repeat <n>`: number of sends
- `--interval-ms <ms>`: delay between sends
- `--max-slice-len <bytes>`: informational value printed by the client for trace/debug; current Iceoryx2 API in this workspace does not expose a direct service-side max-slice override on this builder chain.

If you see `ExceedsMaxLoanSize`, it means the existing `<service>_query_src` service was created with a smaller (or zero) loan size limit. Restart daemon+client with a fresh service name (for example `--service HymekoFastStateV2`) so the service is recreated cleanly.
