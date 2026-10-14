# Eagle Nexus — single ZIP handoff (2026-10-14)

[Eagle_Nexus_FileOnly_20261014.zip](Eagle_Nexus_FileOnly_20261014.zip) is a **file-only** public transfer bundle containing exactly 11 unchanged runtime files, the original OAIAPI MIT LICENSE, per-file SHA-256 checksums and a runtime manifest.

Bundle ZIP SHA-256: `8a20a7d62712eecf6a606ed1f7376f3584fca7644a9767d9e3608d5fc11299fa` (117,299 bytes). Download and verify SHA-256 before staging.

Provenance:
- `s-felician/comfyui-nexus-nodes1` at `7b162848216c2df4351c833c8ed4bb02ce23912e` (Nexus 3 files)
- `s-felician/comfyui-nexus-nodes2` at `a614661fe4a034aea83450cfeb022d5f793249cc` (OAIAPI 8 files)

**Not an installer.** Extract outside the live ComfyUI folder. Use an explicitly authorized new-files-only deployment after checking host/root/interpreter, all node search directories and both queues. No auto pip, Manager, SDK upgrade, security-level change or ComfyUI restart. An already-tested private file-only deployment utility can consume the two staged `custom_nodes/` folders; it should remain private.

The OAIAPI pyproject metadata still declares `openai>=2.7.1` and requirements has an unbounded `openai`. **Never auto-install dependencies from this bundle.** Compatibility with SDK 1.109.1 was tested offline (29 tests); Windows live import and VLM are not proven. File placement, ComfyUI restart, GPU/VLM requests are separately approved steps.

Expected after approved placement and restart: `NexusImpactPrepare` (4 outputs), `NexusImpactBind` (12 outputs), `OAIAPI_Client`, `OAIAPI_ChatCompletion`. Verify with `/object_info`; keep formal 629-node workflow HOLD until new C09 authorization.

This delivery exists on a separate branch. Both public repositories' **main** remain at their original fixed SHA. No private images, prompts, LAN addresses, configuration files, or credentials are included.
