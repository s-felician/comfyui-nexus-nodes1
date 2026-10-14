# Public ComfyUI node handoff slot 1

This is a **reusable delivery repository**, not the authoritative development repository and not permanently dedicated to Nexus. Its contents may be replaced for a different ComfyUI node pack in the future. Always install **by an audited commit SHA**, not by assuming `main` remains the same package.

## Current payload (2026-10-14)

**Nexus Impact Binding**: a self-contained ComfyUI custom-node package for the 629-node Eagle Nexus integration candidate.

- `NexusImpactPrepare` (4 outputs)
- `NexusImpactBind` (12 outputs)
- Supporting modules: `face_anchor.py`, `safe_extent.py`

These runtime files are placed at the **repository root** so ComfyUI can load the cloned repository as a single node pack.

## Installation and safety

Use the Artokun ComfyUI MCP `install_custom_node` with a `git` source and **fixed commit** once the remote Manager policy and an empty render queue have been checked. Installation, Python dependency changes and ComfyUI restart are separate operational steps; publishing this repository does **not** authorize them.

This is a **candidate**, not evidence of successful Windows import, vision-model identification, GPU processing, or final image quality. After installation/restart verify the actual node classes and output port schemas in `/object_info` before attempting to run the 629-node workflow. Do not alter existing Impact/SAM packages or production workflows.

## Runtime fingerprint

```text
4ad5615721be31d670843593a76deaff65bd3673d0140324d667a8299961a28b  __init__.py
91fa364a9611bda0b0b2f2e4dff8d10fd1019e21fadc33b2f2b79d8450b2c7ad  face_anchor.py
3f6b99776e48f1b8eed5e15ce019b81d75c070fa429651d06a803b2eee458161  safe_extent.py
```

Only the audited three Python files are required at runtime. The public repository does not contain private workflows, prompts, images, endpoints, credentials or execution logs.
