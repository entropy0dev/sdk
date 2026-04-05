# @entropy0/express

Entropy0 Trust Control Plane middleware for Express.js.

Evaluates an incoming request's target domain through the Entropy0 `/v1/decide` endpoint and routes it to the appropriate handler based on the recommended action — before your application logic runs.

**[Try it live →](https://entropy0.ai/playground)** — scan any domain instantly, no sign-up required.
**Get a free API key at [entropy0.ai/signup](https://entropy0.ai/signup)** — no credit card required.

## Install

```bash
npm install @entropy0/express
```

## Usage

```typescript
import express from "express";
import { entropy0Guard } from "@entropy0/express";

const app = express();

app.use(
  entropy0Guard({
    apiKey: process.env.ENTROPY0_API_KEY!,
    policy: "balanced",
  })
);

app.get("/proxy", (req, res) => {
  // Only reached if action is "proceed" or "proceed_with_caution"
  // req.entropy0 contains the full decision if caution was flagged
  res.json({ ok: true });
});
```

## Custom action handlers

```typescript
app.use(
  "/outbound",
  entropy0Guard({
    apiKey: process.env.ENTROPY0_API_KEY!,
    policy: "strict",

    // Extract target from a query param instead of req.hostname
    getTarget: (req) => {
      const url = req.query.url as string;
      return url ? { type: "url", value: url } : null;
    },

    // Describe the interaction context
    getInteraction: (req) => ({
      kind: "fetch",
      mode: "read_only",
      sensitivity: req.user?.role === "admin" ? "high" : "medium",
    }),

    onProceed:  (_req, _res, next, _result) => next(),
    onCaution:  (req, res, next, result)    => { req.entropy0 = result; next(); },
    onSandbox:  (_req, res) => res.status(403).json({ blocked: true, reason: "sandbox" }),
    onEscalate: (_req, res) => res.status(403).json({ blocked: true, reason: "review_required" }),
    onDeny:     (_req, res) => res.status(403).json({ blocked: true, reason: "deny" }),

    // Fail open on API errors (default) — swap for fail closed if needed
    onError: (_req, _res, next, err) => {
      console.error("Entropy0 check failed:", err);
      next();
    },

    timeoutMs: 3000,
  })
);
```

## Recommended actions

| Action | Default behavior | Meaning |
|---|---|---|
| `proceed` | `next()` | Normal interaction is safe |
| `proceed_with_caution` | `next()` + attaches `req.entropy0` | Continue with reduced trust assumptions |
| `sandbox` | 403 | Interact only in an isolated environment |
| `escalate_to_human` | 403 | Pause automation and request human review |
| `deny` | 403 | Do not proceed under this policy |

All handlers are overridable. Override `onSandbox` or `onEscalate` to queue for review instead of blocking.

## TypeScript

Full type exports — `DecisionResult`, `TargetDescriptor`, `InteractionDescriptor`, `Entropy0Options`.

`req.entropy0` is automatically typed via Express namespace augmentation.

## Requirements

- Node.js 18+
- Express 4+

## Links

- [API docs](https://entropy0.ai/docs)
- [Decision model](https://entropy0.ai/docs/decision-model)
- [Get an API key](https://entropy0.ai/signup)
