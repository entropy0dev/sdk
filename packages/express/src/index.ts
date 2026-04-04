import type { Request, Response, NextFunction } from "express";

// ── Types ──────────────────────────────────────────────────────────────────

export type RecommendedAction =
  | "proceed"
  | "proceed_with_caution"
  | "sandbox"
  | "escalate_to_human"
  | "deny";

export interface DecisionResult {
  request_id: string;
  decision: {
    recommended_action: RecommendedAction;
    action_confidence: number;
    reason_codes: string[];
  };
  uncertainty: {
    state: "low" | "medium" | "high";
    requires_human_review: boolean;
  };
  validity: {
    evaluated_at: string;
    valid_until: string;
    ttl_seconds: number;
  };
}

export interface TargetDescriptor {
  type: "domain" | "url";
  value: string;
}

export interface InteractionDescriptor {
  kind?: "navigate" | "fetch" | "enrich" | "download_file" | "submit_credentials" | "initiate_payment";
  mode?: "read_only" | "transactional" | "privileged";
  sensitivity?: "low" | "medium" | "high" | "critical";
}

export interface Entropy0Options {
  /** Your Entropy0 API key (sk_ent0_xxxx). Required. */
  apiKey: string;
  /** Override the API base URL. Defaults to https://entropy0.ai/api */
  baseUrl?: string;
  /** Policy profile to evaluate against. Defaults to "balanced". */
  policy?: "open" | "balanced" | "strict" | "critical";
  /** How to extract the target from the incoming request. Defaults to req.hostname. */
  getTarget?: (req: Request) => TargetDescriptor | null;
  /** How to describe the interaction context. Defaults to fetch / read_only / medium. */
  getInteraction?: (req: Request) => InteractionDescriptor;
  /** Called when action is "proceed". Default: calls next(). */
  onProceed?: (req: Request, res: Response, next: NextFunction, result: DecisionResult) => void;
  /** Called when action is "proceed_with_caution". Default: attaches result to req.entropy0, calls next(). */
  onCaution?: (req: Request, res: Response, next: NextFunction, result: DecisionResult) => void;
  /** Called when action is "sandbox". Default: 403. */
  onSandbox?: (req: Request, res: Response, next: NextFunction, result: DecisionResult) => void;
  /** Called when action is "escalate_to_human". Default: 403. */
  onEscalate?: (req: Request, res: Response, next: NextFunction, result: DecisionResult) => void;
  /** Called when action is "deny". Default: 403. */
  onDeny?: (req: Request, res: Response, result: DecisionResult) => void;
  /** Called when the Entropy0 API call fails. Default: fail open (calls next()). */
  onError?: (req: Request, res: Response, next: NextFunction, error: unknown) => void;
  /** Request timeout in ms. Defaults to 5000. */
  timeoutMs?: number;
}

// ── Internal API call ─────────────────────────────────────────────────────

async function callDecide(
  target: TargetDescriptor,
  interaction: InteractionDescriptor,
  policy: string,
  apiKey: string,
  baseUrl: string,
  timeoutMs: number,
): Promise<DecisionResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${baseUrl}/v1/decide`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": apiKey,
      },
      body: JSON.stringify({
        target,
        interaction: {
          kind:        interaction.kind        ?? "fetch",
          mode:        interaction.mode        ?? "read_only",
          sensitivity: interaction.sensitivity ?? "medium",
        },
        policy: { profile: policy },
      }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`Entropy0 API responded ${res.status}: ${body}`);
    }

    return res.json() as Promise<DecisionResult>;
  } finally {
    clearTimeout(timer);
  }
}

// ── Middleware factory ─────────────────────────────────────────────────────

export function entropy0Guard(options: Entropy0Options) {
  const {
    apiKey,
    baseUrl     = "https://entropy0.ai/api",
    policy      = "balanced",
    timeoutMs   = 5000,
    getTarget   = (req) => ({ type: "domain", value: req.hostname }),
    getInteraction = () => ({}),
    onProceed  = (_req, _res, next, _result) => next(),
    onCaution  = (req, res, next, result)   => { req.entropy0 = result; next(); },
    onSandbox  = (_req, res, _next, _result) => res.status(403).json({ blocked: true, action: "sandbox" }),
    onEscalate = (_req, res, _next, _result) => res.status(403).json({ blocked: true, action: "escalate_to_human" }),
    onDeny     = (_req, res, _result)        => res.status(403).json({ blocked: true, action: "deny" }),
    onError    = (_req, _res, next, _err)    => next(), // fail open by default
  } = options;

  return async function entropy0Middleware(
    req: Request,
    res: Response,
    next: NextFunction,
  ): Promise<void> {
    const target = getTarget(req);
    if (!target) return next();

    try {
      const interaction = getInteraction(req);
      const result = await callDecide(target, interaction, policy, apiKey, baseUrl, timeoutMs);

      switch (result.decision.recommended_action) {
        case "proceed":              return onProceed(req, res, next, result);
        case "proceed_with_caution": return onCaution(req, res, next, result);
        case "sandbox":              return onSandbox(req, res, next, result);
        case "escalate_to_human":    return onEscalate(req, res, next, result);
        case "deny":                 return onDeny(req, res, result);
        default:                     return next();
      }
    } catch (err) {
      return onError(req, res, next, err);
    }
  };
}

// ── Express type augmentation ──────────────────────────────────────────────

declare global {
  namespace Express {
    interface Request {
      /** Entropy0 decision result, attached when action is proceed_with_caution. */
      entropy0?: DecisionResult;
    }
  }
}
