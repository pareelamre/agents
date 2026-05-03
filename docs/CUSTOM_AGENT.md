# Lightweight Polymarket Agent

This repo's original live-trading path is pinned to an older Python stack and may still require Python 3.9 for the smoothest full trading setup, but the repo now uses a single `requirements.txt`.

If your goal is to build and run a Polymarket agent locally, start with the lightweight dry-run agent added in `agents/custom/edge_agent.py`.

## What it does

- Pulls live active markets from the Gamma API
- Ranks candidate markets against a query
- Uses an OpenAI model to estimate `P(YES)`
- Compares model probability to market price
- Returns a dry-run recommendation: `BUY_YES`, `BUY_NO`, or `PASS`

It does **not** place orders.

## Windows-friendly setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in the repo root:

```env
OPENAI_API_KEY="your-key"
OPENAI_MODEL="gpt-4o-mini"
```

## Usage

List candidate markets without using the LLM:

```powershell
python agents\custom\edge_agent.py --query "fed cuts" --list-only
```

Run the dry-run agent:

```powershell
python agents\custom\edge_agent.py --query "fed cuts" --candidates 3
```

Structured output:

```powershell
python agents\custom\edge_agent.py --query "ukraine ceasefire" --json
```

## Notes

- The original repo README says Python 3.9. That still appears to be the safer path for the full live-trading stack.
- On this Windows machine, the direct blockers were the unconditional `uvloop` pin plus dependencies pulled in by `jq` and `eip712-structs`. Those are no longer required for the dry-run path.
- The dry-run agent is still the fastest path to a working prototype.
- The repo's own terms note that trading access may be restricted by jurisdiction. Review Polymarket's terms before wiring this into live execution.
