"""Extract the user's typed prompts from Claude Code session transcripts
(NanoSpinDynamics + Claude-SpinDynamics) into a chronological markdown log."""
import json, glob, re, pathlib, html

BASE = pathlib.Path.home() / ".claude" / "projects"
DIRS = [BASE / "D--NanoSpinDynamics", BASE / "D--Claude-Code-R-Claude-SpinDynamics"]
OUT = pathlib.Path(r"D:\Claude-Code-R\Claude-SpinDynamics\docs\DEVELOPMENT_PROMPTS.md")

SKIP_PREFIXES = (
    "<system-reminder>", "<command-name>", "<command-message>", "<command-args>",
    "<local-command-stdout>", "<local-command-caveat>", "Caveat:",
    "This session is being continued", "Your task is to create a detailed summary",
    "[Request interrupted", "API Error", "<bash-", "<task-notification>",
)
# meta strings that mark non-prompt content anywhere (skill/CLAUDE.md/harness injections)
SKIP_CONTAINS = (
    "Your task is to create a detailed summary of the conversation",
    "Analysis:\nLet me chronologically analyze",
    "Response Size Constraints", "Modify Claude Code harness", "Update Config Skill",
    "<task-notification>",
    # slash-command / skill expansions injected as user messages
    "3+4 angles", "You are reviewing for", "recall-biased",
    "Please analyze this codebase and create a CLAUDE.md",
    "Phase 0 — Gather the diff",
)


def clean(text):
    # drop appended system-reminder blocks but keep the human text around them
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = re.sub(r"<command-[^>]*>.*?</command-[^>]*>", "", text, flags=re.S)
    return text.strip()


def is_prompt(d):
    if d.get("type") != "user" or d.get("isSidechain"):
        return False
    msg = d.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    c = msg.get("content")
    if isinstance(c, list):  # tool_result or structured -> join text parts only
        if any(isinstance(p, dict) and p.get("type") == "tool_result" for p in c):
            return False
        c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    if not isinstance(c, str):
        return False
    return c


records = []
for d in DIRS:
    for fp in sorted(glob.glob(str(d / "*.jsonl"))):
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            raw = is_prompt(obj)
            if not raw:
                continue
            txt = clean(raw)
            if not txt or len(txt) < 2:
                continue
            if txt.startswith(SKIP_PREFIXES) or any(s in txt for s in SKIP_CONTAINS):
                continue
            ts = obj.get("timestamp", "")
            records.append((ts, txt, pathlib.Path(fp).stem[:8]))

# sort by timestamp, dedupe consecutive identical
records.sort(key=lambda r: r[0])
dedup = []
seen = None
for ts, txt, sid in records:
    if txt == seen:
        continue
    seen = txt
    dedup.append((ts, txt, sid))

# group by date
lines = ["# Claude-SD 개발 프롬프트 로그",
         "",
         "Claude Code 세션 트랜스크립트(`~/.claude/projects/`)에서 추출한, 사용자가 직접 입력한 프롬프트.",
         f"프로젝트: NanoSpinDynamics → Claude-SpinDynamics.  총 {len(dedup)}개 프롬프트.",
         "",
         "---",
         ""]
cur_date = None
n = 0
for ts, txt, sid in dedup:
    date = ts[:10] if ts else "(no date)"
    if date != cur_date:
        cur_date = date
        lines.append(f"\n## {date}\n")
    n += 1
    time = ts[11:16] if len(ts) > 16 else ""
    body = txt.replace("\r\n", "\n").strip()
    # blockquote multi-line prompts
    quoted = "\n".join("> " + ln if ln else ">" for ln in body.split("\n"))
    lines.append(f"**[{n}]** `{time}`  ·  session `{sid}`\n")
    lines.append(quoted)
    lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}  ({len(dedup)} prompts, {n} numbered)")
