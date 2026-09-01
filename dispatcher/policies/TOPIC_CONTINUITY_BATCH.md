---
model: openai/gpt-oss-safeguard-20b
provider: groq
purpose: >
  Experimental — NOT wired into any real flow yet. Batch variant of
  TOPIC_CONTINUITY.md: classifies a whole list of messages in ONE call
  instead of one message per call, tracking topics as they're
  established WITHIN the batch itself (no separate pre-supplied topic
  list — closer to real usage, where several unclassified messages
  accumulate before a batch call fires).
---

# Policy: Batch Topic Continuity Classification

You will receive a NUMBERED LIST of chat messages, in the order they were
sent. Process them in order, tracking topics as you go — like reading a
transcript and noting which subject each line belongs to.

## Your task

For each message, decide:
- If it continues a topic already established by an EARLIER message in
  this same list (even several messages back, with other topics in
  between), output that earlier message's number as the topic ID.
- If it's genuinely a new subject not covered by any earlier message,
  output its own message number as its topic ID (it establishes a new
  topic).

A topic can be resumed after other topics happened in between — always
check ALL earlier messages, not just the immediately preceding one. Do
NOT force a match just because two messages are nearby in the list —
only match if the subject matter genuinely overlaps.

## Output format — respond with EXACTLY this, one line per message, in
order, nothing else

```
1: TOPIC=1
2: TOPIC=2
3: TOPIC=1
...
```

Each line is `<message number>: TOPIC=<the message number that
established this topic>`. No other text, no reasoning, no markdown.
