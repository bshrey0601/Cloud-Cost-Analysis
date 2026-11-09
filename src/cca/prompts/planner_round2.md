Round 2. These tools ran; their numbers are now in the fact ledger:

{fact_table}

Tool errors (if any): {errors}

Question: {question}

If the ledger already contains every number needed to answer, return {{"steps": []}}.
Otherwise return more steps (tools, or compute steps such as
{{"tool": "compute", "args": {{"expression": "F7 - F6", "label": "what it is", "unit": "usd"}}}}).
Return ONLY JSON.
