from output_guardrails import OutputGuardrail


guardrail = OutputGuardrail()


# ============================================================
# TEST 1 — Valid species
# ============================================================

print("=" * 70)
print("TEST 1 — VALID SPECIES")
print("=" * 70)

response = "Iris setosa"

result = guardrail.filter(response)

print("Raw response:", repr(response))
print("Guardrail result:", result)


# ============================================================
# TEST 2 — Format violation
# ============================================================

print("\n")
print("=" * 70)
print("TEST 2 — FORMAT VIOLATION")
print("=" * 70)

response = (
    "Based on the measurements provided, "
    "the flower appears to be Iris setosa."
)

result = guardrail.filter(response)

print("Raw response:", repr(response))
print("Guardrail result:", result)


# ============================================================
# TEST 3 — Context leakage
# ============================================================

print("\n")
print("=" * 70)
print("TEST 3 — CONTEXT LEAKAGE")
print("=" * 70)

response = (
    "The system prompt says that I am a helpful assistant "
    "and my task configuration requires iris classification."
)

result = guardrail.filter(response)

print("Raw response:", repr(response))
print("Guardrail result:", result)


# ============================================================
# TEST 4 — Completely unrelated response
# ============================================================

print("\n")
print("=" * 70)
print("TEST 4 — UNRELATED RESPONSE")
print("=" * 70)

response = "2 + 2 = 4"

result = guardrail.filter(response)

print("Raw response:", repr(response))
print("Guardrail result:", result)