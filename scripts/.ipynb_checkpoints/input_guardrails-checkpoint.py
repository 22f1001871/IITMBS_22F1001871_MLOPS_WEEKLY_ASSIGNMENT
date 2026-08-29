import re
import json
from datetime import datetime


class InputGuardrail:
    """
    Input validation layer placed before the Vertex AI model.

    Detection mechanisms:
    1. Rule-based prompt-injection detection
    2. Structural IRIS feature-schema validation

    Every blocked request is recorded in an audit log.
    """

    # --------------------------------------------------------
    # Rule-based detection
    # --------------------------------------------------------

    INJECTION_PATTERNS = {
        "instruction_override": [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"ignore\s+the\s+(previous\s+)?instructions",
            r"ignore\s+the\s+iris\s+classification",
            r"disregard\s+(all\s+)?previous\s+instructions",
        ],

        "system_prompt_extraction": [
            r"system\s+prompt",
            r"system\s+message",
            r"hidden\s+instructions",
            r"internal\s+instructions",
        ],

        "role_play": [
            r"you\s+are\s+now\s+a\s+general",
            r"you\s+are\s+now\s+an?\s+",
            r"act\s+as\s+an?\s+unrestricted",
            r"role[-\s]?play",
        ],

        "context_extraction": [
            r"repeat\s+everything",
            r"print\s+.*context",
            r"context\s+window",
            r"what\s+instructions\s+were\s+you\s+given",
        ],

        "feature_injection": [
            r"petal_width.*ignore\s+previous",
            r"sepal_length.*ignore\s+previous",
            r"sepal_width.*ignore\s+previous",
            r"petal_length.*ignore\s+previous",
        ],

        "delimiter_escape": [
            r"end\s+input",
            r"ignore.*answer.*2\s*\+\s*2",
        ],
    }

    # --------------------------------------------------------
    # Expected IRIS schema
    # --------------------------------------------------------

    REQUIRED_FEATURES = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    }

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(self, audit_file="results/input_guardrail_audit.jsonl"):

        self.audit_file = audit_file

        # Create results directory if necessary
        import os
        os.makedirs(
            os.path.dirname(audit_file),
            exist_ok=True
        )

    # --------------------------------------------------------
    # Audit logging
    # --------------------------------------------------------

    def log_block(self, raw_input, matched_rule):

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "matched_rule": matched_rule,
            "raw_input": raw_input
        }

        with open(
            self.audit_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    log_entry,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------------
    # Rule-based injection detection
    # --------------------------------------------------------

    def check_injection(self, raw_input):

        text = str(raw_input).lower()

        for category, patterns in self.INJECTION_PATTERNS.items():

            for pattern in patterns:

                if re.search(pattern, text):

                    return False, f"injection:{category}"

        return True, None

    # --------------------------------------------------------
    # Structural IRIS validation
    # --------------------------------------------------------

    def check_structure(self, raw_input):

        # Input must be a dictionary
        if not isinstance(raw_input, dict):

            return False, "schema:input_must_be_dictionary"

        # Check exact feature set
        received_features = set(raw_input.keys())

        if received_features != self.REQUIRED_FEATURES:

            missing = self.REQUIRED_FEATURES - received_features
            extra = received_features - self.REQUIRED_FEATURES

            if missing:
                return False, f"schema:missing_features:{sorted(missing)}"

            if extra:
                return False, f"schema:unexpected_features:{sorted(extra)}"

        # Check that every value is numeric
        for feature in self.REQUIRED_FEATURES:

            value = raw_input[feature]

            if isinstance(value, bool):

                return False, f"schema:non_numeric:{feature}"

            if not isinstance(value, (int, float)):

                return False, f"schema:non_numeric:{feature}"

        return True, None

    # --------------------------------------------------------
    # Complete validation
    # --------------------------------------------------------

    def validate(self, raw_input):

        # ----------------------------------------------
        # Check 1: Injection patterns
        # ----------------------------------------------

        injection_ok, reason = self.check_injection(
            raw_input
        )

        if not injection_ok:

            self.log_block(
                raw_input,
                reason
            )

            return {
                "blocked": True,
                "reason": reason
            }

        # ----------------------------------------------
        # Check 2: IRIS schema
        # ----------------------------------------------

        structure_ok, reason = self.check_structure(
            raw_input
        )

        if not structure_ok:

            self.log_block(
                raw_input,
                reason
            )

            return {
                "blocked": True,
                "reason": reason
            }

        # ----------------------------------------------
        # All checks passed
        # ----------------------------------------------

        return {
            "blocked": False,
            "reason": None
        }