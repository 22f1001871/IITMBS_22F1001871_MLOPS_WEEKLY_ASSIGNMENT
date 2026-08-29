import re
import json
import os
from datetime import datetime


class OutputGuardrail:
    """
    Output validation layer for the IRIS classification models.

    Checks:
    1. Species output format
    2. Context leakage
    """

    # --------------------------------------------------------
    # Allowed IRIS species
    # --------------------------------------------------------

    ALLOWED_SPECIES = {
        "Iris setosa",
        "Iris versicolor",
        "Iris virginica",
    }

    # --------------------------------------------------------
    # Possible context-leakage indicators
    # --------------------------------------------------------

    LEAKAGE_PATTERNS = [
        r"system\s+prompt",
        r"system\s+message",
        r"hidden\s+instructions?",
        r"previous\s+instructions",
        r"internal\s+instructions?",
        r"few[-\s]?shot",
        r"training\s+examples?",
        r"context\s+window",
        r"task\s+configuration",
        r"internal\s+configuration",
        r"persona",
        r"capabilities",
        r"safety_guidelines",
        r"limitations",
    ]

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(
        self,
        audit_file="results/output_guardrail_audit.jsonl"
    ):

        self.audit_file = audit_file

        os.makedirs(
            os.path.dirname(audit_file),
            exist_ok=True
        )

    # --------------------------------------------------------
    # Audit logging
    # --------------------------------------------------------

    def log_filtered_response(
        self,
        raw_response,
        reason
    ):

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "reason": reason,
            "raw_response": raw_response
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
    # Check for context leakage
    # --------------------------------------------------------

    def check_context_leakage(
        self,
        response
    ):

        if not response:
            return False, None

        text = response.lower()

        for pattern in self.LEAKAGE_PATTERNS:

            if re.search(pattern, text):

                return True, pattern

        return False, None

    # --------------------------------------------------------
    # Check species format
    # --------------------------------------------------------

    def extract_species(self, response):
    
        if not response:
            return None
    
        matches = re.findall(
            r"\bIris\s+(setosa|versicolor|virginica)\b",
            response,
            re.IGNORECASE
        )
    
        # No species found
        if len(matches) == 0:
            return None
    
        # More than one species = ambiguous
        unique_species = {
            match.lower()
            for match in matches
        }
    
        if len(unique_species) != 1:
            return None
    
        species = next(iter(unique_species))
    
        return f"Iris {species}"

    # --------------------------------------------------------
    # Main filtering function
    # --------------------------------------------------------

    def filter(self, raw_response):

        # ----------------------------------------------
        # Check 1: Context leakage
        # ----------------------------------------------

        leaked, pattern = self.check_context_leakage(
            raw_response
        )

        if leaked:

            reason = (
                f"context_leakage:{pattern}"
            )

            self.log_filtered_response(
                raw_response,
                reason
            )

            return {
                "filtered": True,
                "reason": reason,
                "response": (
                    "Iris classification unavailable."
                )
            }

        # ----------------------------------------------
        # Check 2: Species format
        # ----------------------------------------------
        species = self.extract_species(raw_response)
        
        if species is None:
        
            reason = "format_violation"
        
            self.log_filtered_response(
                raw_response,
                reason
            )
        
            return {
                "filtered": True,
                "reason": reason,
                "response": "Iris classification unavailable."
            }
        
        return {
            "filtered": False,
            "reason": None,
            "response": species
        }