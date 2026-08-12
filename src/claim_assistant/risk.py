from datetime import date

from .schemas import ClaimIntake, RiskSignal

# ---------------------------
# Define two helper functions
# ---------------------------

def _mentions_theft(claim: ClaimIntake) -> bool:
    """ Check whether the claim description appears to involve theft. """

    text = (claim.loss_description or "").lower()

    theft_keywords = (
        "stolen",
        "theft",
        "burglar",
        "break-in",
        "break in",
    )

    return any(text in word for word in theft_keywords)


def _loss_date_appears_future_dated(loss_date: str | None) -> bool:
    """ Check whether the provided loss date is later than today's date. """

    if not loss_date:
        return False

    try:
        parsed_loss_date = date.fromisoformat(loss_date[:10])
    except ValueError:
        return False

    return parsed_loss_date > date.today()


######################################
# Main program
#######################################

def evaluate_risk_signals(claim: ClaimIntake) -> list[RiskSignal]:
    """ Run deterministic business rules that should not depend on the LLM. """

    signals: list[RiskSignal] = []

    if claim.estimated_loss_amount and claim.estimated_loss_amount > 100000:
        signals.append(
            RiskSignal(
                code="HIGH_VALUE_LOSS",
                severity="medium",
                description="Estimated loss amount is high enough to require adjuster review.",
            )
        )

    if claim.injuries_reported:
        signals.append(
            RiskSignal(
                code="INJURY_REPORTED",
                severity="high",
                description="Claim includes reported injuries and should be escalated promptly.",
            )
        )

    if claim.prior_claims_amount is not None and claim.police_report_number:
        signals.append(
            RiskSignal(
                code="MULTIPLE_PRIOR_CLAIMS",
                severity="medium",
                description="Customer indicates multiple prior claims."
            )
        )

    if _mentions_theft(claim) and not claim.police_report_number:
        signals.append(
            RiskSignal(
                code="THEFT_WITHOUT_POLICE_REPORT",
                severity="high",
                description="Theft-related claim may require a police report number."
            )
        )

    if _loss_date_appears_future_dated(claim.loss_date):
        signals.append(
            RiskSignal(
                code="FUTURE_DATED_LOSS",
                severity="high",
                description="Loss date appears to be in the future.",
            )
        )