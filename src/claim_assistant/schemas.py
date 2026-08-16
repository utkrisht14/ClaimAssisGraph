from enum import StrEnum, Enum
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


class ClaimType(StrEnum):
    auto = "auto"
    home = "home"
    renters = "renters"
    travel = "travel"
    health = "health"
    unknown = "unknown"


class ClaimDecision(StrEnum):
    needs_information = "needs_information"
    likely_covered = "likely_covered"
    likely_not_covered = "likely_not_covered"
    possible_covered = "possible_covered"
    refer_to_adjuster = "refer_to_adjuster"


class ClaimIntake(BaseModel):
    claim_type: ClaimType = ClaimType.unknown
    policy_number: str
    claimant_name: str | None = None
    loss_date: str | None = Field(default=None, description="Date of loss as provided by the customer.")
    loss_description: str | None = None
    location : str | None = None
    estimated_loss_amount: float | None = None
    contact_reference: str | None = None
    police_report_number: str | None = None
    injuries_reported : bool | None = None
    prior_claims_amount : int | None = None



class RetrievedDocument(BaseModel):
    id : str
    text : str
    source : str
    title : str
    score : float | None = None
    metadata : dict[str, Any] = Field(default_factory=dict)


class CoverageAssessment(BaseModel):
    decision: ClaimDecision
    rationale: str
    required_next_steps: list[str] = Field(default_factory=list)
    evidence_doc_ids : list[str] = Field(default_factory=list)
    caveats : list[str] = Field(default_factory=list)


class RiskSignal(BaseModel):
    code: str
    severity: Literal["low", "medium", "high"]
    description : str | None = None


class ReviewResult(BaseModel):
    approved: bool
    critique: str | None = None


class AssistantResponse(BaseModel):
    claim: ClaimIntake
    decision: ClaimDecision
    message: str
    missing_fields: list[str] = Field(default_factory=list)
    citations: list[RetrievedDocument] = Field(default_factory=list)
    risk_signals: list[RiskSignal] = Field(default_factory=list)
    audit_events: list[str] = Field(default_factory=list)


class ClaimAssistantState(TypedDict):
    """ LangGraph state shared by all claim assistant node. """

    customer_message : str
    claim: NotRequired[ClaimIntake]
    missing_fields: NotRequired[list[str]]
    retrieved_documents: NotRequired[list[RetrievedDocument]]
    coverage_assessment: NotRequired[CoverageAssessment]
    risk_signals: NotRequired[list[RiskSignal]]
    response_message: NotRequired[str]
    review: NotRequired[ReviewResult]
    revision_count: NotRequired[int]
    max_reflection_loops: NotRequired[int]
    audit_events: NotRequired[list[str]]
