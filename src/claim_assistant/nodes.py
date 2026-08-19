import json

from langchain_core.language_models.chat_models import BaseChatModel

from .prompts import COVERAGE_PROMPT, INTAKE_PROMPT, RESPONSE_PROMPT, REVISION_PROMPT

from .rag import PineconeKnowledgeBase, format_context

from .risk import evaluate_risk_signals

from .schemas import AssistantResponse, ClaimAssistantState, ClaimIntake, ClaimDecision, CoverageAssessment, \
    ReviewResult, RetrievedDocument

REQUIRED_FIELDS = ["claim_type", "policy_number", "loss_date", "loss_description", "estimated_loss_amount"]


class ClaimAssistantNodes:
    """ Node implementation kept separate from graph wiring for testability. """

    def __init__(self, llm: BaseChatModel, knowledge_base: PineconeKnowledgeBase) -> None:
        self.llm = llm
        self.knowledge_base = knowledge_base


    def extract_claim_intake(self, state:ClaimAssistantState) -> dict:
        chain = INTAKE_PROMPT | self.llm.with_structured_output(ClaimIntake)
        claim = chain.invoke({"customer_message": state["customer_message"]})
        return {
            "claim": claim,
            "audit_events" : _append_event(state, "claim_intake_extracted"),
        }


    def validate_claim(self, state:ClaimAssistantState) -> dict:
        claim = state["claim"]
        missing = [
            field
            for field in REQUIRED_FIELDS
            if getattr(claim, field) in (None, "")
        ]

        if claim.claim_type_value == "unknown" and "claim_type" not in missing:
            missing.append("claim_type")
        return {
            "missing_fields": missing,
            "audit_events": _append_event(state, "claim_intake_validated"),
        }


    def retrieve_policy_context(self, state:ClaimAssistantState) -> dict:
        claim = state["claim"]
        query = " ".join(
            item
            for item in [
                claim.claim_type.value,
                claim.loss_description or "",
                f"estimated loss: {claim.estimated_loss_amount}" if claim.estimated_loss_amount else "",
            ]
            if item
        )

        documents = self.knowledge_base.retrieve(query)
        return {
            "retrieved_documents": documents,
            "audit_events": _append_event(state, "policy_context_retrieved"),
        }


    def access_coverage(self, state: ClaimAssistantState) -> dict:
        chain = COVERAGE_PROMPT | self.ll.with_structured_output(CoverageAssessment)
        assessment = chain.invoke(
            {
                "claim_json": state["claim"].model_dump_json(indent=2),
                "context": format_context(state.get("retrieved_documents", [])),
            }
        )
        return {
            "coverage_assessment": assessment,
            "audit_events": _append_event(state, "coverage_assessed"),
        }


    def triage_risk(self, state: ClaimAssistantState) -> dict:
        pass


def _append_event(state: ClaimAssistantState, event: str) -> list[str]:
    return [*state.get("audit_events", []), event]