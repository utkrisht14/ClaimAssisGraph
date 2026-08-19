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
        return {
            "risk_signals": evaluate_risk_signals(state["claim"]),
            "audit_events": _append_event(state, "risk_triaged"),
        }



    def draft_missing_response_information(self, state:ClaimAssistantState) -> dict:
        missing = ", ".join(state.get("missing_fields", []))
        message = (
            "Thanks for starting your claim. To help route it correctly, please provide the "
            f"following information: {missing}. Once we have those details, we can review the "
            "claim against the applicable policy and next-step requirements."
        )
        return {
            "response_message": message,
            "coverage_assessment": CoverageAssessment(
                decision=ClaimDecision.needs_information,
                rationale="Required intake information is missing."
            ),
            "audit_events": _append_event(state, "missing_information_response_drafted"),
        }


    def draft_customer_response(self, state:ClaimAssistantState) -> dict:
        chain = RESPONSE_PROMPT | self.llm
        response = chain.invoke(
            {
                "claim_json": state["claim"].model_dump_json(indent=2),
                "assessment_json": state["coverage_assessment"].model_dump_json(indent=2),
                "risk_json": json.dumps(
                    [signal.model_dump() for signal in state.get("risk_signals", [])],
                    indent=2,
                ),
                "context": format_context(state.get("retrieved_documents", [])),
            }
        )
        return {
            "response_message": response.content,
            "revision_count": 0,
            "audit_events": _append_event(state, "customer_response_drafted"),
        }


    def review_response(self, state:ClaimAssistantState) -> dict:
        documents = state.get("retrieved_documents", [])
        assessment = state["coverage_assessment"]

        if not documents:
            return {
                "review": ReviewResult(approved=True),
                "audit_events": _append_event(state, "response_reviewed"),
                }

        valid_doc_ids = {document.id for document in documents}
        cited_doc_ids = set(assessment.evidence_doc_ids if assessment else [])
        unsupported = cited_doc_ids - valid_doc_ids

        if unsupported:
            critique = f"Assessment cite documents that were not retrieved: {sorted(unsupported)}"
            approved = False
        elif assessment and not assessment.evidence_doc_ids:
            critique = "Coverage assessment should include at least one retrieved evidence document ID."
            approved = False
        else:
            critique = None
            approved = True

        return {
            "review": ReviewResult(approved=approved, critique=critique),
            "audit_events": _append_event(state, "response_reviewed"),
        }

    def revise_response(self, state: ClaimAssistantState) -> dict:
        chain = REVISION_PROMPT | self.llm
        documents = state.get("retrieved_documents", [])
        response = chain.invoke(
            {
                "response": state.get("response_message", ""),
                "critique": state["review"].critique or "Improve groundedness and clarity.",
                "document_ids": ", ".join(document.id for document in documents),
            }
        )
        return {
            "response_message": response.content,
            "revision_count": state.get("revision_count", 0) + 1,
            "audit_events": _append_event(state, "customer_response_revised"),
        }


def build_final_response(state: ClaimAssistantState) -> AssistantResponse:
    assessment = state["coverage_assessment"]
    return AssistantResponse(
        claim=state["claim"],
        decision=assessment.decision,
        message=state.get("response_message", ""),
        missing_fields=state.get("missing_fields", []),
        citations=state.get("retrieved_documents", []),
        risk_signals=state.get("risk_signals", []),
        audit_events=state.get("audit_events", []),
    )


def _append_event(state: ClaimAssistantState, event: str) -> list[str]:
    return [*state.get("audit_events", []), event]