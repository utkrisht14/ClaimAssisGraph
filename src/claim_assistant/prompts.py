from langchain_core.prompts import ChatPromptTemplate

INTAKE_PROMPTS = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured insurance claim intake details. "
            "Only use facts present in the customer message. "
            "If a field is unknown, leave it null or use 'unknown' for claim_type.",
        ),
        ("human", "{customer_message}"),
    ]
)


COVERAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an insurance claim support analyst. Assess the claim using only the "
            "retrieved policy/procedure context. Do not invent policy terms. Do not make a "
            "final legal or coverage determination; provide first-pass decision support.",
        ),
        (
            "human",
            "Claim facts:\n{claim_json}\n\nRetrieved context:\n{context}\n\n"
            "Return a coverage assessment with evidence document IDs.",
        )
    ]
)


RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Draft a concise, professional claim support response for a customer. "
            "Be empathetic, specific about next steps, and cite relevant document IDs in brackets. "
            "Do not promise claim approval, payment, or timing beyond the provided context.",
        ),
        (
            "human",
            "Claim facts:\n{claim_json}\n\nCoverage assessment:\n{assessment_json}\n\n"
            "Risk signals:\n{risk_json}\n\nRetrieved context:\n{context}",
        ),
    ]
)


REVISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Revise the claim response so it is grounded, cautious, and customer-ready. "
            "Keep the answer concise and only cite document IDs that appear in the retrieved context.",
        ),
        (
            "human",
            "Original response:\n{response}\n\nReview critique:\n{critique}\n\n"
            "Available document IDs:\n{document_ids}",
        ),
    ]
)