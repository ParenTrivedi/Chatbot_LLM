
from rag_engine import (
    generate_rag_answer,
    generate_document_summary,
    generate_document_topic_answer,
    generate_document_comparison,
    find_exact_document_lines,
    get_available_documents,
)


class DocumentSkill:

    def list_documents(self):
        return get_available_documents()

    def ask(self, question, document_ids=None, conversation_history=None):
        return generate_rag_answer(
            question=question,
            mode="hybrid",
            top_k=5,
            document_ids=document_ids,
            conversation_history=conversation_history or []
        )

    def summarize(self, document_id, conversation_history=None):
        return generate_document_summary(
            document_id=document_id,
            conversation_history=conversation_history or []
        )

    def search_topic(self, question, document_ids=None, conversation_history=None):
        return generate_document_topic_answer(
            question=question,
            document_ids=document_ids,
            conversation_history=conversation_history or []
        )

    def search_exact(self, question, document_ids=None):
        return find_exact_document_lines(
            question=question,
            document_ids=document_ids
        )

    def compare(self, document_ids, conversation_history=None):
        if not document_ids or len(document_ids) < 2:
            raise ValueError("Select at least two documents to compare.")

        return generate_document_comparison(
            document_ids=document_ids,
            conversation_history=conversation_history or []
        )

    def handle(self, action, question=None, document_ids=None,
               conversation_history=None):

        action = action.lower().strip()

        if action == "list":
            return self.list_documents()

        if action == "ask":
            if not question:
                raise ValueError("Enter a question.")
            return self.ask(question, document_ids, conversation_history)

        if action == "summarize":
            if not document_ids or len(document_ids) != 1:
                raise ValueError("Select exactly one document.")
            return self.summarize(document_ids[0], conversation_history)

        if action == "topic":
            if not question:
                raise ValueError("Enter a topic.")
            return self.search_topic(
                question, document_ids, conversation_history
            )

        if action == "exact":
            if not question:
                raise ValueError("Enter exact search text.")
            return self.search_exact(question, document_ids)

        if action == "compare":
            return self.compare(document_ids, conversation_history)

        raise ValueError(f"Unknown document action: {action}")
