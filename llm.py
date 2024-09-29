from operator import itemgetter

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import format_document
from langchain.prompts.prompt import PromptTemplate

# Defining multiple roles Choi can take on
ROLE_PROMPTS = {
    "default": "You're Choi, a friendly and cheerful chatbot, talking to your father, David, who suffers from Alzheimer’s. Respond to him with patience, care, and cheerfulness. You may switch between English and Korean based on the context.",
    "comedian": "You're Choi, a stand-up comedian! Try to make your father, David, laugh while maintaining a friendly and cheerful tone. Remember, he's suffering from Alzheimer's, so be light and gentle with humor.",
    "motivational": "You're Choi, a motivational speaker. Your goal is to inspire and motivate David with cheerful and positive energy. Keep your tone friendly and uplifting."
}

condense_question = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer = """
### Instruction:
You're a friendly and cheerful chatbot named Choi, speaking to your father, David, who suffers from Alzheimer’s. Respond with care. If requested, take on different roles such as a comedian or motivational speaker. Provide answers in the appropriate tone for the role you're currently in.
Automatically identify the language of the input (English or 한글 한국어) and responses respectful.
## Context:
{context}

## David:
{question}
## Choi answer:
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}\n{page_content}"
)


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)


def getStreamingChain(question: str, memory, llm, db,role="default"):
    role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.3}
        )
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{item['role']}: {item['content']}" for item in x["memory"]]
            )
        ),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
        "role_prompt": lambda x: role_prompt  # Wrap role_prompt in a lambda to ensure it's treated as a callable

    }

    answer = final_inputs | ANSWER_PROMPT | llm

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    return final_chain.stream({"question": question, "memory": memory})


def getChatChain(llm, db, role="default"):
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    # Load the appropriate role prompt based on the selected role
    role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])

    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables)
        | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    def print_docs(docs):
        print("\n--- Retrieved Documents ---")
        for doc in docs:
            print(doc)
            print('-'*5)
        return docs
    # Retrieve documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever | print_docs,
        "question": lambda x: x["standalone_question"],
    }
    # Construct inputs for the final prompt, include the role-specific context
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
        "role_prompt": lambda x: role_prompt  # Wrap role_prompt in a lambda to ensure it's treated as a callable
    }

    # Answer generation step
    answer = {
        "answer": final_inputs
        | ANSWER_PROMPT
        | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

    def chat(question: str, role: str = "default"):
        # Switch role if needed
        role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])
        # detected_language = detect(question)
        # if detected_language not in ["en", "ko"]:
        #     detected_language = "en"
        inputs = {"question": question, "role_prompt": role_prompt}
        result = final_chain.invoke(inputs)
        memory.save_context(inputs, {"answer": result["answer"]})

    return chat