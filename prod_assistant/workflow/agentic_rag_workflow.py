from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
'''
PROMPT_REGISTRY: A library of pre-written prompts.
Retriever: Your custom code for fetching documents.
ModelLoader: Your helper for loading the LLM.
MemorySaver: A LangGraph feature to save the conversation history, so the agent can remember past interactions.
'''

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class AgenticRAG:
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Purpose: This defines the "memory" or "state" of your agent as it moves through the flowchart. It's a dictionary that must contain a key called messages.

    Details: The messages value is a sequence (like a list) of BaseMessage objects. 
    The add_messages part is a special helper from LangGraph that ensures new messages are correctly added to the list instead of overwriting it.

    Analogy: Think of AgentState as a clipboard that each worker in an assembly line passes to the next. The clipboard always has a sheet for "messages," and each worker adds their notes to it.
    '''
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        '''
        Retriever(): Creates an instance of your document retriever.
        ModelLoader(): Creates an instance of your model loader.
        self.model_loader.load_llm(): Loads the actual LLM so it's ready to answer questions.
        MemorySaver(): Sets up the memory system to save conversation history.
        _build_workflow(): This is crucial. It calls another method (which we'll see next) to define the actual steps and logic of the agent's flowchart.
        self.workflow.compile(...): This takes the defined flowchart (workflow) and turns it into a runnable application. The checkpointer is included so it can save its state at each step.
        '''
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helpers ----------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    A simple but important utility function. Its only job is to take the raw documents found by the retriever and format them into a clean, human-readable string that the LLM can easily understand.
    '''
    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)

    # ---------- Nodes ----------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    This is the first stop and the main decision-maker (a "router"). It inspects the user's latest message to decide what to do next.

    If the message contains keywords like "price," "review," or "product," it decides that it needs more information. 
    It returns a special message, "TOOL: retriever", which tells the graph to go to the Retriever node next.

    If the message seems like a general question (e.g., "hello"), it answers directly without using the retriever. 
    The workflow then goes straight to the end.
    '''
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    This is the "Retrieval" part of RAG. It uses the retriever object you initialized earlier to search a database for documents relevant to the user's query.

    Logic: It takes the latest message, sends it to the retriever, gets back a list of documents, formats them using the _format_docs helper, and adds the formatted text to the messages in the state.
    '''
    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content
        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)
        context = self._format_docs(docs)
        return {"messages": [HumanMessage(content=context)]}

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Purpose: This is a fantastic and advanced step! Instead of just assuming the retrieved documents are good, this node asks the LLM to act as a "grader."
    Logic: It looks at the user's original question and the retrieved documents and asks the LLM a simple question: "Are these docs relevant?"
        If the LLM says "yes," this node returns the string "generator", telling the graph to proceed to the Generator node to write an answer.
        If the LLM says "no," it returns "rewriter", telling the graph that the retrieved documents were bad and a different approach is needed.
    '''
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Purpose: This is the "Generation" part of RAG. It crafts the final answer for the user.
    Logic: It takes the original question and the relevant documents (the context) and puts them into a detailed prompt. 
    It then sends this complete prompt to the LLM to get a final, context-aware answer.
    '''
    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    '''
    Purpose: This node is the recovery plan. It's used when the _grade_documents node decided the retrieved documents were irrelevant.

    Logic: It asks the LLM to rewrite the user's original question to be clearer or more specific. 
    For example, if the user asked "tell me about that phone," this node might rewrite it to "what are the specifications and price of the iPhone 15?". 
    The new, rewritten question is then added to the state.
    '''
    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}

    # ---------- Build Workflow ----------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    Purpose: This method connects all the nodes into a logical flowchart.
    Analogy: If the nodes are the workers on an assembly line, this section is the map that shows who passes their work to whom.

    The Flow:
    START -> Assistant: Every query begins at the Assistant node.
    From Assistant: A conditional edge.
    If the assistant decided to use a tool, the flow goes to the Retriever node.
    Otherwise, the flow goes directly to END.
    From Retriever: Another conditional edge. This one uses the _grade_documents function to decide the path.
    If the function returns "generator", the flow goes to the Generator node.
    If it returns "rewriter", the flow goes to the Rewriter node.
    Generator -> END: After generating an answer, the process is finished.
    Rewriter -> Assistant: This is a cycle or loop. After rewriting the question, the workflow sends the new question back to the Assistant to try the whole process again. This is a very powerful concept.
    '''
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

    # ---------- Public Run ----------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    '''
    run method: This is the clean, public-facing function you call to use the agent. It hides all the complex internal steps. The thread_id is used by the MemorySaver to keep track of different conversations separately.
    '''
    def run(self, query: str,thread_id: str = "default_thread") -> str:
        result = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content
    
        # function call with be asscoiate
        # you will get some score
        # put condition behalf on that score
        # if relevany>0.75
            #return
        #else:
            #contine

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    rag_agent = AgenticRAG()
    answer = rag_agent.run("What is the price of iPhone 15?")
    print("\nFinal Answer:\n", answer)