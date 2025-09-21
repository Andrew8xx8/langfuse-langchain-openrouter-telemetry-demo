from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages

from .config import OpenRouterConfig, TelemetryContext
from .factories import LLMClientFactory, TelemetryFactory
from .utils import CostExtractor, MetadataBuilder


class GraphState(TypedDict):
    """State for the LangGraph workflow"""
    messages: Annotated[list, add_messages]
    current_step: str
    analysis_result: str
    final_answer: str


class TestRunner:
    """Orchestrates tests with shared infrastructure"""

    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.cost_extractor = CostExtractor()

    def run_direct_openai_test(self, telemetry_context: TelemetryContext):
        print("=== Test 1: Direct OpenAI Client ===")
        print(f"Session ID: {telemetry_context.session_id}")

        client = LLMClientFactory.create_openai_client(self.config)
        questions = ["What is 2*3?", "What is 10/2?", "What is 5+7?"]

        for i, question in enumerate(questions, 1):
            print(f"\n--- Direct Test {i}: {question} ---")

            response = client.chat.completions.create(
                model=self.config.default_model,
                messages=[
                    {"role": "system", "content": "You are a helpful math assistant."},
                    {"role": "user", "content": question}
                ],
                extra_body={"usage": {"include": True}},
                name=f"direct-openai-{i}",
                metadata=MetadataBuilder.for_openai(
                    telemetry_context,
                    test_type="direct_openai",
                    question_number=i
                )
            )

            print(f"Response: {response.choices[0].message.content}")
            if hasattr(response, 'usage') and response.usage:
                cost = getattr(response.usage, 'cost', None)
                if cost:
                    print(f"Cost: ${cost}")
                else:
                    print("No cost in response")

        print("Direct OpenAI test completed\n")

    def run_langchain_test(self, telemetry_context: TelemetryContext):
        print("=== Test 2: LangChain with Cost Tracking ===")
        print(f"Session ID: {telemetry_context.session_id}")

        llm = LLMClientFactory.create_langchain_llm(self.config)
        handler = TelemetryFactory.create_handler()
        questions = ["What is 7*8?", "What is 12*5?", "What is 144/12?"]

        for i, question in enumerate(questions, 1):
            print(f"\n--- Cost Tracking Test {i}: {question} ---")

            messages = [
                SystemMessage(content="You are a helpful math assistant."),
                HumanMessage(content=question)
            ]

            response = llm.invoke(
                messages,
                config={
                    "callbacks": [handler],
                    **MetadataBuilder.for_langchain_config(
                        telemetry_context,
                        test_type="langchain_cost_tracking",
                        question_number=i
                    )
                }
            )

            print(f"Response: {response.content}")
            cost_info = self.cost_extractor.extract_cost_info(response.response_metadata or {})
            self.cost_extractor.display_cost_info(cost_info, "in response")

        print("\nLangChain cost tracking test completed\n")

    def run_langgraph_test(self, telemetry_context: TelemetryContext):
        print("=== Test 3: LangGraph Multi-Node Workflow ===")
        print(f"Session ID: {telemetry_context.session_id}")
        print("Graph structure: analyzer → solver → validator")

        llm = LLMClientFactory.create_langchain_llm(self.config)
        handler = TelemetryFactory.create_handler()

        def analyzer_node(state: GraphState) -> GraphState:
            print("   - Running analyzer_node...")
            messages = state["messages"] + [
                SystemMessage(content="You are a mathematical problem analyzer. Analyze the given problem and identify what type of calculation is needed. Be concise."),
            ]

            response = llm.invoke(
                messages,
                config={
                    "callbacks": [handler],
                    **MetadataBuilder.for_langchain_config(
                        telemetry_context,
                        node_name="analyzer"
                    )
                }
            )

            analysis = response.content
            print(f"     Analysis: {analysis}")

            return {
                **state,
                "messages": state["messages"] + [response],
                "current_step": "analyzed",
                "analysis_result": analysis
            }

        def solver_node(state: GraphState) -> GraphState:
            print("   - Running solver_node...")
            messages = state["messages"] + [
                SystemMessage(content=f"Based on this analysis: '{state['analysis_result']}', now solve the mathematical problem step by step. Show your work."),
            ]

            response = llm.invoke(
                messages,
                config={
                    "callbacks": [handler],
                    **MetadataBuilder.for_langchain_config(
                        telemetry_context,
                        node_name="solver"
                    )
                }
            )

            solution = response.content
            print(f"     Solution: {solution}")

            return {
                **state,
                "messages": state["messages"] + [response],
                "current_step": "solved",
                "final_answer": solution
            }

        def validator_node(state: GraphState) -> GraphState:
            print("   - Running validator_node...")
            messages = state["messages"] + [
                SystemMessage(content=f"Review this solution: '{state['final_answer']}'. Provide a final, clear, and concise answer. If there are any errors, correct them."),
            ]

            response = llm.invoke(
                messages,
                config={
                    "callbacks": [handler],
                    **MetadataBuilder.for_langchain_config(
                        telemetry_context,
                        node_name="validator"
                    )
                }
            )

            final_result = response.content
            print(f"     Final Result: {final_result}")

            return {
                **state,
                "messages": state["messages"] + [response],
                "current_step": "validated",
                "final_answer": final_result
            }

        # Build the graph
        workflow = StateGraph(GraphState)
        workflow.add_node("analyzer", analyzer_node)
        workflow.add_node("solver", solver_node)
        workflow.add_node("validator", validator_node)
        workflow.set_entry_point("analyzer")
        workflow.add_edge("analyzer", "solver")
        workflow.add_edge("solver", "validator")
        workflow.set_finish_point("validator")
        app = workflow.compile()

        problems = [
            "What is 15 * 23?",
            "If I have 144 apples and want to divide them equally among 12 people, how many apples does each person get?",
            "Calculate the area of a rectangle with length 8 meters and width 5 meters."
        ]

        for i, problem in enumerate(problems, 1):
            print(f"\n--- LangGraph Test {i}: {problem} ---")

            initial_state = {
                "messages": [HumanMessage(content=problem)],
                "current_step": "start",
                "analysis_result": "",
                "final_answer": ""
            }

            try:
                final_state = app.invoke(
                    initial_state,
                    config={
                        "callbacks": [handler],
                        **MetadataBuilder.for_langchain_config(
                            telemetry_context,
                            test_type="langgraph_workflow",
                            problem_number=i
                        )
                    }
                )

                print("\n✓ Graph execution completed")
                print(f"Final Answer: {final_state['final_answer']}")
                llm_messages = [msg for msg in final_state['messages'] if hasattr(msg, 'content')]
                print(f"Total LLM calls in workflow: {len(llm_messages) - 1}")

            except Exception as e:
                print(f"✗ Graph execution failed: {e}")
                import traceback
                traceback.print_exc()

        print("LangGraph multi-node workflow test completed")
        print("✓ Each node should have generated separate cost entries in Langfuse")
        print(f"✓ Check session '{telemetry_context.session_id}' for detailed cost breakdown per node")
        print()

