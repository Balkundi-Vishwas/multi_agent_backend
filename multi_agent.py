from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Callable, TypedDict, Literal
from langchain_core.language_models.chat_models import BaseChatModel
# from flask_socketio import SocketIO, emit
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from prompt_loader import PromptLoader
from chatbot import Chatbot
from child_agent1 import ChildAgent1
from config import Config
from logging import getLogger

logger = getLogger(__name__)

class Router(TypedDict):
    """Type definition for router response."""
    next: Literal["child_bot1", "FINISH"]

class AgentState(TypedDict):
    """State definition for the agent system."""
    query: str
    refined_query: str
    next: str
    complete_response: str

class MultiAgent:
    """Main agent orchestrator managing the routing and processing of queries."""

    def __init__(self) -> None:
        """Initialize the multi-agent system."""
        try:
            self.prompt_loader = PromptLoader()
            prompt_paths = self.prompt_loader.get_prompt_paths()
            self.master_prompt = self.prompt_loader.parse_master_prompts(
                self.prompt_loader.load_prompt_file(prompt_paths['master_agent'])
            )
            self.refiner_prompt = self.prompt_loader.load_prompt_file(
                prompt_paths['refiner_agent']
            )
            self.llm = self._initialize_llm()
            self.app = Flask(__name__)
            CORS(self.app)
            # self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self.chatbot = Chatbot()
            self.child_agent1 = ChildAgent1(self.chatbot)
            self.graph = self._build_graph()
            self.initial_query_state = {
                "query": "",
                "refined_query": "",
                "next": START,
                "complete_response": ""
            }
            print("main agent initlized")
            self._register_routes()
        except Exception as e:
            logger.error(f"Failed to initialize MultiAgent: {str(e)}")
            raise

    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the language model."""
        config = Config.MODELS
        try:
            return AzureChatOpenAI(
                azure_deployment=config["models"][0]["deployment"],
                api_version=config["models"][0]["version"],
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=Config.AGENT["max_retries"],
                azure_endpoint=Config.AZURE_OPENAI["azure_endpoint"],
                api_key=Config.AZURE_OPENAI["api_key"],
                streaming=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _register_routes(self) -> None:
        """Register socket event handlers."""
        # @self.socketio.on('data')
        # def process_query(data) -> None:
        #     try:
        #         current_state = self.initial_query_state.copy()
        #         current_state["query"] = data.get("question", "").strip()
        #         for state in self.graph.stream(current_state):
        #             pass
        #     except Exception as e:
        #         logger.error(f"Error processing query: {str(e)}")
        #         emit('data', {'char': "An error occurred while processing your query."})
        #         emit('stream_complete')

        @self.app.route('/api/query', methods = ['POST'])
        def process_query():
            data = request.json
            query = data.get("query")

            if not query:
                return jsonify({"error": "No query provided"}), 400
            
            try:
                current_state = self.initial_query_state.copy()
                current_state["query"] = query

                final_state = None
                for state in self.graph.stream(current_state):
                    final_state = state

                print(final_state)
                # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                if final_state and 'child_bot1' in final_state and 'complete_response' in final_state['child_bot1']:
                    response = final_state['child_bot1']['complete_response']
                    return jsonify({
                        "message": "Query processed successfully",
                        "response": response
                    })
                else:
                    return jsonify({"error": "No response generated"}), 500
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500 


        @self.app.route('/api/models', methods=['GET'])
        def get_models():
            config = Config.MODELS
            return jsonify({"models": config["models"]})
        
        @self.app.route("/api/update-model", methods=["POST"])
        def update_model():
            data = request.json
            model_name = data.get("name")
            deployment = data.get("deployment")
            version = data.get("version")

            if not model_name or not deployment or not version:
                return jsonify({"error": "Missing model details"}), 400

            self.chatbot._initialize_llm(deployment=deployment,version=version)
            return jsonify({
                "message": "Model reinitialized successfully",
                "selected_model": model_name
            })
  
        @self.app.route('/api/refresh', methods=["POST"])
        def handle_refresh():
            try:
                print("reset memory entered into multi_agent")
                self.chatbot.reset_memory()
                return jsonify({
                "message": "memory cleared and Model reinitialized successfully",
                "selected_model": "gpt-4o-mini"
            })
            except Exception as e:
                logger.error(f"Error handling refresh: {str(e)}")
                return jsonify({
                "message": "error in clearing memory or reinitilizing the model"
            })

    def refiner_agent(self, state: AgentState) -> AgentState:
        """Refine the original query."""
        try:
            current_query = state["query"]
            format_template = "Original query: '{query}'"
            refined_query = self.llm.invoke(self.refiner_prompt + "\n" + format_template.format(query=current_query))
            print(state)
            return {"refined_query": refined_query.content}
        except Exception as e:
            logger.error(f"Error in refiner agent: {str(e)}")
            return state

    def _create_supervisor_agent(self) -> Callable:
        """Create the supervisor agent for routing."""
        def supervisor_node(state: AgentState) -> AgentState:
            try:
                print("********************************************************")
                routing_prompt = self.master_prompt.format(
                    query=state['refined_query']
                )
                print(state['refined_query'])
                routing_response = self.llm.with_structured_output(Router).invoke(routing_prompt)
                print(routing_response)
                print("***********************************************************")
                return {"next": routing_response["next"]}
            except Exception as e:
                logger.error(f"Error in supervisor agent: {str(e)}")
                return {"next": "child_bot1"}  # Default fallback
        return supervisor_node

    def _build_graph(self) -> StateGraph:
        """Build the processing graph."""
        try:
            graph_builder = StateGraph(AgentState)
            
            graph_builder.add_node("refiner", self.refiner_agent)
            graph_builder.add_node("child_bot1", self.child_agent1.process_query)
            graph_builder.add_node("supervisor", self._create_supervisor_agent())

            graph_builder.add_edge(START, "refiner")
            graph_builder.add_edge("refiner", "supervisor")
            graph_builder.add_edge("child_bot1", END)
            
            graph_builder.add_conditional_edges(
                "supervisor",
                lambda x: x["next"],
                {
                    "child_bot1": "child_bot1",
                }
            )
            
            return graph_builder.compile()
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            raise

    def run(self, port: int = 5001, debug: bool = True) -> None:
        """Run the multi-agent system."""
        try:
            app = self.app
            app.run(
                port=port,
                debug=False
            )
        except Exception as e:
            logger.error(f"Error running server: {str(e)}")
            raise

def main():
    try:
        app = MultiAgent()
        app.run()
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        raise

if __name__ == '__main__':
    main()