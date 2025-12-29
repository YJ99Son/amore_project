import os
import sys

# Try importing LangChain components
try:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    print("[!] LangChain is not installed. Please run: pip install langchain langchain-community")
    sys.exit(1)

class LLMWrapper:
    """
    Unified interface for LLM interaction.
    Primary Strategy: Use Local Ollama instance via LangChain.
    """
    def __init__(self, model_name="llama3:8b", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._setup_ollama()

    def _setup_ollama(self):
        """
        Sets up the Ollama Chat Model.
        Advantage: Very lightweight in Python (offloads inference to Ollama app).
        """
        print(f"[*] Initializing local LLM: {self.model_name} via Ollama...")
        try:
            # check if ollama is running (simple check)
            # In a real app we might ping http://localhost:11434
            return ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url="http://localhost:11434" # Default Ollama URL
            )
        except Exception as e:
            print(f"[!] Error connecting to Ollama: {e}")
            print("    Please ensure Ollama is running ('ollama serve')")
            raise e

    def _setup_huggingface_fallback(self):
        """
        ALTERNATIVE: If not using Ollama, we load weights directly using HuggingFace.
        Disadvantage:
        - Requires loading 10GB+ weights into Python memory (VRAM/RAM).
        - Slower startup time.
        - Harder to manage 'streaming'.
        """
        # from langchain_huggingface import HuggingFacePipeline
        # return HuggingFacePipeline.from_model_id(
        #     model_id="beomi/Llama-3-Open-Ko-8B",
        #     task="text-generation",
        #     ...
        # )
        pass

    def generate(self, prompt_text, system_role="You are a helpful assistant."):
        """
        Simple generation method.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_role),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"input": prompt_text})
            return response
        except Exception as e:
            if "Connection refused" in str(e):
                return "[!] Error: Ollama is not running. Please launch Ollama first."
            return f"[!] Generation Error: {e}"

if __name__ == "__main__":
    # Test the wrapper
    print("--- Testing LLM Wrapper ---")
    
    # NOTE: You must have 'llama3' pulled in Ollama. 
    # Run: `ollama pull llama3` in terminal if not present.
    try:
        # Trying a smaller model or standard one likely to exist
        bot = LLMWrapper(model_name="llama3", temperature=0.7)
        
        print("\nSending Test Prompt: '안녕하세요! 아모레퍼시픽 마케팅 에이전트입니다.'")
        res = bot.generate("안녕하세요! 자기소개 부탁드립니다.", system_role="You are a polite Korean marketing assistant.")
        
        print("\n[Response]:")
        print(res)
        
    except Exception as e:
        print(f"\n[!] Test Failed: {e}")
