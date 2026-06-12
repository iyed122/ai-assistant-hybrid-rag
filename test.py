from langchain_community.llms import Ollama
from ragas.llms import LangchainLLMWrapper

# 1. Create a custom class to bridge the version gap
class PatchedOllama(Ollama):
    def predict(self, text: str, **kwargs) -> str:
        # Route the old 'predict' call to the new 'invoke' method
        return self.invoke(text, **kwargs)

print("Initializing Patched Judge...")
# 2. Use the new patched class
ollama = PatchedOllama(model="qwen3:8b", base_url="http://localhost:11434")
llm = LangchainLLMWrapper(ollama)

try:
    res = llm.generate_text("Are you working now?")
    print(f"\nSUCCESS: {res}")
except Exception as e:
    print(f"\nFAIL: {e}")