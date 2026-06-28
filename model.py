from pathlib import Path
from typing import Optional, Dict, Any
from llama_cpp import Llama
from chat_template import format_chat_prompt

_llm_instance: Optional[Llama] = None
_states: Dict[str, Any] = {}

def find_gguf_file() -> Path:
    # Check current directory
    for path in Path(".").glob("*.gguf"):
        return path
    # Check model/ directory
    model_dir: Path = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*.gguf"):
            return path
    # Fallback default
    return Path("gemma-4-E4B-it-Q4_K_M.gguf")

def get_llm() -> Llama:
    global _llm_instance
    if _llm_instance is None:
        model_path: Path = find_gguf_file()
        if not model_path.exists():
            raise FileNotFoundError(f"No GGUF model file found. Expected one in root or model/ directory.")
        
        _llm_instance = Llama(
            model_path=str(model_path),
            n_threads=4,
            n_ctx=2048,
            flash_attn=True,
        )
    return _llm_instance

def run_model_query(prompt: str, jid: Optional[str] = None) -> str:
    try:
        llm: Llama = get_llm()
        
        # Load KV cache state if it exists for this conversation
        if jid and jid in _states:
            llm.load_state(_states[jid])
            print(f"[Model] Restored KV cache state for JID: {jid}", flush=True)
            
        formatted_prompt: str = format_chat_prompt(prompt)
        response = llm(
            formatted_prompt,
            max_tokens=256,
        )
        
        # Save updated KV cache state for this conversation
        if jid:
            _states[jid] = llm.save_state()
            print(f"[Model] Saved KV cache state for JID: {jid}", flush=True)
            
        text_result: str = response["choices"][0]["text"]
        return text_result
    except Exception as e:
        return f"Exception raised while running llama-cpp: {e}"