from pathlib import Path
from typing import Optional, Dict, Any
from llama_cpp import Llama, GGML_TYPE_Q8_0
from chat_template import format_chat_prompt

MODEL_CODE = "4bm"

_llm_instance: Optional[Llama] = None
_states: Dict[str, Any] = {}

def find_gguf_file() -> Path:
    # Check current directory
    for path in Path(".").glob("*.gguf"):
        # Make sure it's not the mmproj file
        if "mmproj" not in path.name:
            return path
    # Check model/ directory
    model_dir: Path = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*.gguf"):
            if "mmproj" not in path.name:
                return path
    # Fallback default
    return Path("gemma-4-E4B-it-Q4_K_M.gguf")

def find_mmproj_file() -> Optional[Path]:
    for path in Path(".").glob("*mmproj*.gguf"):
        return path
    model_dir: Path = Path("model")
    if model_dir.exists():
        for path in model_dir.glob("*mmproj*.gguf"):
            return path
    return None

def get_llm() -> Llama:
    global _llm_instance
    if _llm_instance is None:
        model_path: Path = find_gguf_file()
        if not model_path.exists():
            raise FileNotFoundError(f"No GGUF model file found. Expected one in root or model/ directory.")
        
        mmproj_path = find_mmproj_file()
        chat_handler = None
        if mmproj_path:
            try:
                from llama_cpp.llama_chat_format import LlavaChatHandler
                print(f"[Model] Found vision projector file: {mmproj_path}", flush=True)
                chat_handler = LlavaChatHandler(clip_model_path=str(mmproj_path))
            except Exception as e:
                print(f"[Model] Warning: Failed to load LlavaChatHandler: {e}", flush=True)
        
        _llm_instance = Llama(
            model_path=str(model_path),
            n_threads=4,
            n_ctx=40960,
            flash_attn=True,
            type_k=GGML_TYPE_Q8_0,
            type_v=GGML_TYPE_Q8_0,
            chat_handler=chat_handler
        )
    return _llm_instance

def run_model_query(prompt: str, jid: Optional[str] = None, image_base64: Optional[str] = None) -> str:
    try:
        llm: Llama = get_llm()
        
        # Load KV cache state if it exists for this conversation
        if jid and jid in _states:
            llm.load_state(_states[jid])
            print(f"[Model] Restored KV cache state for JID: {jid}", flush=True)
            
        if image_base64 and getattr(llm, "chat_handler", None) is not None:
            print(f"[Model] Running vision query with image of size {len(image_base64)} characters", flush=True)
            if not image_base64.startswith("data:image"):
                image_base64 = f"data:image/jpeg;base64,{image_base64}"
            
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_base64}}
                        ]
                    }
                ],
                max_tokens=150
            )
            text_result: str = response["choices"][0]["message"]["content"]
        else:
            if image_base64:
                print(f"[Model] Text fallback mode: Received image of size {len(image_base64)} characters", flush=True)
                prompt = f"[User uploaded an image. Base64 length: {len(image_base64)}]\n{prompt}"
            
            formatted_prompt: str = format_chat_prompt(prompt)
            response = llm(
                formatted_prompt,
                max_tokens=150,
            )
            text_result: str = response["choices"][0]["text"]
            
        # Save updated KV cache state for this conversation
        if jid:
            _states[jid] = llm.save_state()
            print(f"[Model] Saved KV cache state for JID: {jid}", flush=True)
            
        return text_result
    except Exception as e:
        return f"Exception raised while running llama-cpp: {e}"