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

import pickle
import threading
from pathlib import Path

# Create states directory
STATES_DIR = Path("states")
STATES_DIR.mkdir(parents=True, exist_ok=True)

_llm_lock = threading.Lock()

def run_model_query(prompt: str, jid: Optional[str] = None, image_base64: Optional[str] = None) -> str:
    with _llm_lock:
        try:
            llm: Llama = get_llm()
            
            if image_base64 and getattr(llm, "chat_handler", None) is not None:
                print(f"[Model] Running vision query with image of size {len(image_base64)} characters", flush=True)
                if not image_base64.startswith("data:image"):
                    image_base64 = f"data:image/jpeg;base64,{image_base64}"
                
                response_generator = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_base64}}
                            ]
                        }
                    ],
                    max_tokens=512,
                    stream=True
                )
                text_chunks = []
                print("[Model Vision] Generating: ", end="", flush=True)
                for chunk in response_generator:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        token_text = delta["content"]
                        print(token_text, end="", flush=True)
                        text_chunks.append(token_text)
                print("\n[Model Vision] Generation complete.", flush=True)
                text_result = "".join(text_chunks)
            else:
                if image_base64:
                    print(f"[Model] Text fallback mode: Received image of size {len(image_base64)} characters", flush=True)
                    prompt = f"[User uploaded an image. Base64 length: {len(image_base64)}]\n{prompt}"
                
                formatted_prompt: str = format_chat_prompt(prompt)
                new_tokens = llm.tokenize(formatted_prompt.encode("utf-8"))
                
                state_file = STATES_DIR / f"{jid}.state" if jid else None
                tokens_file = STATES_DIR / f"{jid}.tokens" if jid else None
                common_len = 0
                
                # 1. First, try loading the specific JID cache state
                if jid and state_file.exists() and tokens_file.exists():
                    try:
                        with open(tokens_file, "rb") as tf:
                            cached_tokens = pickle.load(tf)
                        
                        for t1, t2 in zip(cached_tokens, new_tokens):
                            if t1 != t2:
                                break
                            common_len += 1
                            
                        if common_len > 0:
                            llm.load_state(state_file.read_bytes())
                            print(f"[Model] Restored cache state for JID: {jid} (prefix matched: {common_len}/{len(cached_tokens)} tokens)", flush=True)
                    except Exception as cache_err:
                        print(f"[Model] Warning: Failed to load cache state for JID {jid}: {cache_err}", flush=True)
                        common_len = 0
                
                # 2. If JID cache missed, fall back to the pre-cached global prefix
                if common_len == 0:
                    global_state = STATES_DIR / "global_prefix.state"
                    global_tokens = STATES_DIR / "global_prefix.tokens"
                    if global_state.exists() and global_tokens.exists():
                        try:
                            with open(global_tokens, "rb") as tf:
                                cached_tokens = pickle.load(tf)
                            
                            for t1, t2 in zip(cached_tokens, new_tokens):
                                if t1 != t2:
                                    break
                                common_len += 1
                                
                            if common_len > 0:
                                llm.load_state(global_state.read_bytes())
                                print(f"[Model] Restored cache from global prefix cache (prefix matched: {common_len}/{len(cached_tokens)} tokens)", flush=True)
                        except Exception as glob_err:
                            print(f"[Model] Warning: Failed to load global prefix cache: {glob_err}", flush=True)
                            common_len = 0
                            
                if common_len == 0:
                    llm.reset()
                    print(f"[Model] Cache miss/fresh start for JID: {jid}", flush=True)
                
                print(f"[Model] Evaluating context & generating response...", flush=True)
                response_generator = llm(
                    formatted_prompt,
                    max_tokens=512,
                    stream=True
                )
                
                text_result_chunks = []
                print("[Model] Generating: ", end="", flush=True)
                for chunk in response_generator:
                    token_text = chunk["choices"][0]["text"]
                    print(token_text, end="", flush=True)
                    text_result_chunks.append(token_text)
                print("\n[Model] Generation complete.", flush=True)
                text_result = "".join(text_result_chunks)
                
                if jid:
                    try:
                        state_file.write_bytes(llm.save_state())
                        # Tokenize combined prompt and output to align cache end
                        full_evaluated_text = formatted_prompt + text_result
                        full_tokens = llm.tokenize(full_evaluated_text.encode("utf-8"))
                        with open(tokens_file, "wb") as tf:
                            pickle.dump(full_tokens, tf)
                        print(f"[Model] Saved updated cache state for JID: {jid} ({len(full_tokens)} tokens)", flush=True)
                    except Exception as save_err:
                        print(f"[Model] Warning: Failed to save cache state for JID {jid}: {save_err}", flush=True)
                
            return text_result
        except Exception as e:
            return f"Exception raised while running llama-cpp: {e}"