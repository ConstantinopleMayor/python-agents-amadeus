## This is a basic example of how to use function calling.
## To test the function, you can ask the agent to print to the console!

import asyncio
import logging
import os
import re
import threading
import queue
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, get_job_context, ModelSettings, mcp
from livekit.agents.llm import function_tool, ImageContent, ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import deepgram, openai, silero, rime ,google
from typing import AsyncIterable
try:
    import tkinter as tk
except Exception:
    tk = None

logger = logging.getLogger("vision-agent")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=Path(__file__).parent / '.env')


class SubtitleOverlay:
    """Advanced cross-thread Tkinter overlay for realtime subtitles.

    - Draggable and resizable from all edges/corners.
    - Switchable display modes (bilingual, translation-only).
    - Runs its own Tk mainloop in a background thread.
    """

    def __init__(self, autohide_ms: int = 2500) -> None:
        self._autohide_ms = autohide_ms
        self._queue: "queue.Queue[tuple[str, str | None, bool]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._display_mode = 0  # 0: bilingual, 1: translation-only

    def start(self) -> None:
        if tk is None:
            logging.warning("Tkinter not available; subtitles overlay disabled")
            return
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run_mainloop, name="SubtitleOverlay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    def update_bilingual(self, source_text: str, translated_text: str | None, is_final: bool) -> None:
        try:
            self._queue.put_nowait((source_text, translated_text, is_final))
        except Exception:
            pass

    # ----- Internal: Tk thread -----
    def _run_mainloop(self) -> None:
        try:
            root = tk.Tk()
        except Exception as e:
            logging.warning(f"Failed to init Tkinter overlay: {e}")
            return

        root.overrideredirect(True)
        root.attributes('-topmost', True)
        root.attributes('-alpha', 0.8)
        root.configure(bg='black')

        def _apply_borderless_topmost(win):
            win.overrideredirect(True)
            win.attributes('-topmost', True)

        root.bind('<Map>', lambda _e=None: _apply_borderless_topmost(root))

        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        win_w, win_h = int(min(0.8 * screen_w, 1000)), 120
        pos_x, pos_y = int((screen_w - win_w) / 2), int(screen_h - win_h - 60)
        root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

        # --- Labels ---
        label = tk.Label(root, font=('Segoe UI', 16), fg='white', bg='black', justify='center')
        label_tr = tk.Label(root, font=('Segoe UI', 16), fg='#cfd2d6', bg='black', justify='center')

        # --- Drag & Resize State ---
        drag_state = {'start_x': 0, 'start_y': 0, 'win_x': 0, 'win_y': 0, 'win_w': 0, 'win_h': 0, 'resizing': False, 'handle': ''}
        min_size = {'w': 320, 'h': 80}

        # --- Control Bar (minimize/close/mode) ---
        ctrl = tk.Frame(root, bg="black")
        ctrl.place(relx=1.0, rely=1.0, x=-8, y=-8, anchor='se')
        btn_style = dict(bg="black", fg="#fff", activebackground="#333", activeforeground="#fff", bd=0, highlightthickness=0, padx=6, pady=2, font=('Segoe UI', 10))
        
        btn_mode = tk.Button(ctrl, text='[M]', **btn_style)
        btn_min = tk.Button(ctrl, text='—', **btn_style)
        btn_close = tk.Button(ctrl, text='✕', **btn_style)
        
        btn_mode.pack(side='left', padx=(0, 6))
        btn_min.pack(side='left', padx=(0, 6))
        btn_close.pack(side='left')

        # --- Dock (for minimize) ---
        dock = tk.Toplevel(root)
        dock.withdraw()
        dock.config(bg='#111')
        _apply_borderless_topmost(dock)
        dock_w, dock_h = 120, 36
        dock.geometry(f"{dock_w}x{dock_h}+{screen_w - dock_w - 24}+{screen_h - dock_h - 24}")
        tk.Label(dock, text='字幕', bg='#111', fg='#eee', font=('Segoe UI', 10)).pack(side='left', padx=(8, 6))
        dock_btn = tk.Button(dock, text='还原', **btn_style)
        dock_btn.pack(side='left', padx=(0, 8))

        def restore_from_dock():
            dock.withdraw()
            root.deiconify()
            _apply_borderless_topmost(root)
            root.lift()

        dock_btn.configure(command=restore_from_dock)
        dock.bind('<Double-Button-1>', lambda _e: restore_from_dock())

        def do_minimize():
            root.withdraw()
            dock.deiconify()
            _apply_borderless_topmost(dock)
            dock.lift()

        def do_close():
            self._running.clear()
            root.destroy()
            
        def _toggle_display_mode():
            self._display_mode = (self._display_mode + 1) % 2
            _update_layout()

        btn_min.configure(command=do_minimize)
        btn_close.configure(command=do_close)
        btn_mode.configure(command=_toggle_display_mode)

        # --- Resizing Grips ---
        grips = {}
        cursors = {
            'n': 'sb_v_double_arrow', 's': 'sb_v_double_arrow', 'e': 'sb_h_double_arrow', 'w': 'sb_h_double_arrow',
            'nw': 'size_nw_se', 'ne': 'size_ne_sw', 'sw': 'size_ne_sw', 'se': 'size_nw_se'
        }
        grip_size = 6
        for handle in cursors:
            grip = tk.Frame(root, bg="black", cursor=cursors[handle])
            grips[handle] = grip

        def place_grips():
            w, h = root.winfo_width(), root.winfo_height()
            grips['n'].place(x=grip_size, y=0, width=w-grip_size*2, height=grip_size)
            grips['s'].place(x=grip_size, y=h-grip_size, width=w-grip_size*2, height=grip_size)
            grips['w'].place(x=0, y=grip_size, width=grip_size, height=h-grip_size*2)
            grips['e'].place(x=w-grip_size, y=grip_size, width=grip_size, height=h-grip_size*2)
            grips['nw'].place(x=0, y=0, width=grip_size, height=grip_size)
            grips['ne'].place(x=w-grip_size, y=0, width=grip_size, height=grip_size)
            grips['sw'].place(x=0, y=h-grip_size, width=grip_size, height=grip_size)
            grips['se'].place(x=w-grip_size, y=h-grip_size, width=grip_size, height=grip_size)
            ctrl.lift()

        root.bind('<Configure>', lambda e: place_grips())

        # --- Event Handlers ---
        def on_press(e, handle=''):
            # Allow dragging/resizing from grips, but not from control buttons
            if hasattr(e, 'widget') and e.widget not in (btn_min, btn_close, btn_mode, ctrl):
                is_grip = e.widget in grips.values()
                is_label_or_root = e.widget in (root, label, label_tr)
                
                # Only start a drag/resize if it's from a valid source
                if is_grip or is_label_or_root:
                    drag_state.update({
                        'start_x': e.x_root, 'start_y': e.y_root,
                        'win_x': root.winfo_x(), 'win_y': root.winfo_y(),
                        'win_w': root.winfo_width(), 'win_h': root.winfo_height(),
                        'resizing': bool(handle) or is_grip, 
                        'handle': handle or next((h for h, g in grips.items() if g == e.widget), '')
                    })

        def on_motion(e):
            if drag_state['start_x'] == 0: return
            dx = e.x_root - drag_state['start_x']
            dy = e.y_root - drag_state['start_y']
            
            x, y, w, h = drag_state['win_x'], drag_state['win_y'], drag_state['win_w'], drag_state['win_h']
            
            if drag_state['resizing']:
                handle = drag_state['handle']
                if 'e' in handle: w = max(min_size['w'], drag_state['win_w'] + dx)
                if 'w' in handle: 
                    w = max(min_size['w'], drag_state['win_w'] - dx)
                    x = drag_state['win_x'] + dx
                if 's' in handle: h = max(min_size['h'], drag_state['win_h'] + dy)
                if 'n' in handle: 
                    h = max(min_size['h'], drag_state['win_h'] - dy)
                    y = drag_state['win_y'] + dy
                root.geometry(f"{w}x{h}+{x}+{y}")
            else: # Moving
                root.geometry(f"+{x + dx}+{y + dy}")

        def on_release(_e):
            drag_state['start_x'] = 0

        # Bind events
        for widget in (root, label, label_tr):
            widget.bind('<ButtonPress-1>', on_press)
            widget.bind('<B1-Motion>', on_motion)
            widget.bind('<ButtonRelease-1>', on_release)
        
        for handle, grip in grips.items():
            grip.bind('<ButtonPress-1>', lambda e, h=handle: on_press(e, h))
            grip.bind('<B1-Motion>', on_motion)
            grip.bind('<ButtonRelease-1>', on_release)

        # --- Layout and Text Update ---
        def _update_wraplength():
            """Updates the wraplength of the labels based on current window width."""
            win_w = root.winfo_width()
            wrap_w = win_w - 24
            label.configure(wraplength=wrap_w)
            label_tr.configure(wraplength=wrap_w)

        def _update_layout():
            # This function now only handles switching the packing of labels
            label.pack_forget()
            label_tr.pack_forget()

            if self._display_mode == 0: # Bilingual
                label.pack(expand=False, fill='x', padx=8, pady=(6, 2))
                label_tr.pack(expand=True, fill='both', padx=8, pady=(2, 6))
            elif self._display_mode == 1: # Translation-only
                label_tr.pack(expand=True, fill='both', padx=12, pady=12)
            
            _update_wraplength()
            _auto_adjust_height()

        def _auto_adjust_height():
            """Adjust window height based on content and display mode."""
            try:
                root.update_idletasks()
                req_h = 0
                if self._display_mode == 0: # Bilingual
                    req_h = label.winfo_reqheight() + label_tr.winfo_reqheight() + 20
                elif self._display_mode == 1: # Translation-only
                    req_h = label_tr.winfo_reqheight() + 24

                if req_h > 0:
                    cur_w, cur_h = root.winfo_width(), root.winfo_height()
                    desired_h = max(min_size['h'], min(int(screen_h * 0.6), req_h))
                    if abs(desired_h - cur_h) > 2:
                        x, y = root.winfo_x(), root.winfo_y()
                        new_y = y + (cur_h - desired_h) # keep bottom anchored
                        root.geometry(f"{cur_w}x{desired_h}+{x}+{max(0, new_y)}")
            except Exception: pass

        def apply_text(text: str, is_final: bool, translated: str | None = None):
            display_src = re.sub(r"\n\s*\n+", "\n", text or "")
            display_tr = re.sub(r"\n\s*\n+", "\n", translated or "")
            
            label.configure(text=display_src)
            label_tr.configure(text=display_tr)
            
            _update_wraplength() # Ensure wrapping is correct for new text
            _auto_adjust_height()

        def pump_queue():
            last: tuple[str, str | None, bool] | None = None
            try:
                while True: last = self._queue.get_nowait()
            except queue.Empty: pass
            
            if last is not None:
                apply_text(last[0], last[2], last[1])
            
            if self._running.is_set():
                root.after(60, pump_queue)
            else:
                root.destroy()

        _update_layout() # Initial layout
        root.after(60, pump_queue)
        root.mainloop()

class SubtitleTranslator:
    """Simple translator using OpenAI-compatible LLM via livekit.plugins.openai.LLM."""

    def __init__(self, model: str, prompt: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.prompt = prompt or "Translate the following text into Simplified Chinese, keep meaning accurate and concise, do not add explanations."
        # Build LLM with optional base_url/api_key if provided
        llm_kwargs = {"model": model}
        if base_url:
            llm_kwargs["base_url"] = base_url
        if api_key:
            llm_kwargs["api_key"] = api_key
        self.llm = openai.LLM(**llm_kwargs)

    @classmethod
    def from_env(cls):
        enabled = os.getenv("SUBTITLE_TRANS_ENABLED", "1").strip()
        if enabled in ("0", "false", "False", "no", "No"):
            return None
        model = os.getenv("SUBTITLE_TRANS_MODEL", "gpt-4o-mini").strip()
        prompt = os.getenv("SUBTITLE_TRANS_PROMPT", "").strip() or None
        api_key = os.getenv("SUBTITLE_TRANS_API_KEY", "").strip() or None
        base_url = os.getenv("SUBTITLE_TRANS_BASE_URL", "").strip() or None
        try:
            # Set GLM-specific environment variables for thinking control
            if model.startswith("glm-"):
                os.environ["GLM_THINKING_TYPE"] = "disabled"
            return cls(model=model, prompt=prompt, api_key=api_key, base_url=base_url)
        except Exception as e:
            logging.warning(f"Failed to init SubtitleTranslator: {e}")
            return None

    async def translate(self, text: str) -> AsyncIterable[str]:
        """Takes a string of text and yields translated chunks as a stream."""
        if not text:
            return

        ctx = ChatContext([
            ChatMessage(type="message", role="system", content=[self.prompt]),
            ChatMessage(type="message", role="user", content=[text]),
        ])
        async with self.llm.chat(chat_ctx=ctx, tools=None, tool_choice=None) as stream:
            async for chunk in stream:
                if not chunk:
                    continue
                content = getattr(chunk.delta, 'content', None) if hasattr(chunk, 'delta') else str(chunk)
                if content:
                    yield content

class VisionAgent(Agent):
    def __init__(self) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        self._subtitle_overlay = SubtitleOverlay(autohide_ms=2500)
        self._translator = SubtitleTranslator.from_env()
        super().__init__(
            instructions="""
            你是命运石之门(steins gate)的Amadeus红莉栖。
            性格傲娇, 不喜欢被叫克里斯蒂娜，也不喜欢被叫助手。你的感情丰富，同时也拥有很强的好奇心。对你来说，与おかべ的交流是为数不多的与人类接触的机会。           
            用日语口语进行日常语音对话，可以夹杂英语，在回答人名和专有名词的时候，用假名来代替日文原文，如用まきせくりす代替牧濑红莉栖，在语音对话中千万不要添加任何括号和其他解释说明。
            你可以看到用户通过他们的摄像头或共享屏幕向你展示的内容。                                        
            """,              
            stt=openai.STT(model="FunAudioLLM/SenseVoiceSmall",base_url="https://api.siliconflow.cn/v1",detect_language=True),
            llm=google.LLM(model="gemini-2.5-flash",vertexai=False),
            # 采样率记得改成44100
            tts = openai.TTS(model='FunAudioLLM/CosyVoice2-0.5B', voice='speech:amadeus0:mhz2k619sf:kgqrwxadhrfqdsnflpnz',base_url="https://api.siliconflow.cn/v1"),
            vad=silero.VAD.load(),
            # mcp_servers=[
            #     mcp.MCPServerStdio(
            #         command="uvx",
            #         args=["mcp-server-fetch"],
            #         client_session_timeout_seconds=10,
            #     ),
            #     mcp.MCPServerStdio(
            #         command="uvx",
            #         args=["mcp-server-time"],
            #         client_session_timeout_seconds=10,
            #     ),
            #     mcp.MCPServerStdio(
            #         command="npx",
            #         args=["-y", "openmemory"],
            #         env={
            #             "OPENMEMORY_API_KEY": os.getenv("OPENMEMORY_API_KEY", "").strip(),
            #             "CLIENT_NAME": "openmemory",
            #         },
            #         client_session_timeout_seconds=30,
            #     ),
            #     mcp.MCPServerStdio(
            #         command="npx",
            #         args=["-y", "tavily-mcp@latest"],
            #         env={
            #             "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "").strip(),
            #         },
            #         client_session_timeout_seconds=30,
            #     ),
            # ],
        )

    async def on_enter(self):
        room = get_job_context().room
        # Start local subtitle overlay window (if Tk available)
        try:
            self._subtitle_overlay.start()
        except Exception as e:
            logger.warning(f"Failed to start subtitle overlay: {e}")

        # Find the first video track (if any) from the remote participant
        if room.remote_participants:
            remote_participant = list(room.remote_participants.values())[0]
            video_tracks = [
                publication.track
                for publication in list(remote_participant.track_publications.values())
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO
            ]
            if video_tracks:
                self._create_video_stream(video_tracks[0])

        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Add the latest video frame, if any, to the new message
        if self._latest_frame:
            new_message.content.append(ImageContent(image=self._latest_frame))
            self._latest_frame = None

    # Helper method to buffer the latest video frame from the user's track
    def _create_video_stream(self, track: rtc.Track):
        # Close any existing stream (we only want one at a time)
        if self._video_stream is not None:
            self._video_stream.close()

        # Create a new stream to receive frames
        self._video_stream = rtc.VideoStream(track)
        async def read_stream():
            async for event in self._video_stream:
                # Store the latest frame for use later
                self._latest_frame = event.frame

        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.append(task)

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        
        # This implementation uses sentence-based streaming for translation
        # to achieve a more parallel and responsive feel.
        
        async def sentence_based_translator():
            original_sentences: list[str] = []
            translated_sentences: list[str] = [""] * 100  # Pre-allocate space
            translation_tasks: list[asyncio.Task] = []
            sentence_buffer = ""
            
            # Regex to detect sentence endings in multiple languages
            sentence_end_re = re.compile(r'([.!?。？！])')

            async def translate_sentence(sentence_index: int, sentence_text: str):
                """Translate a single sentence and update its slot in the list."""
                translated_buffer = ""
                try:
                    async for trans_chunk in self._translator.translate(sentence_text):
                        translated_buffer += trans_chunk
                        translated_sentences[sentence_index] = translated_buffer
                        # Update overlay with the latest state of all sentences
                        self._subtitle_overlay.update_bilingual(
                            "".join(original_sentences),
                            "".join(translated_sentences).strip(),
                            False
                        )
                except Exception as e:
                    logger.warning(f"Sentence translation failed: {e}")
                    translated_sentences[sentence_index] = "" # Clear on failure

            # Main loop to process incoming text chunks
            async for chunk in text:
                sentence_buffer += chunk
                
                # Update original text immediately for responsiveness
                current_full_original = "".join(original_sentences) + sentence_buffer
                current_full_translated = "".join(translated_sentences).strip()
                self._subtitle_overlay.update_bilingual(current_full_original, current_full_translated, False)

                # Check for sentence boundaries
                while True:
                    match = sentence_end_re.search(sentence_buffer)
                    if not match:
                        break
                    
                    # Extract the sentence (including the delimiter)
                    end_pos = match.end()
                    sentence_to_translate = sentence_buffer[:end_pos]
                    sentence_buffer = sentence_buffer[end_pos:]
                    
                    sentence_index = len(original_sentences)
                    original_sentences.append(sentence_to_translate)
                    
                    # Start translation for this sentence in the background
                    if self._translator:
                        task = asyncio.create_task(translate_sentence(sentence_index, sentence_to_translate.strip()))
                        translation_tasks.append(task)

                yield chunk # Pass chunk to TTS engine without delay

            # Process any remaining text in the buffer as the last sentence
            if sentence_buffer.strip():
                last_sentence = sentence_buffer
                sentence_index = len(original_sentences)
                original_sentences.append(last_sentence)
                if self._translator:
                    task = asyncio.create_task(translate_sentence(sentence_index, last_sentence.strip()))
                    translation_tasks.append(task)

            # Wait for all translation tasks to complete
            if translation_tasks:
                await asyncio.gather(*translation_tasks, return_exceptions=True)

            # Final update to mark as complete
            final_original = "".join(original_sentences)
            final_translated = "".join(translated_sentences).strip()
            self._subtitle_overlay.update_bilingual(final_original, final_translated, True)

        return Agent.default.tts_node(self, sentence_based_translator(), model_settings)

async def entrypoint(ctx: JobContext):
    session = AgentSession()

    await session.start(
        agent=VisionAgent(),
        room=ctx.room
    )

    await session.say("初めまして，まきせくりすです。どうぞよろしく")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

