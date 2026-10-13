
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "src" / "recommend_context.py"

st.set_page_config(
    page_title="ContextTune",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #10101c 0%, #19152b 55%, #111827 100%);
    }
    .block-container {
        max-width: 1050px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, p, label {
        color: #f4f0ff;
    }
    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #b8b3d1;
        font-size: 1.1rem;
    }
    .eyebrow {
        color: #c4b5fd;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    div.stButton > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        padding: 0.8rem;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        color: white;
        font-weight: 700;
        font-size: 1rem;
    }
    div.stButton > button:hover {
        border: 1px solid #ddd6fe;
        color: white;
    }
    div[data-testid="stTextArea"] textarea {
        background: #201b32;
        color: #ffffff;
        border: 1px solid #51436d;
        border-radius: 12px;
    }
    .stage {
        background: #211b33;
        border: 1px solid #403452;
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        min-height: 115px;
    }
    .stage-label {
        color: #c4b5fd;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stage-value {
        color: white;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Your mood. Your music. Your journey.</div>
        <h1>ContextTune 🎧</h1>
        <p>Discover music that moves the way you feel.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### What do you want your soundtrack to feel like?")
st.caption(
    "Describe a mood, a memory, a scenario, or an entire emotional journey."
)

with st.form("recommendation_form"):
    prompt = st.text_area(
        "Describe your vibe",
        placeholder=(
            "I want to feel like I'm on a rollercoaster, "
            "starting with anticipation and ending in euphoria..."
        ),
        height=120,
        max_chars=1000,
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button(
        "✨ Find my soundtrack",
        use_container_width=True,
    )

if submitted:
    if not prompt.strip():
        st.warning("Describe the kind of music experience you're looking for.")
    elif not SCRIPT.exists():
        st.error(f"Could not find the recommendation script: {SCRIPT}")
    else:
        try:
            with st.spinner(
                "Interpreting your vibe and finding matching tracks..."
            ):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT)],
                    input=prompt.strip() + "\n",
                    capture_output=True,
                    text=True,
                    cwd=str(ROOT),
                    timeout=240,
                    encoding="utf-8",
                    errors="replace",
                )

            output = result.stdout.strip()
            errors = result.stderr.strip()

            if result.returncode != 0:
                st.error("ContextTune couldn't complete the recommendation.")
                if output:
                    st.text(output[-12000:])
                if errors:
                    with st.expander("Technical details"):
                        st.code(errors[-6000:])
            elif not output:
                st.error("The recommendation engine returned no output.")
                if errors:
                    with st.expander("Technical details"):
                        st.code(errors[-6000:])
            else:
                st.session_state["contexttune_output"] = output
                st.session_state["contexttune_prompt"] = prompt.strip()

        except subprocess.TimeoutExpired:
            st.error(
                "The recommendation took too long. "
                "Please try again."
            )
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")

if "contexttune_output" in st.session_state:
    output = st.session_state["contexttune_output"]

    st.divider()
    st.markdown("## 🎶 Your soundtrack")
    st.caption(
        f"Your request: {st.session_state['contexttune_prompt']}"
    )

    vibe_match = re.search(
        r"Your vibe:\s*(.*?)(?=\n\s*Emotional journey:|\Z)",
        output,
        re.DOTALL,
    )

    if vibe_match:
        st.markdown("### ✨ Your vibe, interpreted")
        st.write(vibe_match.group(1).strip())

    journey_match = re.search(
        r"Emotional journey:\s*(.*?)(?=\n\s*Finding matching tracks|\Z)",
        output,
        re.DOTALL,
    )

    if journey_match:
        journey_text = journey_match.group(1).strip()
        values = re.findall(
            r"Stage\s*(\d+):\s*valence\s*=\s*"
            r"([-+]?\d*\.?\d+)\s*,\s*arousal\s*=\s*"
            r"([-+]?\d*\.?\d+)",
            journey_text,
            re.IGNORECASE,
        )

        if values:
            st.markdown("### 🌊 Your emotional journey")
            labels = ["Beginning", "Middle", "Ending"]
            columns = st.columns(len(values))

            for i, (stage, valence, arousal) in enumerate(values):
                label = labels[i] if i < len(labels) else f"Stage {stage}"
                with columns[i]:
                    st.markdown(
                        f"""
                        <div class="stage">
                            <div class="stage-label">{label}</div>
                            <div class="stage-value">
                                Valence: {float(valence):.2f}
                            </div>
                            <div style="color:#c4b3e9;">
                                Energy: {float(arousal):.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("### 🎧 Recommended tracks")

    sections = re.split(
        r"(?m)^\s*(\d+)\.\s+",
        output,
    )

    rendered_tracks = 0

    if len(sections) > 1:
        for i in range(1, len(sections) - 1, 2):
            rank = sections[i].strip()
            content = sections[i + 1].strip()

            lines = content.splitlines()
            title_line = lines[0].strip() if lines else ""

            if not title_line or title_line.startswith("ContextTune encountered"):
                continue

            title_artist = title_line.split(" — ", 1)
            title = title_artist[0].strip()
            artist = (
                title_artist[1].strip()
                if len(title_artist) > 1
                else "Unknown artist"
            )

            genre_match = re.search(
                r"(?m)^\s*Genre:\s*(.+)$", content
            )
            spotify_match = re.search(
                r"(?m)^\s*Spotify:\s*(https?://\S+)", content
            )
            youtube_match = re.search(
                r"(?m)^\s*YouTube:\s*(https?://\S+)", content
            )
            overall_match = re.search(
                r"Overall trajectory match score:\s*([\d.]+)",
                content,
            )

            with st.container(border=True):
                left, right = st.columns([5, 1])

                with left:
                    st.markdown(f"### {title}")
                    st.write(f"🎤 {artist}")

                    if genre_match:
                        st.caption(f"Genre · {genre_match.group(1).strip()}")

                with right:
                    st.markdown(f"#### #{rank}")

                    if overall_match:
                        st.metric(
                            "Fit score",
                            f"{float(overall_match.group(1)):.3f}",
                            help="Lower scores indicate a closer model-estimated trajectory match.",
                        )

                stage_matches = re.findall(
                    r"(?m)^\s*(Beginning|Middle|Ending):\s*(.+)$",
                    content,
                )

                if stage_matches:
                    stage_columns = st.columns(3)

                    for j, (stage, description) in enumerate(stage_matches):
                        with stage_columns[j]:
                            st.caption(stage)
                            st.write(description.strip())

                links = []
                if spotify_match:
                    links.append(
                        f"[▶ Search on Spotify]({spotify_match.group(1)})"
                    )
                if youtube_match:
                    links.append(
                        f"[▶ Search on YouTube]({youtube_match.group(1)})"
                    )

                if links:
                    st.markdown(" · ".join(links))

            rendered_tracks += 1

    if rendered_tracks == 0:
        st.warning(
            "The recommendation output could not be separated into "
            "individual tracks. The complete output is available below."
        )
        st.text(output)

    with st.expander("How ContextTune works"):
        st.markdown(
            """
            1. Gemini interprets your natural-language request.
            2. A trained Random Forest model estimates song emotions over time.
            3. The recommender compares each song's emotional trajectory
               with your requested journey.
            4. Tracks with lower trajectory distances rank closer to the target.

            Emotional interpretations are estimates, not universal truths.
            Spotify and YouTube links open search results rather than
            guaranteeing a specific recording is available.
            """
        )

st.markdown("---")
st.caption(
    "ContextTune · Emotion-aware music discovery · "
    "Built with Python, machine learning, and Gemini"
)
