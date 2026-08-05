#==============================================================================
# AI-Powered Data Exploration Assistant (upload-your-own-CSV, LangGraph edition)
#==============================================================================

#==============================================================================
# IMPORTS AND DEPENDENCIES
#==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# Helper: convert fig -> PNG bytes
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# Helper: extract text from Gemini response content
def extract_text_from_content(content):
    """Extract text from Gemini's response content which can be a string, list, or list of dicts."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif hasattr(item, 'text'):
                texts.append(item.text)
        return "".join(texts)
    return str(content)


#==============================================================================
# STREAMLIT APP CONFIGURATION
#==============================================================================
# Configure the Streamlit app with title and wide layout
st.set_page_config(page_title="AI-Powered Data Exploration Assistant", layout="wide")
st.title("📊 AI-Powered Data Exploration")


#==============================================================================
# GEMINI API KEY SETUP
#==============================================================================
# Load Gemini API key from environment variables for local development,
# falling back to Streamlit secrets for cloud deployment (Fix 9).
from dotenv import load_dotenv
import os
load_dotenv()


def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    # st.secrets can raise locally when no secrets.toml exists — don't crash.
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


api_key = _get_api_key()

if not api_key or not api_key.startswith("AIza"):
    st.warning("Please provide a valid Gemini API key in .env or st.secrets as GEMINI_API_KEY to continue.")
    st.stop()


#==============================================================================
# AI MODEL CONFIGURATION
#==============================================================================
# Select the AI model to use for all LLM operations
ai_model = "gemini-3.6-flash"

# Fix 4 + Fix 7: task-specific temperatures and explicit resilience settings.
TEMP_CODE = 0.2      # executor / repair / narrative code-gen: near-deterministic
TEMP_PROSE = 0.5     # planner / reflection / explainer
LLM_TIMEOUT = 90     # seconds per call
LLM_MAX_RETRIES = 2  # client-side retries for transient failures (429s, timeouts)
MAX_REPAIRS = 3      # max auto-repair attempts per code-gen stage


def _make_llm(temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=ai_model,
        temperature=temperature,
        google_api_key=api_key,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT,
    )


llm_plan = _make_llm(TEMP_PROSE)       # also used by the reflection node
llm_exec = _make_llm(TEMP_CODE)
llm_narrative = _make_llm(TEMP_CODE)
llm_explainer = _make_llm(TEMP_PROSE)


#==============================================================================
# DATA LOADING HELPERS
#==============================================================================
def _to_snake(name: str) -> str:
    """Normalize a column name to snake_case so generated code never contains
    spaces, hyphens, or other characters that cause string-wrapping issues."""
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)   # spaces, hyphens, slashes → underscore
    s = re.sub(r"[^\w]", "", s)         # drop anything else non-word
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Fix 8: cache CSV parsing — Streamlit reruns this whole script on every
# interaction (slider moves, button clicks) and previously re-parsed each time.
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(file_bytes))
    frame.columns = [_to_snake(c) for c in frame.columns]  # Fix 3
    return frame


def build_schema_str(dataframe: pd.DataFrame) -> str:
    """Rich schema: exact column names + dtypes + sample values (Fix 3).
    Injected into every agent prompt to prevent column name hallucination —
    critical here because users upload arbitrary CSVs."""
    lines = ["EXACT COLUMN NAMES (use these verbatim, no substitutions):"]
    for col, dtype in dataframe.dtypes.items():
        sample_vals = dataframe[col].dropna().head(3).tolist()
        sample_str = ", ".join(repr(v) for v in sample_vals)
        lines.append(f"  - {col!r}  ({dtype})  e.g. {sample_str}")
    return "\n".join(lines)


#==============================================================================
# USER INTERFACE FOR DATA INPUT
#==============================================================================
# File upload widget for CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    try:
        df = load_csv(uploaded_file.getvalue())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        df = None
    else:
        st.subheader("📄 Data Preview")
        st.caption("Column names are normalized to snake_case so the generated code stays robust.")
        # Allow user to control how many rows to preview
        n_rows = st.slider("Number of rows to preview:", 5, 100, 5, step=5)
        st.write(df.head(n_rows))
else:
    df = None

# Text area for optional dataset context (helps AI understand the data better)
data_context = st.text_area(
    "Optional: Provide context about your dataset (e.g., how was data collected, what columns mean, measurement scales, units, etc.)",
    ""
)

# Fix 5: placeholder= instead of a real default value — the example text was
# previously submitted verbatim as the analysis request on a bare click.
instructions = st.text_area(
    "Enter your question or instructions for data visualization:",
    placeholder="e.g. Create a scatterplot with engagement and job satisfaction",
)


#==============================================================================
# STATE MANAGEMENT FOR LANGGRAPH WORKFLOW
#==============================================================================
# Define the state structure that flows through the LangGraph workflow.
# Fix 1: exec and narrative errors are SEPARATE fields — the shared `error`
# field previously let narrative_node wipe a fatal execution error.
class VizState(TypedDict):
    schema: str                       # Dataset column names, dtypes, samples
    instructions: str                 # User's request/question
    data_context: str                 # Optional context about the dataset
    plan: Optional[str]               # Generated analysis plan
    code: Optional[str]               # Generated Python code for visualization
    explanation: Optional[str]        # Human-readable explanation of results
    df: Optional[object]              # The pandas DataFrame
    fig: Optional[object]             # The matplotlib Figure object
    exec_error: Optional[str]         # Visualization execution error
    narrative_error: Optional[str]    # Narrative generation error
    narrative_code: Optional[str]     # Code for generating narrative text
    narrative_text: Optional[str]     # Generated narrative with computed values
    retry_count_exec: int             # Number of code execution repair attempts
    retry_count_narrative: int        # Number of narrative repair attempts


#==============================================================================
# SAFE CODE EXECUTION HELPERS
#==============================================================================
def run_exec(code: str, df: pd.DataFrame) -> plt.Figure:
    """
    Execute visualization code and return the matplotlib Figure.

    Fix 6: line-anchored plt.show() stripping (the old blanket str.replace could
    corrupt code mid-line or inside strings); display() is handled by a no-op
    shim in the exec env instead of string surgery; plt.close("all") prevents
    figure state leaking between runs; a clear RuntimeError (instead of a
    cryptic KeyError) when `fig` is missing gives the repair agent a much more
    actionable error message.

    NOTE: exec of LLM-generated code is unsandboxed. Local single-user use only.
    """
    safe_code = re.sub(r"^[ \t]*plt\.show\(\)[ \t]*$", "", code, flags=re.MULTILINE)
    plt.close("all")
    exec_env = {
        "df": df, "sns": sns, "plt": plt, "pd": pd, "np": np,
        "display": lambda *a, **k: None,   # no-op shim
    }
    exec(safe_code, exec_env)
    if "fig" not in exec_env:
        raise RuntimeError(
            "Generated code ran but did not define a matplotlib Figure named `fig`."
        )
    return exec_env["fig"]


def run_narrative(code: str, df: pd.DataFrame) -> str:
    """
    Execute narrative generation code and return the narrative string.
    Same hardening as run_exec (Fix 6).
    """
    exec_env = {
        "df": df, "sns": sns, "plt": plt, "pd": pd, "np": np,
        "display": lambda *a, **k: None,   # no-op shim
    }
    exec(code, exec_env)
    if "narrative" not in exec_env:
        raise RuntimeError(
            "Generated code ran but did not define a string named `narrative`."
        )
    return exec_env["narrative"]


#==============================================================================
# WORKFLOW NODES - MAIN ANALYSIS PIPELINE
#==============================================================================

def planner_node(state: VizState) -> VizState:
    """
    Generate an initial analysis plan based on user request and dataset schema.
    This is the first step that creates a high-level strategy for the analysis.
    """
    with st.spinner("📝 Generating plan..."):
        plan_msg = llm_plan.invoke(f"""
            Dataset schema: 
            {state['schema']}

            Dataset context: 
            {state['data_context']}

            User request: 
            {state['instructions']}
            
            Task: Break down the steps for both data cleaning and wrangling (pandas) and visualization (seaborn + matplotlib) that would fulfill the user's request - for example, creating a data visualization, answering a question, providing insights, or helping test a hypothesis.
            Requirements:
            - Wrangling must operate on the input DataFrame df
            - There can be multiple plots or subplots, but they must all be contained in a single matplotlib Figure
            - Visualization must end with a matplotlib Figure object called fig
            - Provide a single, concise step plan that best achieves the user's request or answers their question.
            - Make the analysis as simple as possible while still being effective.
            - Do more complex analysis only if it clearly adds value or if you are explicitly asked to do so by the user.
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid distortion or chartjunk.
        """)
        state["plan"] = extract_text_from_content(plan_msg.content)
    return state

def reflection_node(state: VizState) -> VizState:
    """
    Review and improve the initial plan by checking for appropriateness and completeness.
    This quality control step helps ensure the plan aligns well with user needs.
    """
    with st.spinner("🪞 Reflecting on plan quality..."):
        reflection_msg = llm_plan.invoke(f"""
            Dataset schema:
            {state['schema']}

            Dataset context:
            {state['data_context']}

            User request:
            "{state['instructions']}"

            Initial analysis and visualization plan:
            ```
            {state['plan']}
            ```

            Task:
            Critically reflect on the quality and appropriateness of this plan.
            - Check whether the plan aligns well with the user's request.
            - Verify if the visualization types are appropriate for the dataset schema.
            - Check if variable selections make sense and are valid according to the schema.
            - Evaluate whether the plan is logically coherent and sufficient to answer the question.
            - Identify missing steps (e.g., grouping, aggregation, filtering, labeling, clarity).
            - If the plan is already strong, affirm that.
            - If improvements are needed, rewrite the plan to make it stronger.

            Output only the final, improved plan (no explanations).
        """)
        state["plan"] = extract_text_from_content(reflection_msg.content).strip()
    return state


#==============================================================================
# CODE GENERATION AND EXECUTION NODES
#==============================================================================

# Pydantic model to ensure structured output from LLM for code generation
class ExecCode(BaseModel):
    code: str

# LLM instance with structured output for reliable code generation
exec_llm = llm_exec.with_structured_output(ExecCode)

def executor_node(state: VizState) -> VizState:
    """
    Generate Python code based on the plan and execute it to create visualizations.
    Fix 3: the executor now receives the exact schema — previously it worked
    only from the plan's prose and freely hallucinated column names.
    """
    with st.spinner("⚙️ Generating and executing code..."):
        exec_plan = exec_llm.invoke(f"""
            Context plan:
            {state['plan']}

            CRITICAL — COLUMN NAMES:
            {state['schema']}
            You MUST use only the exact column names listed above. Do NOT invent, rename,
            or abbreviate any column name. If the plan mentions a column not in this list,
            use the closest exact match from the list above.

            The input DataFrame is named df and is already loaded.
            Write Python code that:
            - Performs necessary wrangling according to the plan
            - Produces the requested visualization
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid distortion or chartjunk.
            - Prefer simplicity and clarity over complexity
            - You may create multiple plots or subplots if it enhances the analysis and user's understanding, but they must all be contained in a single matplotlib Figure
            - Keep in mind that the generated plot should be well readable on the common computer screen (not too small, not too crowded)
            - Assign the final matplotlib Figure object to variable fig
            - Do NOT use `return` statements anywhere
            - Do NOT use `display()`, `print()`, or `plt.show()`
            - Only return runnable code
            - Don't forget to import any necessary libraries
            - Always end with `fig` defined as the final Figure object
        """)
        code = exec_plan.code
        state["code"] = code

        # Attempt to execute the generated code
        try:
            state["fig"] = run_exec(code, state["df"])
            state["exec_error"] = None
        except Exception as e:
            state["exec_error"] = str(e)
    return state

def repair_exec_node(state: VizState) -> VizState:
    """
    Attempt to fix code execution errors by generating corrected code.
    Fix 2: this node now self-loops (see graph edges) instead of routing back
    to blind from-scratch regeneration in the executor.
    Fix 3: the repairer now receives the schema, plan, and user request —
    previously it repaired code toward an intent it was never shown.
    """
    if not state["exec_error"]:
        return state

    state["retry_count_exec"] += 1
    if state["retry_count_exec"] > MAX_REPAIRS:  # defensive; edges enforce this
        return state

    with st.spinner(f"🔧 Repairing failed visualization code (attempt {state['retry_count_exec']}/{MAX_REPAIRS})..."):
        repair_msg = exec_llm.invoke(f"""
            User request:
            {state['instructions']}

            Analysis plan the code must fulfil:
            {state['plan']}

            CRITICAL — ONLY USE THESE EXACT COLUMN NAMES:
            {state['schema']}
            If the error mentions a column not found, replace it with the correct name
            from the list above. Do NOT invent new column names.

            The following visualization code failed with an error:
            ```
            {state['code']}
            ```
            Error message:
            {state['exec_error']}

            Please suggest corrected Python code that fixes this issue.
            Constraints:
            - Performs necessary wrangling according to the plan
            - Produces the requested visualization
            - You may create multiple plots or subplots if it enhances the analysis and user's understanding, but they must all be contained in a single matplotlib Figure
            - Prefer simplicity and clarity over complexity
            - Assign the final matplotlib Figure object to variable fig
            - Do NOT use `return` statements anywhere
            - Do NOT use `display()`, `print()`, or `plt.show()`
            - Only return runnable code
            - Don't forget to import any necessary libraries
            - Keep in mind that the generated plot should be well readable (not too small, not too crowded)
            - Always end with `fig` defined as the final Figure object
        """)
        repaired_code = repair_msg.code
        state["code"] = repaired_code

        # Attempt to execute the repaired code
        try:
            state["fig"] = run_exec(repaired_code, state["df"])
            state["exec_error"] = None
        except Exception as e:
            state["exec_error"] = str(e)
    return state


#==============================================================================
# NARRATIVE GENERATION NODES
#==============================================================================

# Pydantic model for structured narrative code output
class NarrativeCode(BaseModel):
    code: str

# LLM instance for narrative generation with structured output
narrative_llm = llm_narrative.with_structured_output(NarrativeCode)

def narrative_node(state: VizState) -> VizState:
    """
    Generate code that creates a narrative text with computed statistics.
    Fix 1: writes narrative_error — it previously overwrote the shared `error`
    field and could wipe a fatal execution error.
    """
    with st.spinner("📜 Generating narrative code..."):
        narrative_plan = narrative_llm.invoke(f"""
            User request: {state['instructions']}
            Dataset schema: {state['schema']}
            Dataset context: {state['data_context']}
            Context plan: {state['plan']}
            Visualization code:
            ```
            {state['code']}
            ```
            
            Task: Write Python code that generates a string variable named `narrative`.
            Requirements:
            - Only use the EXACT column names from the dataset schema above.
            - Use the input DataFrame df (already loaded).
            - Perform actual computations on df (mean, median, counts, correlations, SEM as relevant).
            - Explicitly insert computed values into the string (rounded to 2 decimals).
            - Always wrap the entire narrative text inside triple quotes (\"\"\" ... \"\"\").
            - Make sure both opening and closing triple quotes are present.
            - Assign the result to a variable named `narrative`.
            - Only return runnable Python code.
        """)

        state["narrative_code"] = narrative_plan.code

        # Execute the narrative generation code
        try:
            state["narrative_text"] = run_narrative(state["narrative_code"], state["df"])
            state["narrative_error"] = None
        except Exception as e:
            state["narrative_error"] = str(e)
    return state

def repair_narrative_node(state: VizState) -> VizState:
    """
    Repair failed narrative generation code with retry logic.
    Fix 2: self-loops via graph edges. Fix 3: now receives schema, plan, and
    user request.
    """
    if not state["narrative_error"]:
        return state

    state["retry_count_narrative"] += 1
    if state["retry_count_narrative"] > MAX_REPAIRS:  # defensive; edges enforce this
        return state

    with st.spinner(f"🔧 Repairing failed narrative code (attempt {state['retry_count_narrative']}/{MAX_REPAIRS})..."):
        repair_msg = narrative_llm.invoke(f"""
            User request: {state['instructions']}
            Analysis plan: {state['plan']}

            CRITICAL — ONLY USE THESE EXACT COLUMN NAMES:
            {state['schema']}

            The following narrative code failed with an error:
            ```
            {state['narrative_code']}
            ```
            Error message:
            {state['narrative_error']}

            Please suggest corrected Python code that fixes this issue.
            Constraints:
            - Only use the exact column names from the dataset schema above.
            - Use the input DataFrame df (already loaded).
            - Perform actual computations on df (mean, median, counts, correlations, SEM as relevant).
            - Explicitly insert computed values into the string (rounded to 2 decimals).
            - Always wrap the entire narrative text inside triple quotes (\"\"\" ... \"\"\").
            - Make sure both opening and closing triple quotes are present.
            - Assign the result to a variable named `narrative`.
            - Only return runnable Python code.
        """)
        repaired_code = repair_msg.code
        state["narrative_code"] = repaired_code

        # Attempt to execute the repaired narrative code
        try:
            state["narrative_text"] = run_narrative(repaired_code, state["df"])
            state["narrative_error"] = None
        except Exception as e:
            state["narrative_error"] = str(e)
    return state

def explainer_node(state: VizState) -> VizState:
    """
    Generate a human-readable explanation of the analysis results.
    Fix 1: instructed not to invent statistics when the narrative is missing
    (the narrative repair loop can exhaust its retries — that is non-fatal,
    but the explanation must not fabricate numbers to fill the gap).
    """
    with st.spinner("💬 Generating explanation..."):
        explain_msg = llm_explainer.invoke(f"""
            User request: {state['instructions']}
            Dataset context: {state['data_context']}
            Context plan:
            {state['plan']}

            Narrative string:
            {state['narrative_text'] or "(narrative could not be computed)"}

            Task: Create a narrative explanation of what the generated chart(s) show and how to interpret them,
            and provide specific insights revealed by the analysis for a non-technical audience.
            Constraints:
            - Make it concise and clear.
            - Do not output code. Write only text.
            - If the narrative string is empty or missing, base the explanation only on the plan
              and the general nature of the chart — do NOT invent specific numbers or statistics.
        """)
        state["explanation"] = extract_text_from_content(explain_msg.content).strip()
    return state


#==============================================================================
# LANGGRAPH WORKFLOW CONSTRUCTION
#==============================================================================
# Build the complete analysis workflow using LangGraph
workflow = StateGraph(VizState)

# Add all workflow nodes
workflow.add_node("planner", planner_node)
workflow.add_node("reflection", reflection_node) 
workflow.add_node("executor", executor_node)
workflow.add_node("repair_exec", repair_exec_node)
workflow.add_node("narrative", narrative_node)
workflow.add_node("repair_narrative", repair_narrative_node)
workflow.add_node("explainer", explainer_node)

# Define the workflow flow - linear progression with error handling branches
workflow.set_entry_point("planner")
workflow.add_edge("planner", "reflection")  
workflow.add_edge("reflection", "executor") 

# Conditional flow for code execution with error handling
workflow.add_conditional_edges(
    "executor",
    lambda state: "repair_exec" if state.get("exec_error") else "narrative"
)

# Fix 2: repair self-loops with accumulated error context (was: back to a blind
# from-scratch executor). Fix 1: when repairs are exhausted and there is still
# no figure, the graph TERMINATES — no narrative, no explanation for a chart
# that doesn't exist. The display layer surfaces exec_error to the user.
workflow.add_conditional_edges(
    "repair_exec",
    lambda state: (
        "narrative" if not state.get("exec_error")
        else ("repair_exec" if state["retry_count_exec"] < MAX_REPAIRS else END)
    )
)

# Conditional flow for narrative generation with error handling
workflow.add_conditional_edges(
    "narrative",
    lambda state: "repair_narrative" if state.get("narrative_error") else "explainer"
)

# Fix 2: narrative repair also self-loops. A narrative failure is NON-fatal:
# the chart exists, so we still explain it (without invented statistics —
# see explainer_node).
workflow.add_conditional_edges(
    "repair_narrative",
    lambda state: (
        "explainer" if not state.get("narrative_error")
        else ("repair_narrative" if state["retry_count_narrative"] < MAX_REPAIRS else "explainer")
    )
)

# Final step - always end with explanation
workflow.add_edge("explainer", END)

# Compile the workflow into an executable application
app = workflow.compile()


#==============================================================================
# MAIN EXECUTION AND RESULTS DISPLAY
#==============================================================================
# Execute the complete analysis pipeline when user clicks the button
if "viz_result" not in st.session_state:
    st.session_state.viz_result = None

# Fix 5: the button now renders whenever data is loaded; empty instructions get
# a warning instead of silently analyzing placeholder text.
if df is not None and st.button("Generate Insights"):
    if not instructions.strip():
        st.warning("Please enter a question or instructions before generating insights.")
    else:
        # Prepare the rich dataset schema for the AI models (Fix 3)
        schema_str = build_schema_str(df)

        # Initialize the workflow state with user inputs and empty results
        state: VizState = {
            "schema": schema_str,
            "instructions": instructions.strip(),
            "data_context": data_context if data_context else "",
            "df": df,
            "plan": None,
            "code": None,
            "fig": None,
            "explanation": None,
            "exec_error": None,
            "narrative_error": None,
            "narrative_code": None,
            "narrative_text": None,
            "retry_count_exec": 0,
            "retry_count_narrative": 0,
        }

        # Fix 7: an API failure that survives client retries surfaces as a
        # friendly error instead of crashing the session with a raw traceback.
        result = None
        try:
            result = app.invoke(state)
        except Exception as e:
            st.error(f"Pipeline failed: {type(e).__name__}: {e}")

        if result is not None:
            # Convert fig to PNG bytes if it exists, then close it (Fix 6 —
            # don't keep live Figure objects around between runs)
            if result.get("fig") is not None:
                result["fig_png"] = fig_to_png_bytes(result["fig"])
                plt.close(result["fig"])
                result["fig"] = None
            else:
                result["fig_png"] = None

            st.session_state.viz_result = result

    # DEBUG SECTIONS - Hidden for production UI but useful for development
    # Uncomment these sections to see intermediate workflow results
    # if result and result.get("plan"):
    #     with st.expander("📝 Plan"):
    #         st.code(result["plan"], language="markdown")

    # if result and result.get("code"):
    #     with st.expander("⚙️ Final Code"):
    #         st.code(result["code"], language="python")

    # if result and result.get("narrative_code"):
    #     with st.expander("📜 Narrative Code"):
    #         st.code(result["narrative_code"], language="python")

    # if result and result.get("narrative_text"):
    #     with st.expander("📖 Narrative Text"):
    #         st.write(result["narrative_text"])

# MAIN RESULTS DISPLAY
# ---- Show persisted results (if any) ----
if st.session_state.viz_result:
    result = st.session_state.viz_result

    if result.get("fig_png"):
        st.image(result["fig_png"], use_container_width=True)

    if result.get("explanation"):
        st.subheader("💡 Explanation")
        st.write(result["explanation"])

    # Fix 1: a terminal execution failure is shown loudly (previously it was
    # silently wiped and the user got an explanation of a nonexistent chart).
    if result.get("exec_error"):
        st.error(
            f"Couldn't produce the chart after {MAX_REPAIRS} repair attempts. "
            f"Last error: {result['exec_error']}"
        )
    elif result.get("narrative_error"):
        st.warning(
            "The statistical narrative couldn't be computed after "
            f"{MAX_REPAIRS} repair attempts (last error: {result['narrative_error']}). "
            "The explanation above is based on the plan and chart only."
        )
