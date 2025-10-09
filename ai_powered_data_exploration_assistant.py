import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END


# Streamlit app configuration
st.set_page_config(page_title="AI-Powered Data Exploration Assistant", layout="wide")
st.title("📊 AI-Powered Data Exploration")

# Load OpenAI API key
# For local dev
from dotenv import load_dotenv
import os
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.warning("Please provide a valid OpenAI API key in .env as OPENAI_API_KEY to continue.")
    st.stop()

# For deployment (e.g., Streamlit Cloud)
# api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.warning("Please provide a valid OpenAI API key in st.secrets as OPENAI_API_KEY to continue.")
    st.stop()

# Select AI model
ai_model = "gpt-5-mini" # "gpt-5-mini", "gpt-5"

# Initialize LLM models for different tasks
llm_plan = ChatOpenAI(
    model=ai_model,
    temperature=1,
    openai_api_key=api_key
)

llm_exec = ChatOpenAI(
    model=ai_model,
    temperature=1,
    openai_api_key=api_key
)

llm_narrative = ChatOpenAI(
    model=ai_model,
    temperature=1,
    openai_api_key=api_key
)

llm_explainer = ChatOpenAI(
    model=ai_model,
    temperature=1,
    openai_api_key=api_key
)


# File upload and data preview
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    n_rows = st.slider("Number of rows to preview:", 5, 100, 5, step=5)
    st.write(df.head(n_rows))
else:
    df = None

data_context = st.text_area(
    "Optional: Provide context about your dataset (e.g., how was data collected, what columns mean, measurement scales, units, etc.)",
    ""
)

instructions = st.text_area(
    "Enter your question or instructions for data visualization:",
    "Example: Create scatterplot with engagement and job satisfaction"
)


# State management for the LangGraph workflow
class VizState(TypedDict):
    schema: str
    instructions: str
    data_context: str
    plan: Optional[str]
    code: Optional[str]
    explanation: Optional[str]
    df: Optional[object]
    fig: Optional[object]
    error: Optional[str]
    narrative_code: Optional[str]
    narrative_text: Optional[str]
    retry_count_exec: int
    retry_count_narrative: int


# Safe code execution helpers
def run_exec(code: str, df: pd.DataFrame) -> plt.Figure:
    # Remove display and show commands to prevent conflicts
    safe_code = (
        code.replace("display(fig)", "")
            .replace("plt.show()", "")
            .replace("display(", "# display(")
    )
    exec_env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(safe_code, exec_env)
    return exec_env["fig"]

def run_narrative(code: str, df: pd.DataFrame) -> str:
    exec_env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(code, exec_env)
    return exec_env["narrative"]


# Workflow nodes
def planner_node(state: VizState) -> VizState:
    with st.spinner("📝 Generating plan..."):
        plan_msg = llm_plan.invoke(f"""
            Dataset schema: {state['schema']}
            Dataset context: {state['data_context']}
            User request: {state['instructions']}

            Task: Break down the steps for both data cleaning and wrangling (pandas) and visualization (seaborn + matplotlib) that would fulfill the user's request - for example, creating a data visualization, answering a question, providing insights, or helping test a hypothesis.
            Requirements:
            - Wrangling must operate on the input DataFrame df
            - There can be multiple plots or subplots, but they must all be contained in a single matplotlib Figure
            - Visualization must end with a matplotlib Figure object called fig
            - Provide a single, concise step plan that best achieves the user's request or answers their question.
            - Make the analysis as simple as possible while still being effective.
            - Do more complex analysis only if it clearly adds value or if your explictly asked to do so by a user.
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid distortion or chartjunk.
        """)
        state["plan"] = plan_msg.content
    return state


class ExecCode(BaseModel):
    code: str

exec_llm = llm_exec.with_structured_output(ExecCode)


def executor_node(state: VizState) -> VizState:
    with st.spinner("⚙️ Generating and executing code..."):
        exec_plan = exec_llm.invoke(f"""
            Context plan:
            {state['plan']}

            The input DataFrame is named df and is already loaded.
            Write Python code that:
            - Performs necessary wrangling according to the plan
            - Produces the requested visualization
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid distortion or chartjunk.
            - Prefer simplicity and clarity over complexity
            - You may create multiple plots or subplots if it enhances the analysis and user's understanding, but they must all be contained in a single matplotlib Figure
            - Keep in mind that the generated plot should be well readable on the common computer screen (not too small, not too big)
            - Assign the final matplotlib Figure object to variable fig
            - Do NOT use `return` statements anywhere
            - Do NOT use `display()`, `print()`, or `plt.show()`
            - Only return runnable code
            - Don't forget to import any necessary libraries
            - Always end with `fig` defined as the final Figure object
        """)
        code = exec_plan.code
        state["code"] = code

        try:
            state["fig"] = run_exec(code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state


def repair_exec_node(state: VizState) -> VizState:
    if not state["error"]:
        return state

    state["retry_count_exec"] += 1
    if state["retry_count_exec"] > 3:
        return state

    with st.spinner(f"🔧 Repairing failed visualization code (attempt {state['retry_count_exec']}/3)..."):
        repair_msg = exec_llm.invoke(f"""
            The following visualization code failed with an error:
            ```
            {state['code']}
            ```
            Error message:
            {state['error']}

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
            - Keep in mind that the generated plot should be well readadble (not too small, not too crowded)
            - Always end with `fig` defined as the final Figure object
        """)
        repaired_code = repair_msg.code
        state["code"] = repaired_code

        try:
            state["fig"] = run_exec(repaired_code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state


class NarrativeCode(BaseModel):
    code: str

narrative_llm = llm_narrative.with_structured_output(NarrativeCode)

def narrative_node(state: VizState) -> VizState:
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
            - Only use column names from the dataset schema.
            - Use the input DataFrame df (already loaded).
            - Perform actual computations on df (mean, median, counts, correlations, SEM as relevant).
            - Explicitly insert computed values into the string (rounded to 2 decimals).
            - Always wrap the entire narrative text inside triple quotes (\"\"\" ... \"\"\").
            - Make sure both opening and closing triple quotes are present.
            - Assign the result to a variable named `narrative`.
            - Only return runnable Python code.
        """)

        state["narrative_code"] = narrative_plan.code

        try:
            state["narrative_text"] = run_narrative(state["narrative_code"], state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state


def repair_narrative_node(state: VizState) -> VizState:
    if not state["error"]:
        return state

    state["retry_count_narrative"] += 1
    if state["retry_count_narrative"] > 3:
        return state

    with st.spinner(f"🔧 Repairing failed narrative code (attempt {state['retry_count_narrative']}/3)..."):
        repair_msg = narrative_llm.invoke(f"""
            The following narrative code failed with an error:
            ```
            {state['narrative_code']}
            ```
            Error message:
            {state['error']}

            Please suggest corrected Python code that fixes this issue.
            Constraints:
            - Only use column names from the dataset schema.
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

        try:
            state["narrative_text"] = run_narrative(repaired_code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state


def explainer_node(state: VizState) -> VizState:
    with st.spinner("💬 Generating explanation..."):
        explain_msg = llm_explainer.invoke(f"""
            User request: {state['instructions']}
            Dataset context: {state['data_context']}
            Context plan:
            {state['plan']}

            Narrative string:
            {state['narrative_text']}

            Task: Create a narrative explanation of what the generated chart(s) show and how to interpret them,
            and provide specific insights revealed by the analysis for a non-technical audience.
            Constraints:
            - Make it concise and clear.
            - Do not output code. Write only text.
        """)
        state["explanation"] = explain_msg.content.strip()
    return state


# Build the LangGraph workflow
workflow = StateGraph(VizState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("repair_exec", repair_exec_node)
workflow.add_node("narrative", narrative_node)
workflow.add_node("repair_narrative", repair_narrative_node)
workflow.add_node("explainer", explainer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")

# Conditional routing based on error states
workflow.add_conditional_edges(
    "executor",
    lambda state: "repair_exec" if state.get("error") else "narrative"
)

workflow.add_conditional_edges(
    "repair_exec",
    lambda state: "executor" if state.get("error") and state["retry_count_exec"] <= 3 else "narrative"
)

workflow.add_conditional_edges(
    "narrative",
    lambda state: "repair_narrative" if state.get("error") else "explainer"
)

workflow.add_conditional_edges(
    "repair_narrative",
    lambda state: "narrative" if state.get("error") and state["retry_count_narrative"] <= 3 else "explainer"
)

workflow.add_edge("explainer", END)

app = workflow.compile()


# Main execution and UI rendering
if df is not None and instructions and st.button("Generate Insights"):
    schema_str = ", ".join(f"{col}:{dtype}" for col, dtype in df.dtypes.items())

    state: VizState = {
        "schema": schema_str,
        "instructions": instructions,
        "data_context": data_context if data_context else "",
        "df": df,
        "plan": None,
        "code": None,
        "fig": None,
        "explanation": None,
        "error": None,
        "narrative_code": None,
        "narrative_text": None,
        "retry_count_exec": 0,
        "retry_count_narrative": 0,
    }

    result = app.invoke(state)

    # Hidden debug sections (commented out for production UI)
    # if result["plan"]:
    #     with st.expander("📝 Plan"):
    #         st.code(result["plan"], language="markdown")

    # if result["code"]:
    #     with st.expander("⚙️ Final Code"):
    #         st.code(result["code"], language="python")

    # if result["narrative_code"]:
    #     with st.expander("📜 Narrative Code"):
    #         st.code(result["narrative_code"], language="python")

    # if result["narrative_text"]:
    #     with st.expander("📖 Narrative Text"):
    #         st.write(result["narrative_text"])

    if result["fig"]:
        st.pyplot(result["fig"], clear_figure=True, use_container_width=False)

    if result["explanation"]:
        st.subheader("💡 Explanation")
        st.write(result["explanation"])

    if result["error"]:
        st.error(f"Execution still failing: {result['error']}")
