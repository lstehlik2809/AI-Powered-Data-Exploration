#==============================================================================
# IMPORTS AND DEPENDENCIES
#==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# Helper: convert fig -> PNG bytes
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.read()


#==============================================================================
# STREAMLIT APP CONFIGURATION
#==============================================================================
# Configure the Streamlit app with title and wide layout
st.set_page_config(page_title="AI-Powered Data Exploration Assistant", layout="wide")
st.title("📊 AI-Powered Data Exploration")


#==============================================================================
# OPENAI API KEY SETUP
#==============================================================================
# Load OpenAI API key from environment variables for local development
from dotenv import load_dotenv
import os
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.warning("Please provide a valid OpenAI API key in .env as OPENAI_API_KEY to continue.")
    st.stop()

# Alternative setup for deployment (e.g., Streamlit Cloud) - currently commented out
# api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.warning("Please provide a valid OpenAI API key in st.secrets as OPENAI_API_KEY to continue.")
    st.stop()


#==============================================================================
# AI MODEL CONFIGURATION
#==============================================================================
# Select the AI model to use for all LLM operations
ai_model = "gpt-5-mini" # Options: "gpt-5-mini", "gpt-5"

# Initialize separate LLM instances for different workflow tasks
# Each instance can have different temperature settings for task-specific behavior
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


#==============================================================================
# USER INTERFACE FOR DATA INPUT
#==============================================================================
# File upload widget for CSV files
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Load the uploaded CSV file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
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

# Text area for user instructions/questions about the data
instructions = st.text_area(
    "Enter your question or instructions for data visualization:",
    "Example: Create scatterplot with engagement and job satisfaction"
)


#==============================================================================
# STATE MANAGEMENT FOR LANGGRAPH WORKFLOW
#==============================================================================
# Define the state structure that flows through the LangGraph workflow
# This contains all data and intermediate results needed for the analysis pipeline
class VizState(TypedDict):
    schema: str                    # Dataset column names and types
    instructions: str              # User's request/question
    data_context: str             # Optional context about the dataset
    plan: Optional[str]           # Generated analysis plan
    code: Optional[str]           # Generated Python code for visualization
    explanation: Optional[str]    # Human-readable explanation of results
    df: Optional[object]          # The pandas DataFrame
    fig: Optional[object]         # The matplotlib Figure object
    error: Optional[str]          # Any execution errors
    narrative_code: Optional[str] # Code for generating narrative text
    narrative_text: Optional[str] # Generated narrative with computed values
    retry_count_exec: int         # Number of code execution retry attempts
    retry_count_narrative: int    # Number of narrative generation retry attempts


#==============================================================================
# SAFE CODE EXECUTION HELPERS
#==============================================================================
def run_exec(code: str, df: pd.DataFrame) -> plt.Figure:
    """
    Safely execute visualization code and return the matplotlib Figure.
    Removes potentially problematic display commands before execution.
    """
    # Remove display and show commands to prevent conflicts with Streamlit
    safe_code = (
        code.replace("display(fig)", "")
            .replace("plt.show()", "")
            .replace("display(", "# display(")
    )
    # Create execution environment with necessary libraries and data
    exec_env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(safe_code, exec_env)
    return exec_env["fig"]

def run_narrative(code: str, df: pd.DataFrame) -> str:
    """
    Safely execute narrative generation code and return the narrative string.
    """
    exec_env = {"df": df, "sns": sns, "plt": plt, "pd": pd, "np": np}
    exec(code, exec_env)
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
            - Do more complex analysis only if it clearly adds value or if your explictly asked to do so by a user.
            - Apply good data visualization principles: choose the right chart for the data, keep visuals clear and uncluttered, label everything, use accessible colors, highlight the key insight, and avoid distortion or chartjunk.
        """)
        state["plan"] = plan_msg.content
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
        state["plan"] = reflection_msg.content.strip()
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
    This is where the actual data analysis and plotting happens.
    """
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

        # Attempt to execute the generated code
        try:
            state["fig"] = run_exec(code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state

def repair_exec_node(state: VizState) -> VizState:
    """
    Attempt to fix code execution errors by generating corrected code.
    Includes retry logic with a maximum of 3 attempts.
    """
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

        # Attempt to execute the repaired code
        try:
            state["fig"] = run_exec(repaired_code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
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
    This provides data-driven insights embedded in readable text.
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

        # Execute the narrative generation code
        try:
            state["narrative_text"] = run_narrative(state["narrative_code"], state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state

def repair_narrative_node(state: VizState) -> VizState:
    """
    Repair failed narrative generation code with retry logic.
    Similar to repair_exec_node but specifically for narrative generation.
    """
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

        # Attempt to execute the repaired narrative code
        try:
            state["narrative_text"] = run_narrative(repaired_code, state["df"])
            state["error"] = None
        except Exception as e:
            state["error"] = str(e)
    return state

def explainer_node(state: VizState) -> VizState:
    """
    Generate a human-readable explanation of the analysis results.
    This creates the final interpretation that users will see.
    """
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
    lambda state: "repair_exec" if state.get("error") else "narrative"
)

# Retry logic for failed code execution
workflow.add_conditional_edges(
    "repair_exec",
    lambda state: "executor" if state.get("error") and state["retry_count_exec"] <= 3 else "narrative"
)

# Conditional flow for narrative generation with error handling
workflow.add_conditional_edges(
    "narrative",
    lambda state: "repair_narrative" if state.get("error") else "explainer"
)

# Retry logic for failed narrative generation
workflow.add_conditional_edges(
    "repair_narrative",
    lambda state: "narrative" if state.get("error") and state["retry_count_narrative"] <= 3 else "explainer"
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
    
if df is not None and instructions and st.button("Generate Insights"):
    # Prepare the dataset schema for the AI models
    schema_str = ", ".join(f"{col}:{dtype}" for col, dtype in df.dtypes.items())

    # Initialize the workflow state with user inputs and empty results
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

    # Execute the complete workflow
    result = app.invoke(state)

    # 🔑 Convert fig to PNG bytes if it exists
    if result["fig"]:
        result["fig_png"] = fig_to_png_bytes(result["fig"])
    else:
        result["fig_png"] = None

    st.session_state.viz_result = result

    # DEBUG SECTIONS - Hidden for production UI but useful for development
    # Uncomment these sections to see intermediate workflow results
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

# MAIN RESULTS DISPLAY
# ---- Show persisted results (if any) ----
if st.session_state.viz_result:
    result = st.session_state.viz_result

    if result.get("fig_png"):
        st.image(result["fig_png"], use_container_width=True)

    if result.get("explanation"):
        st.subheader("💡 Explanation")
        st.write(result["explanation"])

    if result.get("error"):
        st.error(f"Execution still failing: {result['error']}")
