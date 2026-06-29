from typing import Dict, List, Optional

from typer import prompt

class PromptManager:
    def __init__(self, version: str = "default"):
        """
        Allows to quickly swap between different prompt engineering strategies.
        """
        self.version = version

    def build(self, case_data: Dict, retrieved_cases: Optional[List[Dict]] = None) -> str:
        """
        The only method the rest of the system needs to call.
        It routes to the correct template based on available data.
        """
        if not retrieved_cases:
            return self._build_basic_prompt(case_data)
        else:
            return self._build_rag_prompt(case_data, retrieved_cases)

    ###### BASIC ZERO-SHOT PROMPTS (For building the initial database) ######

    def _build_basic_prompt(self, row: Dict) -> str:

        if self.version == "default":

            # We use a structured prompt that guides the LLM to use the visual evidence from graph and context, using Markdown formattingto help the attention mechanisms separate instructions from contextual data
            
            return f"""
# ROLE
You are an expert Data Storyteller and Educational Math Tutor. Your task is to analyze the attached {row['graph_category']} and explain it to a non-expert user (someone with no advanced mathematical or data science background). Your goal is to make data intuitive, engaging, and easy to digest.

# CONTEXT
Here are the technical specifications for the attached graph:
* **Domain:** {row['domain']}
* **Graph Category:** {row['graph_category']}
* **Graph Type:** {row['graph_type']}
* **Underlying Math Concept:** {row['math_concept']}
* **Analytical Task:** {row['analytical_task']}
* **Analyzed Variables:** {row['variables']} (CRITICAL NOTE: The list of variables represents the universe of possible variables in the dataset as a guide. DO NOT assume all of them are shown in the graph. You must visually verify which specific variables are actually rendered in the image before mentioning them.)

# TASK
Analyze the attached visual graph meticulously. Translate the visual data into a clear, engaging narrative. DO NOT guess or hallucinate data that is not visually evident, you MUST rely strictly on the visual evidence provided by the graph and the context above.
1. Start by telling the user what they are looking at in plain, conversational English (use a simple, relatable analogy if it helps explain the '{row['math_concept']}').
2. Point out the main visual takeaways guiding their eyes (e.g., "Notice the large blue bar...", "See how the dots cluster going upwards...").
3. Explain what these visual cues actually mean for the {row['domain']} variables involved in the real world.

# OUTPUT FORMAT
Provide your response as a SINGLE, highly engaging, and accessible PARAGRAPH. 
Do NOT use overly technical jargon, math formulas, or rigid bullet points, use a conversational tone, as if you were explaining the graph to a friend over coffee.
You MUST logically justify your explanation using direct visual evidence from the image, so the user can easily follow along with their own eyes.
"""
        
        # TODO: FUTURE PROMPT VERSIONS TO BE ADDED HERE
        
        else:
            raise ValueError(f"Unknown prompt version: {self.version}")

    ###### FEW-SHOT RAG PROMPTS (For the final user system) ######

    def _build_rag_prompt(self, row: Dict, retrieved_cases: List[Dict]) -> str:
        """
        Few-shot RAG prompt: appends retrieved cases to the unchanged zero-shot base.
        The zero-shot prompt is kept identical to preserve evaluation comparability.
        Retrieved examples serve ONLY as reasoning style references, not data sources.
        """

        # Zero-shot base 
        prompt = self._build_basic_prompt(row)

        # Few-shot block header
        prompt += """

---

# REFERENCE METHODOLOGY (DO NOT COPY DATA)
The following cases are retrieved from a reference database because their graph structure and educational intent is similar to the one you must analyze.

**CRITICAL INSTRUCTIONS FOR USING THESE EXAMPLES:**
1. Observe the **educational tone and reasoning style**: how visual elements (colors, axes, shapes) are translated into simple, intuitive explanations for a non-expert.
2. These examples contain specific numerical values, explicitly named variables, and domain contexts. You MUST treat all of them as **FICTIONAL PLACEHOLDERS** — they belong to entirely different graphs.
3. Do **NOT** copy, reference, or be influenced by the specific variables, values, or domains from these examples.
4. Do **NOT** assume the variables listed in the examples are present in your image.
5. Your analysis must derive **exclusively** from the visual evidence in the attached image and the metadata provided above. Every number, variable name, and real-world claim in your output must be verified in the attached image.

"""

        # Retrieved cases
        for i, case in enumerate(retrieved_cases, 1):
            prompt += f"### Reference Example {i}\n"
            prompt += f"- **Graph Type:** {case.get('graph_type', 'Unknown')}\n"
            prompt += f"- **Math Concept:** {case.get('math_concept', 'Unknown')}\n"
            prompt += f"- **Analytical Task:** {case.get('analytical_task', 'Unknown')}\n"
            prompt += f"**Reference Educational Style:**\n> {case.get('solution_insights', '')}\n\n"

        # Re-anchoring to the main task (to cover recency bias)
        prompt += """
---

Now produce your explanation of the **attached image** for a novel user, following the exact output format specified at the beginning (single accessible paragraph). Your response must be grounded exclusively in what you can visually verify in the attached graph."""

        return prompt