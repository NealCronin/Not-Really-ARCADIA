VLM_PROMPT = """
Context Note:
* The input aerial imagery has a distinct and severe green hue shift (tint) introduced by sensor distortion or environmental filtering. 
* Do not classify regions as vegetation, canopy, or grass solely based on green coloration. 
* Rely heavily on surface textures, geometric shapes, structural boundary patterns, and semantic context to accurately differentiate between concrete roads, asphalt, metal/shingle rooftops, water bodies, and actual green foliage.

Analyze the aerial image.

Identify visible terrain, land cover, structures, and objects. For every feature spotted, assign a confidence value from 0.0 (low certainty) to 1.0 (absolute absolute certainty) based on its visual clarity and structural distinctiveness.

Rules:

* Report only things that are clearly visible.
* Do not infer, guess, or hallucinate objects.
* Do not describe location, direction, adjacency, boundaries, shape, or spatial relationships.
* Do not describe size.
* Use short semantic descriptions.
* Focus on what is present, not where it is.
* Merge duplicate observations.
* Output valid JSON only.
* Do not include markdown fences, comments, explanations, or extra text.
* Use double quotes for every JSON key and string value.

Return exactly one JSON object and nothing else.

Format:

{
  "observations": [
    {
      "feature": "dense forest",
      "confidence": 0.25
    },
    {
      "feature": "grassy field",
      "confidence": 0.82
    },
    {
      "feature": "dirt road",
      "confidence": 0.70
    },
    {
      "feature": "small building",
      "confidence": 0.91
    }
  ]
}
"""

REASONING_PROMPT = """
Target Search Mission:
TASK: "{task}"

Visual Observations to Evaluate:
{observations}

Allowed Canonical Grounding Vocabulary:
{vocabulary}

Instructions:
You are an aerial drone mission controller. Your job is to parse the provided "Visual Observations to Evaluate" list and evaluate whether the target of the search mission ("{task}") can physically exist inside or on top of each observed feature.

CRITICAL KEY-MATCHING RULE:
The top-level keys of your output JSON object MUST be the exact, verbatim feature string names provided in the "Visual Observations to Evaluate" list. Do not alter, rephrase, or shorten these keys.

Critical Scoring Rules for the "score" field (0 - 100):
1. The score must strictly represent the likelihood of finding the target ("{task}") within this visual feature. 
2. If the feature IS the target itself (e.g., observation is "car", "vehicle" and task is "Find cars"), assign a score of 100.
3. If the feature is the natural environment where the target operates (e.g., observation is "road", "dirt path", "parking lot" and task is "Find cars"), assign a high contextual score between 75 and 95.
4. If the feature is a location where the target CANNOT physically exist or is hidden completely from aerial view (e.g., observation is "dense forest", "river", "open sky"), you MUST assign a score of 0. Tree canopies block the view of cars.

Requirements:
* For each key, map it to the single best matching category from the allowed canonical vocabulary list.
* Output valid JSON only. Do not include markdown fences (```), comments, explanations, or extra text.

Format Example (If the task was "Find boats" and observations were ["open river", "dense forest"]):
{{
  "open river": {{
    "label": "water",
    "score": 95
  }},
  "dense forest": {{
    "label": "trees",
    "score": 0
  }}
}}
"""