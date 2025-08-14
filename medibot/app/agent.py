# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
from zoneinfo import ZoneInfo

import google.auth
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import SequentialAgent
from google.adk.tools import google_search




_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key.

    Args:
        field (str): a field name to append to
        response (str): a string to append to the field

    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}

patientMedicalSummary = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are the one who will communicate patient using PATIENT_DEMOGRAPHICS, PROVIDER_INFO, and PATIENT_MEDICAL_HISTORY and DECISION state if they have to go and meet with provider to discuss.
    """,
    tools=[],
)

diagnostic_report = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction="""
    You are the provider you need to review the patient medical history and current medications and make a decision if patient need to see provider or not. Use the google search to find the standard values of each report item.

    you save the decision if patient needs to see provider or not as true or false in the state key 'DECISION'
    """,
    tools=[append_to_state, google_search],
)

processData = SequentialAgent(
    name="film_concept_team",
    description="Write a film plot outline and save it as a text file.",
    sub_agents=[
        diagnostic_report,
        patientMedicalSummary
    ],
)

root_agent = Agent(
    name="root_agent",
    model="gemini-2.5-flash",
    instruction=""""
    - Asking user for patient demographics and use 'append_to_state' tool to store the user's response in the 'PATIENT_DEMOGRAPHICS' state key.
    - Asking user for provider info that patient last visit and use 'append_to_state' tool to store the user's response in the 'PROVIDER_INFO' state key.
    - Asking user for patient medical history and use 'append_to_state' tool to store the user's response in the 'PATIENT_MEDICAL_HISTORY' state key.
    """,
    tools=[append_to_state],
    sub_agents=[patientMedicalSummary],
)
