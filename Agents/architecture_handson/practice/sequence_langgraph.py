import os
import sys
from typing import Annotated, TypedDict, List

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

class SequenceState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
