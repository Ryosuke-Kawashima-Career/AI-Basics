"""WebAgent: an LLM agent that operates a web browser through Selenium.

How the agent works
-------------------
The agent repeats an *observe -> think -> act* loop until the task is finished:

1. Observe
   Selenium opens the target URL in Chrome. The page source is cleaned with
   BeautifulSoup: <script>, <style>, <svg>, <head> and similar noisy tags are
   removed, whitespace is collapsed and the HTML is truncated so it fits into
   the model's context window. A screenshot is saved for every step.

2. Think
   The cleaned HTML, the task and the list of previous actions are written into
   the last (empty) user message of a few-shot chat template stored in
   ``web_prompt.json``. A local Hugging Face chat model (``apply_chat_template``
   + ``generate``) predicts the next action as a small JSON-like dictionary:

       {"Action": "TYPE", "Tag": "input[name='my-text']", "Value": "Hello"}

3. Parse
   ``postprocess_action`` turns the raw model output into a dict with the keys
   ``Action`` / ``Tag`` / ``Value``. It tries ``json.loads``, then
   ``ast.literal_eval`` (Python-style quotes and ``None``), and finally extracts
   the first ``{...}`` block with a regex. ``eval`` is never used, so the
   model cannot execute arbitrary code.

4. Act
   ``action_commands`` finds the element by its CSS selector (``Tag``) and
   executes one of the supported actions:
       * CLICK  - click the element (via JavaScript, robust to overlays)
       * TYPE   - clear the field and type ``Value``
       * SELECT - choose ``Value`` in a <select> drop-down
       * DONE   - the model signals that the task is complete

5. Repeat
   The executed action is appended to the history ("previous actions") and the
   loop starts again on the updated page. It stops when the model answers
   DONE / None, when the same action is predicted twice in a row, or when
   ``max_steps`` is reached.

Prompt templates (web_prompt.json)
----------------------------------
* template1 - Mind2Web style, answer ``{'Action', 'Value'}`` (used for offline
  accuracy evaluation on the Mind2Web demo data).
* template2 - live-page style, answer ``{"Action", "Tag", "Value"}`` where Tag
  is a CSS selector. This template drives the real browser.
* template3 - multiple choice over candidate DOM elements, answer
  ``{"Answer", "Action", "Tag", "Value"}`` (used for ranking evaluation).

Evaluation
----------
* ``evaluate_accuracy``  - exact-match accuracy of predicted "ACTION VALUE"
  strings against the Mind2Web gold targets.
* ``evaluate_by_ranking`` - extracts DOM elements from the live page, samples k
  candidates that contain the gold element, asks the model to pick one
  (multiple choice) and checks whether the right element was chosen.

Requirements
------------
* Google Chrome installed. Selenium >= 4.6 downloads a matching chromedriver
  automatically (Selenium Manager).
* ``uv add selenium`` (beautifulsoup4, transformers and torch are already used
  in this repository).

Usage
-----
    uv run python Agents/webagent/webagent.py
    uv run python Agents/webagent/webagent.py --no-headless --max-steps 3 \\
        --task "Select 'Two' in the Dropdown (select)"
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import re
from pathlib import Path
from typing import Callable, Optional

import torch
from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_PATH = Path(__file__).with_name("web_prompt.json")
DEFAULT_URL = "https://www.selenium.dev/selenium/web/web-form.html"
DEFAULT_TASK = "Type 'Hello' into the Text input."
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Tags that carry no information for choosing an action.
NOISY_TAGS = ["script", "style", "svg", "head", "noscript", "meta", "link", "iframe"]

# "Rule:" blocks appended to the user message; they match each template's answer format.
RULES = {
    "template1": (
        "Rule:\n"
        "* You have skill to 'CLICK', 'TYPE', and 'SELECT'\n"
        "* Your response must be following format: {'Action': str, 'Value': str}\n"
        "* If action is 'CLICK', value is None\n"
        "* Do not generate unnecessary text such as system messages."
    ),
    "template2": (
        "Rule:\n"
        "* You have skill to 'CLICK', 'TYPE', and 'SELECT'\n"
        "* If the task is already completed, answer with action 'DONE'\n"
        '* Your response must be following format: {"Action": str, "Tag": str, "Value": str}\n'
        "* Tag must be a CSS selector of the target element\n"
        "* If action is 'CLICK', value is None\n"
        "* Do not generate unnecessary text such as system messages."
    ),
    "template3": (
        "Rule:\n"
        "* You have skill to 'CLICK', 'TYPE', and 'SELECT'\n"
        '* Your response must be following format: {"Answer": str, "Action": str, "Tag": str, "Value": str}\n'
        "* If action is 'CLICK', value is None.\n"
        "* Do not generate unnecessary text such as system messages."
    ),
}


def load_dataset(path: str = "/content/mind2web_demo_data.json") -> list[dict]:
    """Loads the Mind2Web demo data (a list of steps)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class WebAgent:
    """LLM agent that reads a web page and operates it with Selenium."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prompt_path: Path | str = PROMPT_PATH,
        template_key: str = "template2",
        headless: bool = True,
        max_html_chars: int = 6000,
        screenshot_dir: Path | str = "./content",
    ):
        self.model_name = model_name
        self.template_key = template_key
        self.headless = headless
        self.max_html_chars = max_html_chars
        self.screenshot_dir = Path(screenshot_dir)
        self.prompt_templates = self.load_prompt_templates(prompt_path)
        self.model, self.tokenizer = self.load_model()
        self.driver: Optional[webdriver.Chrome] = None

    # ------------------------------------------------------------------ setup
    def load_model(self):
        """Loads a causal LM and its tokenizer from the Hugging Face Hub."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype="auto", device_map="auto"
        )
        model.eval()
        return model, tokenizer

    @staticmethod
    def load_prompt_templates(prompt_path: Path | str) -> dict:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def start_browser(self) -> webdriver.Chrome:
        """Starts Chrome (only once)."""
        if self.driver is None:
            options = Options()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--window-size=1280,1024")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            self.driver = webdriver.Chrome(options=options)
        return self.driver

    def close(self):
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- observe
    def get_clean_html(self) -> str:
        """Returns a compact version of the current page's HTML."""
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        for tag in soup(NOISY_TAGS):
            tag.decompose()
        html = soup.prettify(formatter=None)
        html = re.sub(r"\s+", " ", html).strip()
        return html[: self.max_html_chars]

    def save_screenshot(self, name: str) -> Path:
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self.screenshot_dir / name
        self.driver.save_screenshot(str(path))
        return path

    # ------------------------------------------------------------------ think
    def create_prompt(
        self,
        task: str,
        html: str,
        previous_actions: Optional[str] = None,
        template_key: Optional[str] = None,
    ) -> list[dict]:
        """Fills the last user message of a few-shot template.

        Args:
            task: natural-language task for the agent
            html: (cleaned) HTML of the current page
            previous_actions: actions already executed, one per line
            template_key: key in web_prompt.json (defaults to self.template_key)
        Returns:
            list of chat messages ready for ``apply_chat_template``
        """
        template_key = template_key or self.template_key
        if not previous_actions:
            previous_actions = "None"

        prompt = (
            f"'''\n{html}\n'''\n\n"
            "Based on the HTML webpage above, try to complete the following task:\n"
            f"Task: {task}\n"
            f"Previous actions:\n{previous_actions}\n"
            "What should be the next action?\n"
            f"{RULES[template_key]}"
        )
        messages = copy.deepcopy(self.prompt_templates[template_key])
        messages[-1]["content"] = prompt
        return messages

    @torch.no_grad()
    def generate(self, messages: list[dict], max_new_tokens: int = 128) -> str:
        """Runs the chat model and returns only the newly generated text."""
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ parse
    @staticmethod
    def postprocess_action(response: str) -> dict:
        """Parses the model output into {"Action", "Tag", "Value", "Answer"}.

        Args:
            response: str such as {"Action": str, "Tag": str, "Value": str}
        """
        def _parse(text: str):
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None

        data = _parse(response.strip())
        if not isinstance(data, dict):
            matched = re.search(r"\{.*?\}", response, flags=re.DOTALL)
            data = _parse(matched.group()) if matched else None
        if not isinstance(data, dict):
            return {"Action": "", "Tag": "", "Value": "", "Answer": "", "raw": response}

        def _clean(key: str) -> str:
            value = data.get(key)
            return "" if value is None else str(value).strip()

        return {
            "Action": _clean("Action").upper(),
            "Tag": _clean("Tag"),
            "Value": _clean("Value"),
            "Answer": _clean("Answer").upper(),
            "raw": response,
        }

    @staticmethod
    def format_action(action: dict) -> str:
        """{"Action": "TYPE", "Value": "Boston"} -> "TYPE Boston" (Mind2Web format)."""
        return f"{action['Action']} {action['Value']}".strip()

    # -------------------------------------------------------------------- act
    def action_commands(
        self, action: str, tag: str, value: str, press_enter: bool = False, timeout: float = 5.0
    ) -> bool:
        """Executes one action in the browser.

        Args:
            action: CLICK | TYPE | SELECT
            tag: CSS selector of the target element
            value: text for TYPE / visible text (or value) for SELECT
            press_enter: press Enter after TYPE (submits most forms)
        Returns:
            True if the action was executed successfully
        """
        action = action.upper()
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, tag))
            )
            if action == "CLICK":
                self.driver.execute_script("arguments[0].click();", element)
            elif action == "TYPE":
                element.clear()
                element.send_keys(value)
                if press_enter:
                    element.send_keys(Keys.RETURN)
            elif action == "SELECT":
                select = Select(element)
                try:
                    select.select_by_visible_text(value)
                except Exception:
                    select.select_by_value(value)
            else:
                print(f"Invalid action: {action!r}")
                return False
            return True
        except Exception as e:
            print(f"Failed to execute {action} on {tag!r}: {type(e).__name__}: {e}")
            return False

    # ------------------------------------------------------------- agent loop
    def run(self, url: str, task: str, max_steps: int = 5) -> list[dict]:
        """Lets the agent operate the browser until the task is done.

        Returns:
            history of executed steps
        """
        self.start_browser()
        self.driver.get(url)
        history: list[dict] = []
        previous_actions: list[str] = []

        for step in range(max_steps):
            self.save_screenshot(f"step_{step}.png")
            html = self.get_clean_html()
            messages = self.create_prompt(task, html, "\n".join(previous_actions))
            response = self.generate(messages)
            action = self.postprocess_action(response)
            print(f"[step {step}] model: {response}")

            if action["Action"] in ("", "NONE", "DONE"):
                print(f"[step {step}] agent finished.")
                break
            if history and all(
                history[-1][k] == action[k] for k in ("Action", "Tag", "Value")
            ):
                print(f"[step {step}] same action repeated; stopping.")
                break

            success = self.action_commands(action["Action"], action["Tag"], action["Value"])
            action.update(step=step, success=success)
            history.append(action)
            line = f"[{action['Tag']}] -> {action['Action']}"
            if action["Value"]:
                line += f": {action['Value']}"
            previous_actions.append(line)

        self.save_screenshot("final.png")
        return history

    # ------------------------------------------------------------- evaluation
    def evaluate_accuracy(self, dataset: list[dict], template_key: str = "template1") -> float:
        """Exact-match accuracy on Mind2Web steps.

        Each step is expected to contain "task", "html", "previous_actions"
        and the gold "target" (e.g. "TYPE Boston").
        """
        if not dataset:
            return 0.0
        correct = 0
        for step in dataset:
            messages = self.create_prompt(
                step["task"], step["html"], step.get("previous_actions"), template_key
            )
            prediction = self.format_action(self.postprocess_action(self.generate(messages)))
            if prediction.lower() == step["target"].strip().lower():
                correct += 1
        return correct / len(dataset)

    def extract_dom_elements(self) -> list[dict]:
        """Lists the elements of the current page in the template3 format."""
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        for tag in soup(NOISY_TAGS):
            tag.decompose()

        def _xpath(el: Tag) -> str:
            parts = []
            while isinstance(el, Tag) and el.name != "[document]":
                siblings = [s for s in el.parent.find_all(el.name, recursive=False)] if el.parent else [el]
                index = siblings.index(el) + 1
                parts.append(el.name if len(siblings) == 1 else f"{el.name}[{index}]")
                el = el.parent
            return "/" + "/".join(reversed(parts))

        elements = []
        for dom_id, el in enumerate(soup.find_all(True)):
            attributes = {
                k: " ".join(v) if isinstance(v, list) else v for k, v in el.attrs.items()
            }
            elements.append({
                "dom_id": dom_id,
                "tag": el.name,
                "attributes": attributes,
                "text": el.get_text(" ", strip=True)[:100] if el.name in ("button", "a", "option", "label") else "",
                "xpath": _xpath(el),
            })
        return elements

    @staticmethod
    def sample_candidates(
        dom_elements: list[dict], k: int, is_target: Callable[[dict], bool]
    ) -> tuple[list[dict], Optional[dict]]:
        """Samples k candidates that include the gold element (if found).

        Returns:
            (shuffled candidates, gold element or None)
        """
        target = next((el for el in dom_elements if is_target(el)), None)
        candidates = [target] if target else []
        others = [el for el in dom_elements if el is not target]
        candidates += random.sample(others, min(len(others), max(0, k - len(candidates))))
        random.shuffle(candidates)
        return candidates, target

    def evaluate_by_ranking(
        self,
        task: str,
        is_target: Callable[[dict], bool],
        k: int = 4,
        previous_actions: Optional[str] = None,
    ) -> dict:
        """Asks the model to pick the right element among k candidates.

        Choice "A" means "None of the above"; it is correct when the gold
        element is not on the page.
        """
        candidates, target = self.sample_candidates(self.extract_dom_elements(), k, is_target)
        letters = [chr(ord("B") + i) for i in range(len(candidates))]
        choices = "\n".join(f"{l}. {c}" for l, c in zip(letters, candidates))
        prompt = (
            "Try to complete the following task:\n"
            f"Task: {task}\n"
            f"Previous actions:\n{previous_actions or 'None'}\n"
            "What should be the next action? Please select from the following choices "
            "(If the correct action is not in the page above, please select A. 'None of the above'):\n\n"
            f"A. None of the above\n{choices}\n"
            f"{RULES['template3']}"
        )
        messages = copy.deepcopy(self.prompt_templates["template3"])
        messages[-1]["content"] = prompt
        prediction = self.postprocess_action(self.generate(messages))

        gold = letters[candidates.index(target)] if target else "A"
        return {
            "prediction": prediction,
            "gold_answer": gold,
            "correct": prediction["Answer"] == gold,
        }


# Backwards-compatible name
webAgent = WebAgent


def main():
    parser = argparse.ArgumentParser(description="Let an LLM operate a web browser.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--no-headless", action="store_true", help="show the browser window")
    args = parser.parse_args()

    with WebAgent(model_name=args.model, headless=not args.no_headless) as agent:
        history = agent.run(args.url, args.task, max_steps=args.max_steps)
        print("\n=== History ===")
        for h in history:
            print(f"{h['step']}: {h['Action']} {h['Tag']} {h['Value']} (success={h['success']})")
        if args.url == DEFAULT_URL:
            value = agent.driver.find_element(By.NAME, "my-text").get_attribute("value")
            print(f"Text input value: {value!r}")


if __name__ == "__main__":
    main()
