import asyncio
import os
import sys
import textwrap
from typing import List, Optional
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_cleaning_env'))
from models import DataCleaningAction, DataCleaningObservation
from server.data_cleaning_env_environment import DataCleaningEnvironment

IMAGE_NAME = os.getenv("IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "data_cleaning_env"
MAX_STEPS = 3

TASKS = ["easy", "medium", "hard"]

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data cleaning agent. You will receive messy data and must return cleaned data.
    Rules:
    - Fix dates to YYYY-MM-DD format
    - Replace None/missing names with "Unknown"
    - Replace None/missing ages with 0
    - Replace None/missing emails with "no-email@example.com"
    - Remove duplicate rows, keep first occurrence
    
    You must reply with ONLY a Python list of dictionaries, no explanation, no markdown.
    Example: [{"id": 1, "name": "Alice", "dob": "1995-01-14"}]
""").strip()

def get_model_action(client, task_description, messy_data):
    user_prompt = f"""
Task: {task_description}

Messy data:
{messy_data}

Return ONLY a Python list of cleaned dictionaries. No explanation.
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean up markdown if model wraps in backticks
        text = text.replace("```python", "").replace("```", "").strip()
        cleaned = eval(text)
        if isinstance(cleaned, list):
            return cleaned
        return []
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return []

async def run_task(client, env, task_id):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            cleaned = get_model_action(client, obs.task_description, obs.messy_data)
            action = DataCleaningAction(cleaned_data=cleaned)
            obs = env.step(action)

            reward = obs.reward or 0.0
            done = obs.done
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=str(cleaned)[:80], reward=reward, done=done, error=None)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DataCleaningEnvironment()

    total_score = 0.0
    for task_id in TASKS:
        score = await run_task(client, env, task_id)
        total_score += score
        print(f"[DEBUG] Task {task_id} final score: {score:.3f}", flush=True)

    avg = total_score / len(TASKS)
    print(f"[DEBUG] Average score across all tasks: {avg:.3f}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())