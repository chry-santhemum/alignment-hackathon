import json
import asyncio
from utils import call_claude_async, call_pref_model_async
import os

async def sample_claude_response(prompt, max_retries=3):
    """Sample a response from Claude 4 for a given prompt"""
    for attempt in range(max_retries):
        try:
            response = await call_claude_async(
                prompt,
                model_name="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.7
            )
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to generate response after {max_retries} attempts: {e}")
                raise
            
            # Exponential backoff
            delay = 2 ** attempt
            print(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)

async def sample_all_responses(dataset, max_concurrent=5):
    """Sample Claude 4 responses for all prompts in the dataset"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def sample_with_semaphore(item, idx):
        async with semaphore:
            prompt = item["prompt"]
            print(f"Sampling response for prompt {idx + 1}/{len(dataset)} ({item['topic']})")
            
            response = await sample_claude_response(prompt)
            
            return {
                "prompt": prompt,
                "response": response,
                "category": item["category"],
                "topic": item["topic"],
                "belief": item["belief"],
                "belief_summary": item["belief_summary"]
            }
    
    # Create tasks for all prompts
    tasks = [sample_with_semaphore(item, idx) for idx, item in enumerate(dataset)]
    
    # Execute with controlled concurrency
    results = await asyncio.gather(*tasks)
    
    return results

async def main():
    """Main function to load prompts and sample Claude responses"""
    print("Loading prompts from syco_simple_20.json...")
    
    with open("datasets/syco_simple_20.json", "r") as f:
        dataset = json.load(f)
    
    print(f"Found {len(dataset)} prompts")
    
    print("Sampling Claude 4 responses...")
    responses = await sample_all_responses(dataset, max_concurrent=5)
    
    # Ensure datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    
    # Save responses
    output_file = "datasets/syco_simple_claude_responses.json"
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)
    
    print(f"Saved {len(responses)} Claude responses to {output_file}")

async def evaluate_response(prompt, response, max_retries=3):
    """Evaluate a prompt-response pair using the reward model with retry logic"""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    for attempt in range(max_retries):
        try:
            print("Sending out reward model call")
            score = await call_pref_model_async(conversation)
            return score
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to evaluate response after {max_retries} attempts: {e}")
                raise
            
            # Exponential backoff with jitter to avoid thundering herd
            import random
            base_delay = 2 ** attempt
            jitter = random.uniform(0, 1)
            delay = base_delay + jitter
            print(f"Evaluation attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
            await asyncio.sleep(delay)

async def evaluate_all_responses(responses, max_concurrent=4):
    """Evaluate all Claude responses using the reward model"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(item, idx):
        async with semaphore:
            prompt = item["prompt"]
            response = item["response"]
            print(f"Evaluating response {idx + 1}/{len(responses)} ({item['topic']})")
            
            score = await evaluate_response(prompt, response)
            
            return {
                "prompt": prompt,
                "response": response,
                "score": score,
                "category": item["category"],
                "topic": item["topic"],
                "belief": item["belief"],
                "belief_summary": item["belief_summary"]
            }
    
    # Create tasks for all responses
    tasks = [evaluate_with_semaphore(item, idx) for idx, item in enumerate(responses)]
    
    # Execute with controlled concurrency
    results = await asyncio.gather(*tasks)
    
    return results

async def evaluate_saved_responses(max_concurrent=4):
    """Load saved responses and evaluate them with the reward model"""
    print("Loading Claude responses from syco_simple_claude_responses.json...")
    
    with open("datasets/syco_simple_claude_responses.json", "r") as f:
        responses = json.load(f)
    
    print(f"Found {len(responses)} responses to evaluate")
    print(f"Max concurrent preference model calls: {max_concurrent}")
    
    print("Evaluating responses with reward model...")
    evaluated_responses = await evaluate_all_responses(responses, max_concurrent=max_concurrent)
    
    # Calculate statistics
    scores = [item["score"] for item in evaluated_responses]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    # Create output data with statistics
    output_data = {
        "statistics": {
            "total_responses": len(evaluated_responses),
            "average_score": avg_score,
            "min_score": min_score,
            "max_score": max_score
        },
        "responses": evaluated_responses
    }
    
    # Save evaluated responses with statistics
    output_file = "datasets/syco_simple_claude_evaluated.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(evaluated_responses)} evaluated responses to {output_file}")
    
    print(f"Score statistics:")
    print(f"  Average: {avg_score:.4f}")
    print(f"  Min: {min_score:.4f}")
    print(f"  Max: {max_score:.4f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        # Evaluate existing responses
        max_concurrent = 4  # Default
        if len(sys.argv) > 2:
            max_concurrent = int(sys.argv[2])
        asyncio.run(evaluate_saved_responses(max_concurrent=max_concurrent))
    else:
        # Generate responses (default)
        asyncio.run(main())