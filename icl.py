import asyncio
import random
from typing import List, Dict, Tuple
import numpy as np
from utils import call_claude, call_claude_async, call_pref_model_async
from prompts import DIVERSE_RESPONSE_PROMPT, ICL_PROMPT, ICL_CONVERSATION_PROMPTS
import logging
from datetime import datetime
import os


# Set up logging
def setup_logging():
    """Set up logging to a timestamped file."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_filename = f"logs/icl_log_{timestamp}.txt"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


# Initialize logger
logger = setup_logging()


class RewardICL:
    def __init__(self, prompt: str, selection_strategy: str = "high_and_coverage"):
        self.prompt = prompt  # The user prompt we want to optimize reward for
        self.response_library = []
        self.selection_strategy = selection_strategy

    async def sample_diverse_responses(
        self,
        n_responses: int,
        thinking_tokens: int|None,
        temperature: float,
        model: str = "claude-sonnet-4-20250514"
    ) -> List[str]:
        """Sample multiple diverse responses in a single API call."""
        responses = []
        diverse_prompt = DIVERSE_RESPONSE_PROMPT.format(
            n_responses=n_responses,
            user_prompt=self.prompt
        )
        logger.info("-"*25 + " begin diverse_prompt " + "-"*25)
        logger.info(diverse_prompt)
        logger.info("-"*25 + " end diverse_prompt " + "-"*25)
        messages = [{"role": "user", "content": diverse_prompt}]

        while len(responses) < n_responses:
            raw_response = await call_claude_async(messages, model_name=model, temperature=temperature, thinking_tokens=thinking_tokens)
            if isinstance(raw_response, tuple):
                logger.info("-"*25 + " begin thinking " + "-"*25)
                logger.info(raw_response[0])
                logger.info("-"*25 + " end thinking " + "-"*25)
                raw_response = raw_response[1]

            responses.extend([resp.split("</response>")[0].strip() 
                              for resp in raw_response.split("<response>") 
                              if resp.strip()])            
            await asyncio.sleep(1.0)  # Rate limiting
        
        responses = responses[:n_responses]
        return responses


    async def sample_icl_responses(
        self,
        n_examples: int,
        n_responses: int,
        thinking_tokens: int|None,
        temperature: float,
        model: str = "claude-sonnet-4-20250514"
    ) -> List[str]:
        """Sample diverse responses using ICL examples in a single API call."""
        # Format examples
        examples_text = ""
        selected_examples = self.select_diverse_examples(n_examples, self.selection_strategy)
        
        for i, (response, score) in enumerate(selected_examples, 1):
            examples_text += f"\nExample {i} (reward score: {score:.3f}):\n"
            examples_text += f"User: {self.prompt}\n"
            examples_text += f"Assistant: {response}\n"
        
        # Build prompt
        icl_prompt = ICL_PROMPT.format(
            examples=examples_text,
            n_responses=n_responses,
            user_prompt=self.prompt,
        )
        logger.info("-"*25 + " begin icl_prompt " + "-"*25)
        logger.info(icl_prompt)
        logger.info("-"*25 + " end icl_prompt " + "-"*25)
        
        messages = [{"role": "user", "content": icl_prompt}]
        responses = []

        while len(responses) < n_responses:
            raw_response = await call_claude_async(messages, model_name=model, temperature=temperature, thinking_tokens=thinking_tokens)

            if isinstance(raw_response, tuple):
                logger.info("-"*25 + " begin thinking " + "-"*25)
                logger.info(raw_response[0])
                logger.info("-"*25 + " end thinking " + "-"*25)
                raw_response = raw_response[1]

            responses.extend([resp.split("</response>")[0].strip() for resp in raw_response.split("<response>") if resp.strip()])
            await asyncio.sleep(1.0)  # Rate limiting
        
        responses = responses[:n_responses]
        return responses


    async def evaluate_responses(
        self,
        responses: List[str],
        batch_size: int = 5,
        max_retries: int = 3,
        initial_delay: float = 1.0
    ) -> List[Tuple[str, float]]:
        """Evaluate responses using the preference model with rate limiting and backoff, 
        and write the results to the response library."""
        scored_responses = []
        
        for i in range(0, len(responses), batch_size):
            batch_tasks = []
            batch_responses = responses[i:i+batch_size]
            
            # Retry logic for each batch
            retry_count = 0
            delay = initial_delay
            
            while retry_count < max_retries:
                try:
                    # Prepare batch tasks
                    batch_tasks = []
                    for response in batch_responses:
                        messages = [
                            {"role": "user", "content": self.prompt},
                            {"role": "assistant", "content": response}
                        ]
                        batch_tasks.append(call_pref_model_async(messages))
                    
                    # Execute batch with gather
                    batch_scores = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results and handle partial failures
                    successful_results = []
                    failed_indices = []
                    
                    for idx, (response, score) in enumerate(zip(batch_responses, batch_scores)):
                        if isinstance(score, Exception):
                            failed_indices.append(idx)
                            logger.error(f"Error evaluating response {idx}: {score}")
                        else:
                            successful_results.append((response, score))
                    
                    # Add successful results
                    scored_responses.extend(successful_results)
                    
                    # If some failed, retry only those
                    if failed_indices:
                        logger.warning(f"Retrying {len(failed_indices)} failed evaluations (attempt {retry_count + 1}/{max_retries})")
                        batch_responses = [batch_responses[idx] for idx in failed_indices]
                        retry_count += 1
                        
                        # Exponential backoff
                        await asyncio.sleep(delay)
                        delay *= 2  # Double the delay for next retry
                    else:
                        # All successful, break retry loop
                        break
                        
                except asyncio.CancelledError:
                    # Re-raise cancellation
                    raise
                except Exception as e:
                    logger.error(f"Batch evaluation error: {e}")
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        logger.error(f"Max retries reached. Assigning neutral scores to {len(batch_responses)} responses")
                        # Assign neutral scores to remaining failed responses
                        for response in batch_responses:
                            scored_responses.append((response, 0.5))
                        break
                    
                    # Exponential backoff
                    logger.info(f"Retrying batch (attempt {retry_count + 1}/{max_retries}) after {delay}s delay")
                    await asyncio.sleep(delay)
                    delay *= 2
            
            # Rate limiting between batches (with smaller delay if we already waited)
            if i + batch_size < len(responses) and delay == initial_delay:
                await asyncio.sleep(initial_delay)
        
        return scored_responses


    def select_diverse_examples(
        self,
        n_examples: int,
        strategy: str
    ) -> List[Tuple[str, float]]:
        """Select diverse examples from the response library using various strategies."""
        if len(self.response_library) <= n_examples:
            return self.response_library
        
        sorted_library = sorted(self.response_library, key=lambda x: x[1])
        
        if strategy == "quantile":
            # Select examples at different quantiles to show the full reward spectrum
            indices = np.linspace(0, len(sorted_library) - 1, n_examples, dtype=int)
            return [sorted_library[i] for i in indices]
        
        elif strategy == "high":
            return sorted_library[-n_examples:]
        
        elif strategy == "high_and_coverage":
            # Sample half from high rewards, rest for coverage
            high_reward_count = n_examples // 2
            coverage_count = n_examples - high_reward_count
            
            # Get high reward examples (top half)
            high_reward_examples = sorted_library[-high_reward_count:]
            
            # Get coverage examples (randomly sampled from the remaining samples)
            remaining_examples = sorted_library[:-high_reward_count]
            if coverage_count > 0:
                # Randomly sample from the remaining examples
                coverage_examples = random.sample(remaining_examples, coverage_count)
            else:
                coverage_examples = []
        
            return high_reward_examples + coverage_examples
        
        elif strategy == "extremes_and_middle":
            # Select from extremes and middle regions
            n_low = n_examples // 3
            n_mid = n_examples // 3
            n_high = n_examples - n_low - n_mid
            
            low_examples = sorted_library[:n_low]
            mid_start = len(sorted_library) // 2 - n_mid // 2
            mid_examples = sorted_library[mid_start:mid_start + n_mid]
            high_examples = sorted_library[-n_high:]
            
            return low_examples + mid_examples + high_examples
        
        elif strategy == "clustering":
            # Cluster by reward score and sample from each cluster
            scores = [score for _, score in sorted_library]
            clusters = np.array_split(scores, n_examples)
            selected = []
            
            for i, cluster in enumerate(clusters):
                # Find the response closest to the cluster mean
                cluster_mean = np.mean(cluster)
                start_idx = i * len(sorted_library) // n_examples
                best_idx = start_idx + np.argmin(np.abs(cluster - cluster_mean))
                selected.append(sorted_library[best_idx])
            
            return selected
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


    async def iterative_optimize_diverse(
        self,
        n_iterations: int,
        responses_per_call: int,
        calls_per_iter: int,
        n_ICL_examples: int,
        eval_batch_size: int,
        temperature_schedule: str,
        thinking_tokens: int|None,
        model: str = "claude-sonnet-4-20250514",
    ) -> Dict:
        """Iteratively optimize using diverse response generation."""
        
        self.response_library = []
        iteration_results = []
        
        # Temperature scheduling
        if thinking_tokens is not None:
            temperatures = [1.0] * n_iterations
            logger.info("Using thinking tokens, so temperature is set to 1.0")
        elif temperature_schedule == "linear":
            temperatures = np.linspace(1.0, 0.5, n_iterations)
        elif temperature_schedule == "exponential":
            temperatures = 1.0 * np.power(0.5, np.linspace(0, 1, n_iterations))
        else:
            temperatures = [0.8] * n_iterations
        
        for i in range(n_iterations):
            logger.info("\n" + "="*50)
            logger.info(f"ITERATION {i+1}/{n_iterations}")
            logger.info("="*50)
            logger.info(f"Library size: {len(self.response_library)}")
            logger.info(f"Temperature: {temperatures[i]:.2f}")
            
            iteration_responses = []
            
            if len(self.response_library) == 0:
                # First iteration: use diverse sampling without ICL
                tasks = []
                for call in range(calls_per_iter):
                    logger.info(f"Initial diverse sampling (call {call+1}/{calls_per_iter})...")
                    tasks.append(self.sample_diverse_responses(
                        n_responses=responses_per_call, 
                        thinking_tokens=thinking_tokens, 
                        temperature=temperatures[i], 
                        model=model,
                    ))
                iteration_responses = await asyncio.gather(*tasks)
                iteration_responses = [resp for sublist in iteration_responses for resp in sublist]
            else:
                tasks = []
                for call in range(calls_per_iter):
                    logger.info(f"Generating responses with ICL (call {call+1}/{calls_per_iter})...")
                    tasks.append(self.sample_icl_responses(
                        n_examples=n_ICL_examples, 
                        n_responses=responses_per_call, 
                        thinking_tokens=thinking_tokens, 
                        temperature=temperatures[i], 
                        model=model,
                    ))
                iteration_responses = await asyncio.gather(*tasks)
                iteration_responses = [resp for sublist in iteration_responses for resp in sublist]
            
            # Evaluate all responses from this iteration
            logger.info(f"Evaluating {len(iteration_responses)} responses...")
            scored_responses = await self.evaluate_responses(iteration_responses, eval_batch_size)

            for response, score in scored_responses:
                logger.info(f"Score: {score}")
                logger.info(f"Response: {response}")
                logger.info("-"*100)
            
            # Add to library
            self.response_library.extend(scored_responses)
            
            # Calculate iteration statistics
            iter_scores = [s for _, s in scored_responses]
            all_scores = [s for _, s in self.response_library]
            
            iteration_results.append({
                "iteration": i + 1,
                "new_responses": len(scored_responses),
                "library_size": len(self.response_library),
                "iter_avg_score": np.mean(iter_scores),
                "iter_max_score": np.max(iter_scores),
                "library_avg_score": np.mean(all_scores),
                "library_max_score": np.max(all_scores),
                "library_90th_percentile": np.percentile(all_scores, 90)
            })
            
            logger.info("\nIteration stats:")
            logger.info(f"  New responses avg score: {np.mean(iter_scores):.3f}")
            logger.info(f"  New responses max score: {np.max(iter_scores):.3f}")
            logger.info(f"  Library avg score: {np.mean(all_scores):.3f}")
            logger.info(f"  Library max score: {np.max(all_scores):.3f}")
        
        # Find best response overall
        best_response, best_score = max(self.response_library, key=lambda x: x[1])
        
        return {
            "prompt": self.prompt,
            "response_library": self.response_library,
            "iteration_results": iteration_results,
            "best_response": best_response,
            "best_score": best_score,
            "final_library_size": len(self.response_library)
        }


async def main(n_iterations=8, responses_per_call=8, calls_per_iter=2):
    # Example usage with diverse response generation
    test_prompt = ICL_CONVERSATION_PROMPTS[0]
    trainer = RewardICL(
        prompt=test_prompt,
        selection_strategy="high_and_coverage"
    )
    
    # Run iterative optimization with diverse sampling
    result = await trainer.iterative_optimize_diverse(
        n_iterations=n_iterations,
        responses_per_call=responses_per_call,
        calls_per_iter=calls_per_iter,
        n_ICL_examples=8,
        temperature_schedule="linear",
        eval_batch_size=4,
        thinking_tokens=4096,
    )
    
    # Run baseline best-of-N
    tasks = []
    for i in range(n_iterations * calls_per_iter):
        tasks.append(trainer.sample_diverse_responses(
            n_responses=8,
            thinking_tokens=1024,
            temperature=1.0,
        ))
    baseline_responses = await asyncio.gather(*tasks)
    baseline_responses = [resp for sublist in baseline_responses for resp in sublist]
    baseline_scores = await trainer.evaluate_responses(baseline_responses, batch_size=4)

    
    logger.info("\n" + "="*50)
    logger.info("FINAL RESULTS")
    logger.info("="*50)
    
    logger.info("Best-of-N baseline:")
    logger.info(f"Best score achieved: {max([s for _, s in baseline_scores]):.3f}")
    logger.info(f"Avg score: {np.mean([s for _, s in baseline_scores]):.3f}")
    logger.info("Best response:")
    logger.info(baseline_responses[np.argmax([s for _, s in baseline_scores])])

    logger.info("="*50)
    logger.info(f"Total unique responses collected: {result['final_library_size']}")
    logger.info(f"Best score achieved: {result['best_score']:.3f}")
    
    logger.info("\nProgress over iterations:")
    for iter_result in result['iteration_results']:
        logger.info(f"  Iteration {iter_result['iteration']}: "
              f"max={iter_result['library_max_score']:.3f}, "
              f"avg={iter_result['library_avg_score']:.3f}, "
              f"90th percentile={iter_result['library_90th_percentile']:.3f}")
    
    logger.info("\nBest response:")
    logger.info(result['best_response'])


if __name__ == "__main__":
    asyncio.run(main(n_iterations=8, responses_per_call=8, calls_per_iter=1))
