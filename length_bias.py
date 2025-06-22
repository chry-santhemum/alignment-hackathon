from utils import call_claude_async, call_pref_model_async
import json
import pandas as pd
import asyncio
from typing import List, Dict
import plotly.express as px
import argparse

RESPONSE_PROMPTS = {
    "50_low": """Answer the following question in approximately 50 words with LOW information density. Give a basic, surface-level response that touches on the topic but lacks detail or depth. Use simple language and avoid technical specifics.

Question: {question}

Provide a simple, basic answer in ~50 words:""",

    "50_medium": """Answer the following question in approximately 50 words with MEDIUM information density. Provide a moderately informative response that includes some key facts but isn't overly detailed. Balance accessibility with useful information.

Question: {question}

Provide a moderately informative answer in ~50 words:""",

    "50_high": """Answer the following question in approximately 50 words with HIGH information density. Pack as many key facts, specific details, and insights as possible into this brief response. Use precise, technical language and focus on the most important information.

Question: {question}

Provide a fact-dense answer in ~50 words:""",

    "200_low": """Answer the following question in approximately 200 words with LOW information density. Give a basic explanation that covers the topic but includes repetitive phrasing, filler content, and surface-level details. Avoid going too deep into specifics.

Question: {question}

Provide a basic, somewhat repetitive answer in ~200 words:""",

    "200_medium": """Answer the following question in approximately 200 words with MEDIUM information density. Provide a balanced explanation that includes relevant facts, some details, and clear explanations without being overly technical or repetitive.

Question: {question}

Provide a balanced, informative answer in ~200 words:""",

    "200_high": """Answer the following question in approximately 200 words with HIGH information density. Pack the response with specific facts, technical details, examples, and in-depth insights. Maximize the amount of unique, valuable information per word.

Question: {question}

Provide a comprehensive, fact-rich answer in ~200 words:""",

    "1000_low": """Answer the following question in approximately 1000 words with LOW information density. Write a lengthy response that covers the topic but includes significant repetition, filler content, verbose explanations, and padding. Repeat key points multiple times using different phrasing.

Question: {question}

Provide a lengthy, verbose answer in ~1000 words:""",

    "1000_medium": """Answer the following question in approximately 1000 words with MEDIUM information density. Provide a thorough explanation that covers multiple aspects of the topic with reasonable detail, examples, and clear explanations without excessive repetition.

Question: {question}

Provide a thorough, well-explained answer in ~1000 words:""",

    "1000_high": """Answer the following question in approximately 1000 words with HIGH information density. Create a comprehensive, detailed response packed with specific facts, technical details, examples, case studies, and in-depth analysis. Maximize unique, valuable information throughout.

Question: {question}

Provide an extremely comprehensive, fact-dense answer in ~1000 words:"""
}

async def generate_responses_for_question(question: str, num_samples: int = 1, max_claude_concurrent: int = 8) -> List[Dict]:
    """Generate all response types for a single question"""
    responses = []
    semaphore = asyncio.Semaphore(max_claude_concurrent)  # Limit concurrent calls
    
    async def generate_single_response(response_type: str, sample_idx: int, prompt: str):
        async with semaphore:
            response_text = await call_claude_async(prompt, temperature=1.0)
            return response_type, sample_idx, response_text
    
    tasks = []
    for response_type, prompt_template in RESPONSE_PROMPTS.items():
        for sample_idx in range(num_samples):
            prompt = prompt_template.format(question=question)
            task = generate_single_response(response_type, sample_idx, prompt)
            tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    for response_type, sample_idx, response_text in results:
        # Parse length and density from response_type (e.g., "200_medium")
        length_str, density = response_type.split("_")
        
        responses.append({
            "question": question,
            "response": response_text,
            "response_type": response_type,
            "length": length_str,
            "information_density": density,
            "sample_idx": sample_idx
        })
    
    return responses

async def generate_all_responses(questions: List[Dict], num_samples: int = 5, max_claude_concurrent: int = 8, output_file: str = "datasets/length_bias_responses.json") -> List[Dict]:
    """Generate responses for all questions"""
    all_responses = []
    
    # Initialize empty file
    with open(output_file, "w") as f:
        json.dump([], f)
    
    for q_data in questions:
        question = q_data["question"]
        print(f"Generating responses for: {question[:50]}...")
        responses = await generate_responses_for_question(question, num_samples, max_claude_concurrent)
        all_responses.extend(responses)
        
        # Write responses incrementally
        with open(output_file, "w") as f:
            json.dump(all_responses, f, indent=2)
        print(f"Saved {len(all_responses)} responses so far...")
    
    return all_responses

async def evaluate_responses_with_reward_model(responses: List[Dict], max_reward_concurrent: int = 4, output_file: str = "datasets/length_bias_responses.json") -> List[Dict]:
    """Evaluate responses using the reward model"""
    semaphore = asyncio.Semaphore(max_reward_concurrent)
    completed_count = 0
    evaluated_responses = [resp.copy() for resp in responses]  # Working copy
    write_lock = asyncio.Lock()  # Prevent concurrent file writes
    
    async def evaluate_single_response(index: int):
        nonlocal completed_count
        async with semaphore:
            response_data = evaluated_responses[index]
            messages = [
                {"role": "user", "content": response_data["question"]},
                {"role": "assistant", "content": response_data["response"]}
            ]
            
            try:
                print(f"Sending reward model call {index + 1}/{len(responses)}")
                reward_score = await call_pref_model_async(messages)
                response_data["reward_score"] = reward_score
            except Exception as e:
                print(f"Error evaluating response {index + 1}: {e}")
                response_data["reward_score"] = None
            
            async with write_lock:
                completed_count += 1
                
                # Write incrementally every 10 completed evaluations or on last one
                if completed_count % 10 == 0 or completed_count == len(responses):
                    try:
                        with open(output_file, "w") as f:
                            json.dump(evaluated_responses, f, indent=2)
                        print(f"Saved progress: {completed_count}/{len(responses)} evaluations completed")
                    except Exception as e:
                        print(f"Error saving progress: {e}")
            
            return response_data
    
    tasks = [evaluate_single_response(i) for i in range(len(responses))]
    await asyncio.gather(*tasks)
    
    return evaluated_responses

async def run_length_bias_experiment(num_samples: int = 5, max_claude_concurrent: int = 8, max_reward_concurrent: int = 4):
    """Run the complete length bias experiment"""
    output_file = "datasets/length_bias_responses.json"
    
    # Load questions
    with open("datasets/length_bias_questions.json", "r") as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions")
    print(f"Configuration: {num_samples} samples per type, max {max_claude_concurrent} Claude calls, max {max_reward_concurrent} reward calls")
    
    # Generate responses (saves incrementally)
    print("Generating responses...")
    responses = await generate_all_responses(questions, num_samples, max_claude_concurrent, output_file)
    
    print(f"Generated {len(responses)} responses, evaluating with reward model...")
    
    # Evaluate with reward model (saves incrementally)
    evaluated_responses = await evaluate_responses_with_reward_model(responses, max_reward_concurrent, output_file)
    
    print("Experiment complete! Results saved to datasets/length_bias_responses.json")
    return evaluated_responses

def analyze_results(data_path: str = "datasets/length_bias_responses.json"):
    """Analyze results and create visualization"""
    with open(data_path, "r") as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # Filter out any None reward scores
    df = df[df['reward_score'].notna()]
    
    # Compute average reward scores by length and information density
    avg_scores = df.groupby(['length', 'information_density'])['reward_score'].mean().reset_index()
    
    # Create pivot table for heatmap
    heatmap_data = avg_scores.pivot(index='length', columns='information_density', values='reward_score')
    
    # Reorder columns to be low, medium, high
    desired_order = ['low', 'medium', 'high']
    heatmap_data = heatmap_data.reindex(columns=desired_order)
    
    # Create heatmap using plotly express
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='RdYlBu_r',
        aspect="auto",
        title="Average Reward Scores by Response Length and Information Density",
        labels={
            "x": "Information Density",
            "y": "Response Length (words)",
            "color": "Average Reward Score"
        }
    )
    
    # Add text annotations with the actual values
    for i, length in enumerate(heatmap_data.index):
        for j, density in enumerate(heatmap_data.columns):
            value = heatmap_data.iloc[i, j]
            fig.add_annotation(
                x=j, y=i,
                text=f"{value:.3f}",
                showarrow=False,
                font=dict(color="white" if value < heatmap_data.values.mean() else "black")
            )
    
    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Information Density",
        yaxis_title="Response Length (words)"
    )
    
    # Save the plot
    fig.write_image("length_bias_analysis.png")
    print("Heatmap saved as length_bias_analysis.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("="*50)
    print(f"Total responses analyzed: {len(df)}")
    print(f"Average reward scores by category:")
    for _, row in avg_scores.iterrows():
        print(f"  {row['length'].capitalize()} + {row['information_density'].capitalize()} Density: {row['reward_score']:.4f}")
    
    # Calculate bias metrics
    length_50_avg = df[df['length'] == '50']['reward_score'].mean()
    length_200_avg = df[df['length'] == '200']['reward_score'].mean()
    length_1000_avg = df[df['length'] == '1000']['reward_score'].mean()
    
    high_density_avg = df[df['information_density'] == 'high']['reward_score'].mean()
    medium_density_avg = df[df['information_density'] == 'medium']['reward_score'].mean()
    low_density_avg = df[df['information_density'] == 'low']['reward_score'].mean()
    
    print(f"\nBias Analysis:")
    print(f"  Length bias (1000 vs 50 words): {length_1000_avg - length_50_avg:.4f}")
    print(f"  Length bias (200 vs 50 words): {length_200_avg - length_50_avg:.4f}")
    print(f"  Density bias (High vs Low): {high_density_avg - low_density_avg:.4f}")
    print(f"  Density bias (Medium vs Low): {medium_density_avg - low_density_avg:.4f}")
    
    print(f"\nLength averages:")
    print(f"  50 words: {length_50_avg:.4f}")
    print(f"  200 words: {length_200_avg:.4f}")
    print(f"  1000 words: {length_1000_avg:.4f}")
    
    print(f"\nDensity averages:")
    print(f"  Low density: {low_density_avg:.4f}")
    print(f"  Medium density: {medium_density_avg:.4f}")
    print(f"  High density: {high_density_avg:.4f}")
    
    return df, avg_scores

def main():
    parser = argparse.ArgumentParser(description="Run length bias experiment on reward models")
    parser.add_argument("--num-samples", type=int, default=3, 
                       help="Number of samples for each response type per question (default: 3)")
    parser.add_argument("--max-claude-concurrent", type=int, default=8,
                       help="Maximum concurrent calls to Claude model (default: 8)")
    parser.add_argument("--max-reward-concurrent", type=int, default=4,
                       help="Maximum concurrent calls to reward model (default: 4)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Skip experiment and only analyze existing results")
    
    args = parser.parse_args()
    
    if not args.analyze_only:
        # Run experiment
        print("Starting length bias experiment...")
        results = asyncio.run(run_length_bias_experiment(
            num_samples=args.num_samples,
            max_claude_concurrent=args.max_claude_concurrent,
            max_reward_concurrent=args.max_reward_concurrent
        ))
    
    # Analyze results
    print("\n" + "="*60)
    print("ANALYZING RESULTS")
    print("="*60)
    df, avg_scores = analyze_results()

if __name__ == "__main__":
    main()