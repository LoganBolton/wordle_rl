import json
import pandas as pd
from typing import List, Dict, Any

def read_jsonl_file(filename: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    Each line in the file should be a valid JSON object.
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                # Strip whitespace and skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                # Parse JSON
                json_obj = json.loads(line)
                data.append(json_obj)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
    
    return data

def analyze_wordle_data(data: List[Dict[str, Any]]) -> None:
    """
    Analyze the Wordle game data and print some basic statistics.
    """
    print(f"Total games: {len(data)}")
    
    # Extract metadata
    solved_games = [game for game in data if game['metadata']['solved']]
    failed_games = [game for game in data if not game['metadata']['solved']]
    
    print(f"Games solved: {len(solved_games)}")
    print(f"Games failed: {len(failed_games)}")
    print(f"Success rate: {len(solved_games) / len(data) * 100:.1f}%")
    
    if solved_games:
        avg_attempts = sum(game['metadata']['attempts_used'] for game in solved_games) / len(solved_games)
        print(f"Average attempts for solved games: {avg_attempts:.2f}")
    
    # Show some target words
    target_words = [game['metadata']['target_word'] for game in data[:10]]
    print(f"\nFirst 10 target words: {target_words}")

def extract_conversations(data: List[Dict[str, Any]]) -> List[str]:
    """
    Extract all assistant responses from the conversations.
    """
    all_responses = []
    
    for game in data:
        for message in game['messages']:
            if message['role'] == 'assistant':
                all_responses.append(message['content'])
    
    return all_responses

def convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the JSONL data to a pandas DataFrame for easier analysis.
    """
    records = []
    
    for game in data:
        record = {
            'target_word': game['metadata']['target_word'],
            'attempts_used': game['metadata']['attempts_used'],
            'solved': game['metadata']['solved'],
            'total_reward': game['metadata']['total_reward'],
            'num_messages': len(game['messages']),
        }
        
        # Extract first guess (usually the second assistant message)
        assistant_messages = [msg for msg in game['messages'] if msg['role'] == 'assistant']
        if assistant_messages:
            first_guess = assistant_messages[0]['content']
            # Extract the word from \\boxed{word} format
            import re
            match = re.search(r'\\boxed\{(\w+)\}', first_guess)
            record['first_guess'] = match.group(1) if match else 'unknown'
        
        records.append(record)
    
    return pd.DataFrame(records)

# Example usage
if __name__ == "__main__":
    # Read the JSONL file
    print("Reading JSONL file...")
    data = read_jsonl_file('wordle_examples.jsonl')
    
    # Basic analysis
    print("\n=== Basic Analysis ===")
    analyze_wordle_data(data)
    
    # Convert to DataFrame for more analysis
    print("\n=== DataFrame Analysis ===")
    df = convert_to_dataframe(data)
    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame head:")
    print(df.head())
    
    print("\nSolved games by attempts:")
    solved_df = df[df['solved'] == True]
    print(solved_df['attempts_used'].value_counts().sort_index())
    
    print("\nMost common first guesses:")
    print(df['first_guess'].value_counts().head(10))
    
    # Example: Look at one complete conversation
    print("\n=== Sample Conversation ===")
    if data:
        sample_game = data[0]
        print(f"Target word: {sample_game['metadata']['target_word']}")
        print(f"Solved: {sample_game['metadata']['solved']}")
        print(f"Attempts: {sample_game['metadata']['attempts_used']}")
        print("\nConversation excerpt:")
        for i, message in enumerate(sample_game['messages'][:4]):  # First 4 messages
            print(f"{message['role']}: {message['content'][:100]}...") 