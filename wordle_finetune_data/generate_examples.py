#!/usr/bin/env python3
"""
Generate synthetic Wordle fine-tuning examples using hybrid approach:
- Rule-based strategy for logical moves
- LLM-generated reasoning text
- Qwen3 chat format
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from rl_env.env import WordleEnv


def format_word_with_hyphens(word: str) -> str:
    """Format a word with hyphens between letters (e.g., 'apple' -> 'a-p-p-l-e')"""
    if not word or not word.isalpha():
        return word
    return '-'.join(word.lower())


def format_text_with_hyphenated_words(text: str) -> str:
    """Format text by adding hyphens to words that appear to be guesses or game words"""
    # Pattern to match quoted words or words in certain contexts
    # This will match words in quotes, or standalone 5-letter words
    def replace_word(match):
        word = match.group(1)
        if len(word) == 5 and word.isalpha():
            return f"'{format_word_with_hyphens(word)}'"
        return match.group(0)
    
    # Replace quoted words
    text = re.sub(r"'([a-zA-Z]{5})'", replace_word, text)
    
    # Also replace words in boxed format
    def replace_boxed(match):
        word = match.group(1)
        return f"\\boxed{{{format_word_with_hyphens(word)}}}"
    
    text = re.sub(r"\\boxed\{([a-zA-Z]{5})\}", replace_boxed, text)
    
    return text


class WordleStrategy:
    """Rule-based Wordle strategy for logical gameplay"""
    
    # Common starting words with good vowel/consonant coverage
    STARTER_WORDS = ["arose", "adieu", "audio", "ounce", "slice"]
    
    # Common 5-letter words for follow-up guesses
    COMMON_WORDS = [
        "plant", "blade", "shine", "mount", "glide", "storm", "pride", "grape", 
        "frost", "spice", "crumb", "flute", "bench", "might", "depth", "chair",
        "world", "light", "train", "beach", "house", "phone", "money", "music"
    ]
    
    def __init__(self):
        self.letter_freq = "etaoinshrdlcumwfgypbvkjxqz"  # English letter frequency
    
    def get_next_guess(self, env: WordleEnv, attempt_num: int) -> str:
        """Get next logical guess based on game state"""
        
        if attempt_num == 1:
            # First guess: use a good starter word
            return random.choice(self.STARTER_WORDS)
        
        # Get what we know so far
        known_positions = {}  # position -> letter
        wrong_positions = {}  # letter -> set of wrong positions
        excluded_letters = set()
        included_letters = set()
        
        # Analyze previous guesses
        for result in getattr(env.game, 'attempts', []):
            # Skip unused slots (marked with -1)
            if result and result[0][1] != -1:
                for i, (letter, score) in enumerate(result):
                    if score == 2:  # Correct position
                        known_positions[i] = letter
                        included_letters.add(letter)
                    elif score == 1:  # Wrong position
                        included_letters.add(letter)
                        if letter not in wrong_positions:
                            wrong_positions[letter] = set()
                        wrong_positions[letter].add(i)
                    elif score == 0:  # Not in word
                        excluded_letters.add(letter)
        
        # Generate candidate words
        candidates = self._generate_candidates(
            known_positions, wrong_positions, excluded_letters, included_letters
        )
        
        if candidates:
            return random.choice(candidates)
        
        # Fallback: use common words not yet guessed
        guessed = env.guessed_words  # Use the tracked guessed words from env
        available = [w for w in self.COMMON_WORDS if w not in guessed]
        
        return random.choice(available) if available else "plant"
    
    def _generate_candidates(self, known_pos, wrong_pos, excluded, included) -> List[str]:
        """Generate candidate words based on constraints"""
        candidates = []
        
        # Expanded word pool with more common 5-letter words
        word_pool = self.COMMON_WORDS + [
            "about", "above", "abuse", "actor", "acute", "admit", "adopt", "adult", "after", "again",
            "agent", "agree", "ahead", "alarm", "album", "alert", "alien", "align", "alike", "alive",
            "allow", "alone", "along", "alter", "amber", "amend", "among", "angel", "anger", "angle",
            "angry", "apart", "apple", "apply", "arena", "argue", "arise", "array", "arrow", "aside",
            "asset", "avoid", "awake", "award", "aware", "badly", "baker", "bases", "basic", "batch",
            "beach", "began", "begin", "being", "below", "bench", "billy", "birth", "black", "blame",
            "blank", "blast", "blind", "block", "blood", "board", "boost", "booth", "bound", "brain",
            "brand", "brass", "brave", "bread", "break", "breed", "brick", "brief", "bring", "broad",
            "broke", "brown", "brush", "build", "built", "burst", "buses", "buyer", "cable", "calif",
            "carry", "catch", "cause", "chain", "chair", "chaos", "charm", "chart", "chase", "cheap",
            "check", "chest", "chief", "child", "china", "chose", "civil", "claim", "class", "clean",
            "clear", "click", "climb", "clock", "close", "cloud", "coach", "coast", "could", "count",
            "court", "cover", "craft", "crash", "crazy", "cream", "crime", "cross", "crowd", "crown",
            "crude", "curve", "cycle", "daily", "damage", "dance", "dated", "dealt", "death", "debut",
            "delay", "depth", "doing", "doubt", "dozen", "draft", "drama", "drank", "drawn", "dream",
            "dress", "drill", "drink", "drive", "drove", "dying", "eager", "early", "earth", "eight",
            "elite", "empty", "enemy", "enjoy", "enter", "entry", "equal", "error", "event", "every",
            "exact", "exist", "extra", "faith", "false", "fault", "fiber", "field", "fifth", "fifty",
            "fight", "final", "first", "fixed", "flash", "fleet", "floor", "fluid", "focus", "force",
            "forth", "forty", "forum", "found", "frame", "frank", "fraud", "fresh", "front", "fruit",
            "fully", "funny", "giant", "given", "glass", "globe", "going", "grace", "grade", "grand",
            "grant", "grass", "grave", "great", "green", "gross", "group", "grown", "guard", "guess",
            "guest", "guide", "happy", "harry", "heart", "heavy", "henry", "horse", "hotel", "house",
            "human", "ideal", "image", "index", "inner", "input", "issue", "japan", "jimmy", "joint",
            "jones", "judge", "known", "label", "large", "laser", "later", "laugh", "layer", "learn",
            "lease", "least", "leave", "legal", "level", "lewis", "light", "limit", "links", "lives",
            "local", "loose", "lower", "lucky", "lunch", "lying", "magic", "major", "maker", "march",
            "maria", "match", "maybe", "mayor", "meant", "media", "metal", "might", "minor", "minus",
            "mixed", "model", "money", "month", "moral", "motor", "mount", "mouse", "mouth", "moved",
            "movie", "music", "needs", "never", "newly", "night", "noise", "north", "noted", "novel",
            "nurse", "occur", "ocean", "offer", "often", "order", "other", "ought", "paint", "panel",
            "paper", "party", "peace", "peter", "phase", "phone", "photo", "piano", "piece", "pilot",
            "pitch", "place", "plain", "plane", "plant", "plate", "point", "pound", "power", "press",
            "price", "pride", "prime", "print", "prior", "prize", "proof", "proud", "prove", "queen",
            "quick", "quiet", "quite", "radio", "raise", "range", "rapid", "ratio", "reach", "ready",
            "realm", "rebel", "refer", "relax", "repay", "reply", "right", "rigid", "rival", "river",
            "robin", "roger", "roman", "rough", "round", "route", "royal", "rural", "scale", "scene",
            "scope", "score", "sense", "serve", "seven", "shall", "shape", "share", "sharp", "sheet",
            "shelf", "shell", "shift", "shine", "shirt", "shock", "shoot", "short", "shown", "sides",
            "sight", "silly", "since", "sixth", "sixty", "sized", "skill", "sleep", "slide", "small",
            "smart", "smile", "smith", "smoke", "snake", "snow", "sober", "solar", "solid", "solve",
            "sorry", "sort", "sound", "south", "space", "spare", "speak", "speed", "spend", "spent",
            "split", "spoke", "sport", "staff", "stage", "stake", "stand", "start", "state", "steam",
            "steel", "steep", "steer", "steve", "stick", "still", "stock", "stone", "stood", "store",
            "storm", "story", "strip", "stuck", "study", "stuff", "style", "sugar", "suite", "super",
            "sweet", "swift", "swing", "swiss", "table", "taken", "taste", "taxes", "teach", "team",
            "teeth", "terry", "texas", "thank", "theft", "their", "theme", "there", "these", "thick",
            "thing", "think", "third", "those", "three", "threw", "throw", "thumb", "tiger", "tight",
            "timer", "tiny", "title", "today", "token", "topic", "total", "touch", "tough", "tower",
            "track", "trade", "train", "treat", "trend", "trial", "tribe", "trick", "tried", "tries",
            "truck", "truly", "trunk", "trust", "truth", "twice", "uncle", "under", "undue", "union",
            "unity", "until", "upper", "upset", "urban", "usage", "usual", "valid", "value", "video",
            "virus", "visit", "vital", "vocal", "voice", "waste", "watch", "water", "wheel", "where",
            "which", "while", "white", "whole", "whose", "woman", "women", "world", "worry", "worse",
            "worst", "worth", "would", "write", "wrong", "wrote", "young", "youth"
        ]
        
        for word in word_pool:
            if self._is_valid_candidate(word, known_pos, wrong_pos, excluded, included):
                candidates.append(word)
        
        return candidates[:15]  # Return more candidates for better selection
    
    def _is_valid_candidate(self, word, known_pos, wrong_pos, excluded, included):
        """Check if word satisfies all constraints"""
        word = word.lower()
        
        # Check excluded letters
        if any(letter in word for letter in excluded):
            return False
        
        # Check known positions
        for pos, letter in known_pos.items():
            if pos >= len(word) or word[pos] != letter:
                return False
        
        # Check included letters are present
        for letter in included:
            if letter not in word:
                return False
        
        # Check wrong positions
        for letter, wrong_positions in wrong_pos.items():
            for pos in wrong_positions:
                if pos < len(word) and word[pos] == letter:
                    return False
        
        return True


class ReasoningGenerator:
    """Generate natural reasoning text using LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key) if (HAS_ANTHROPIC and api_key) else None
    
    def generate_reasoning(self, 
                         guess: str, 
                         attempt_num: int, 
                         game_state: Dict,
                         target_word: str = None,
                         env = None) -> str:
        """Generate reasoning text for a guess"""
        
        if not self.client:
            # Fallback to template-based reasoning
            return self._template_reasoning(guess, attempt_num, game_state, env)
        
        try:
            prompt = self._build_reasoning_prompt(guess, attempt_num, game_state)
            
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            print(f"LLM reasoning failed: {e}, using template")
            return self._template_reasoning(guess, attempt_num, game_state, env)
    
    def _build_reasoning_prompt(self, guess, attempt_num, game_state):
        """Build prompt for LLM reasoning generation"""
        
        if attempt_num == 1:
            return f"""Generate short reasoning (1-2 sentences) for why '{guess}' is a good first Wordle guess. 
            Focus on vowel coverage and common letters. Be concise and logical."""
        
        # Extract constraints from game state
        constraints = []
        if game_state.get('last_result'):
            for letter, score in game_state['last_result']:
                if score == 2:
                    constraints.append(f"'{letter}' is in correct position")
                elif score == 1:
                    constraints.append(f"'{letter}' is in word but wrong position")
                else:
                    constraints.append(f"'{letter}' is not in word")
        
        constraint_text = "; ".join(constraints[:5])  # Limit length
        
        return f"""Generate short reasoning (2-3 sentences) for guessing '{guess}' in Wordle attempt {attempt_num}.
        
        Previous constraints: {constraint_text}
        
        Explain why this guess makes logical sense given what we know. Be concise and focus on strategy."""
    
    def _template_reasoning(self, guess, attempt_num, game_state, env=None):
        """Generate explicit reasoning with confirmed positions and word options"""
        
        if attempt_num == 1:
            return f"I'll start with '{format_word_with_hyphens(guess)}' to test common vowels and consonants for maximum information."
        
        # Analyze what we know from the game state
        if env:
            known_positions = {}
            wrong_positions = {}
            excluded_letters = set()
            included_letters = set()
            
            # Extract constraints from game attempts
            for result in getattr(env.game, 'attempts', []):
                if result and result[0][1] != -1:
                    for i, (letter, score) in enumerate(result):
                        if score == 2:  # Correct position
                            known_positions[i] = letter
                            included_letters.add(letter)
                        elif score == 1:  # Wrong position
                            included_letters.add(letter)
                            if letter not in wrong_positions:
                                wrong_positions[letter] = set()
                            wrong_positions[letter].add(i)
                        elif score == 0:  # Not in word
                            excluded_letters.add(letter)
            
            # Build reasoning
            reasoning = []
            
            # State what we know so far
            if known_positions or wrong_positions or excluded_letters:
                reasoning.append("Based on the feedback,")
            
            # State confirmed positions
            if known_positions:
                if len(known_positions) == 1:
                    pos = list(known_positions.keys())[0]
                    letter = known_positions[pos]
                    reasoning.append(f"I know '{letter}' goes in position {pos+1}.")
                else:
                    pos_text = []
                    for pos in sorted(known_positions.keys()):
                        pos_text.append(f"'{known_positions[pos]}' in position {pos+1}")
                    reasoning.append(f"I have {', '.join(pos_text)} confirmed.")
            
            # State wrong position letters
            if wrong_positions:
                wrong_letters = list(wrong_positions.keys())
                if len(wrong_letters) == 1:
                    letter = wrong_letters[0]
                    positions = wrong_positions[letter]
                    pos_list = [str(p+1) for p in sorted(positions)]
                    reasoning.append(f"The letter '{letter}' is in the word but not in position {' or '.join(pos_list)}.")
                else:
                    wrong_text = []
                    for letter, positions in wrong_positions.items():
                        pos_list = [str(p+1) for p in sorted(positions)]
                        wrong_text.append(f"'{letter}' (is not in position {', '.join(pos_list)})")
                    reasoning.append(f"Letters {', '.join(wrong_text)} need to be repositioned.")
            
            # State excluded letters
            if excluded_letters:
                excluded_sorted = sorted(excluded_letters)
                if len(excluded_sorted) <= 3:
                    reasoning.append(f"I can rule out {', '.join(excluded_sorted)}.")
                else:
                    reasoning.append(f"I've eliminated {', '.join(excluded_sorted)}.")
            
            # Generate three word options
            strategy = WordleStrategy()
            candidates = strategy._generate_candidates(
                known_positions, wrong_positions, excluded_letters, included_letters
            )
            
            # Get 3 options (including the chosen guess)
            options = []
            if candidates:
                # Ensure our guess is in the options
                if guess in candidates:
                    # Get 2 other candidates and include our guess
                    other_candidates = [c for c in candidates if c != guess]
                    options = [guess] + other_candidates[:2]
                else:
                    options = candidates[:3]
                
                # Randomize the order of the options
                random.shuffle(options)
            
            # Just use whatever valid candidates we have, no fallbacks
            # Randomize the order of available options
            if options:
                random.shuffle(options)
            
            # Explain word choice more naturally
            if options and len(options) > 1:
                formatted_options = [format_word_with_hyphens(word) for word in options]
                if len(options) == 2:
                    reasoning.append(f"The words that fit the constraints are: {', '.join(formatted_options)}.")
                else:
                    reasoning.append(f"Let me list out some possible words that fit the constraints: {', '.join(formatted_options)}.")
            
            # Explain why the chosen guess is best
            formatted_guess = format_word_with_hyphens(guess)
            if options and len(options) > 1:
                choice_explanations = [
                    f"I'll go with '{formatted_guess}' since it fits the pattern and tests new letters.",
                    f"'{formatted_guess}' looks promising - it follows all the constraints.",
                    f"'{formatted_guess}' seems like the best bet to narrow things down.",
                    f"I think '{formatted_guess}' gives me the most information.",
                    f"'{formatted_guess}' makes sense given what I know so far."
                ]
                reasoning.append(random.choice(choice_explanations))
            else:
                # When there's only one option or no clear options
                reasoning.append(f"I'll try '{formatted_guess}' as it fits the constraints.")
            
            return " ".join(reasoning)
        
        # Fallback if no env provided
        return f"Based on the constraints, '{format_word_with_hyphens(guess)}' seems like the most logical choice to test."


class WordleExampleGenerator:
    """Main generator for Wordle fine-tuning examples"""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.strategy = WordleStrategy()
        self.reasoning_gen = ReasoningGenerator(anthropic_api_key)
        self.target_words = self._load_target_words()
    
    def _load_target_words(self) -> List[str]:
        """Load list of target words for examples"""
        # Collection of common 5-letter words that make good Wordle targets
        words = [
            "about", "above", "abuse", "actor", "acute", "admit", "adopt", "adult",
            "after", "again", "agent", "agree", "ahead", "alarm", "album", "alert",
            "alien", "align", "alike", "alive", "allow", "alone", "along", "alter",
            "among", "anger", "angle", "angry", "apart", "apple", "apply", "arena",
            "argue", "arise", "array", "arrow", "aside", "asset", "avoid", "awake",
            "award", "aware", "badly", "baker", "bases", "basic", "beach", "began",
            "begin", "being", "below", "bench", "billy", "birth", "black", "blame",
            "blank", "blast", "blind", "block", "blood", "blown", "blues", "board",
            "boost", "booth", "bound", "brain", "brand", "brass", "brave", "bread",
            "break", "breed", "brief", "bring", "broad", "broke", "brown", "build",
            "built", "buyer", "cable", "calif", "carry", "catch", "cause", "chain",
            "chair", "chaos", "charm", "chart", "chase", "cheap", "check", "chest",
            "chief", "child", "china", "chose", "chunk", "civic", "civil", "claim",
            "class", "clean", "clear", "click", "climb", "clock", "close", "cloud",
            "coach", "coast", "could", "count", "court", "cover", "craft", "crash",
            "crazy", "cream", "crime", "cross", "crowd", "crown", "crude", "curve",
            "delay", "depth", "doing", "doubt", "dozen", "draft", "drama", "drank",
            "dream", "dress", "drill", "drink", "drive", "drove", "dying", "eager",
            "early", "earth", "eight", "elite", "empty", "enemy", "enjoy", "enter",
            "entry", "equal", "error", "event", "every", "exact", "exist", "extra",
            "faith", "false", "fault", "fiber", "field", "fifth", "fifty", "fight",
            "final", "first", "fixed", "flash", "fleet", "floor", "fluid", "focus",
            "force", "forth", "forty", "forum", "found", "frame", "frank", "fraud",
            "fresh", "front", "fruit", "fully", "funny", "giant", "given", "glass",
            "globe", "going", "grace", "grade", "grand", "grant", "grass", "grave",
            "great", "green", "gross", "group", "grown", "guard", "guess", "guest",
            "guide", "happy", "harry", "heart", "heavy", "hence", "henry", "horse",
            "hotel", "house", "human", "ideal", "image", "index", "inner", "input",
            "issue", "japan", "jimmy", "joint", "jones", "judge", "known", "label",
            "large", "laser", "later", "laugh", "layer", "learn", "lease", "least",
            "leave", "legal", "level", "lewis", "light", "limit", "links", "lives",
            "local", "loose", "lower", "lucky", "lunch", "lying", "magic", "major",
            "maker", "march", "maria", "match", "maybe", "mayor", "meant", "media",
            "metal", "might", "minor", "minus", "mixed", "model", "money", "month",
            "moral", "motor", "mount", "mouse", "mouth", "moved", "movie", "music",
            "needs", "never", "newly", "night", "noise", "north", "noted", "novel",
            "nurse", "ocean", "offer", "often", "order", "other", "ought", "paint",
            "panel", "paper", "party", "peace", "peter", "phase", "phone", "photo",
            "piano", "piece", "pilot", "pitch", "place", "plain", "plane", "plant",
            "plate", "point", "pound", "power", "press", "price", "pride", "prime",
            "print", "prior", "prize", "proof", "proud", "prove", "queen", "quick",
            "quiet", "quite", "radio", "raise", "range", "rapid", "ratio", "reach",
            "ready", "realm", "rebel", "refer", "relax", "repay", "reply", "right",
            "rigid", "rival", "river", "robot", "rocky", "roman", "rough", "round",
            "route", "royal", "rural", "scale", "scene", "scope", "score", "sense",
            "serve", "setup", "seven", "shall", "shape", "share", "sharp", "sheet",
            "shelf", "shell", "shift", "shine", "shirt", "shock", "shoot", "short",
            "shown", "sides", "sight", "silly", "since", "sixth", "sixty", "sized",
            "skill", "sleep", "slide", "small", "smart", "smile", "smith", "smoke",
            "space", "spare", "speak", "speed", "spend", "spent", "split", "spoke",
            "sport", "staff", "stage", "stake", "stand", "start", "state", "steam",
            "steel", "steep", "steer", "stern", "stick", "still", "stock", "stone",
            "stood", "store", "storm", "story", "strip", "stuck", "study", "stuff",
            "style", "sugar", "suite", "super", "sweet", "swift", "swing", "swiss",
            "table", "taken", "taste", "taxes", "teach", "teams", "tears", "terry",
            "texas", "thank", "theft", "their", "these", "thick", "thing", "think",
            "third", "those", "three", "threw", "throw", "thumb", "tiger", "tight",
            "tower", "track", "trade", "train", "treat", "trend", "trial", "tribe",
            "trick", "tried", "tries", "truck", "truly", "trunk", "trust", "truth",
            "twice", "under", "undue", "union", "unity", "until", "upper", "upset",
            "urban", "usage", "usual", "valid", "value", "video", "virus", "visit",
            "vital", "vocal", "voice", "waste", "watch", "water", "wheel", "where",
            "which", "while", "white", "whole", "whose", "woman", "women", "world",
            "worry", "worse", "worst", "worth", "would", "write", "wrong", "wrote",
            "young", "youth", "zebra"
        ]
        
        return words
    
    def generate_example(self, target_word: str) -> Dict:
        """Generate a single Wordle conversation example"""
        
        env = WordleEnv(word=target_word, max_attempts=6)
        messages = []
        
        # System message
        system_msg = {
            "role": "system",
            "content": "You are an agent playing Wordle. Your job is to guess a real 5-letter word. At the start of every turn, think step by step of what word you should guess. Make your reasoning very short and to the point because time will run out if you think too long. Format your final one word answer in \\boxed{}"
        }
        messages.append(system_msg)
        
        # Initial user message
        messages.append({
            "role": "user", 
            "content": "Respond with your initial guess to start the game."
        })
        
        attempt_num = 1
        
        while not env.gameOver and attempt_num <= 6:
            # Get logical guess from strategy
            guess = self.strategy.get_next_guess(env, attempt_num)
            
            # Generate reasoning
            reasoning = self.reasoning_gen.generate_reasoning(
                guess, attempt_num, env.get_observation(), target_word, env
            )
            
            # Format assistant response
            assistant_content = f"{reasoning}\n\\boxed{{{format_word_with_hyphens(guess)}}}"
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            # Step environment
            obs, reward, done, info = env.step(guess)
            
            if done:
                break
            
            # Generate user feedback
            feedback = env.get_state_prompt(guess)
            messages.append({
                "role": "user",
                "content": feedback
            })
            
            attempt_num += 1
        
        # Return example with metadata
        return {
            "messages": messages,
            "metadata": {
                "target_word": target_word,
                "attempts_used": attempt_num,
                "solved": env.game.solved,
                "total_reward": sum(env.get_reward() for _ in range(attempt_num))
            }
        }
    
    def generate_dataset(self, num_examples, output_file: str = "wordle_examples.jsonl") -> None:
        """Generate full dataset of examples"""
        
        examples = []
        target_words = random.choices(self.target_words, k=num_examples)
        
        print(f"Generating {num_examples} Wordle examples...")
        
        for target_word in tqdm(target_words):
            try:
                example = self.generate_example(target_word)
                
                # Filter out examples that took 2 or fewer attempts
                if example["metadata"]["attempts_used"] <= 2:
                    continue

                if example["metadata"]["solved"] == False:
                    continue
                examples.append(example)
                
            except Exception as e:
                print(f"Error generating example for '{target_word}': {e}")
                continue
        
        # Save examples
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Generated {len(examples)} examples, saved to {output_path}")
        
        # Print sample
        if examples:
            print("\nSample example:")
            print(json.dumps(examples[0], indent=2))


def main():
    """Main function to generate examples"""
    
    # You can set your Anthropic API key here or as environment variable
    api_key = None  # or os.getenv("ANTHROPIC_API_KEY")
    
    generator = WordleExampleGenerator(anthropic_api_key=api_key)
    generator.generate_dataset(num_examples=10_000)


if __name__ == "__main__":
    main()