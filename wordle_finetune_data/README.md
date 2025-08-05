# Wordle Synthetic Data

- I want to build conversation examples of logically playing Wordle to fine tune a model
- I want examples where it takes the assistant 3- 6 guesses to play Wordle
- After the assistant guesses the correct word, the game should immediately end
- When making the games, I also want to store some metadata about whath the ground truth word was
- I will be using `Qwen/Qwen3-1.7B` as the model I am finetuning. It is important that the generated examples follow this model's chat format
    - I want to do a FULL finetune. Not LORA.
- I want to create a few thousand examples with different ground truth words
    - The fine tune data is `wordle_finetune.jsonl`
- I want to monitor the fine tuned with wandb


## Wordle Package Example
```
>>> from wordle.wordle import Wordle
>>> game = Wordle(word="apple")
>>> game.guess("plant") # incorrect guess with partial reward
[('p', 1), ('l', 1), ('a', 1), ('n', 0), ('t', 0)]
>>> game.guess("plantaoeu") # too long!
[]
>>> game.guess("pla") # too short!
[]
>>> game.guess("apple") # correct guess
[('a', 2), ('p', 2), ('p', 2), ('l', 2), ('e', 2)]
```


## Wordle conversation examples


**system**
You are an agent playing Wordle. Your job is to guess a real 5-letter word. At the start of every turn, think step by step of what word you should guess. Make your reasoning very short and to the point because time will run out if you think too long. Format your final one word answer in \boxed{}

**user**
Respond with your initial guess to start the game.

**assistant**  
Let me start with a word that has common vowels and consonants to get maximum information.
\boxed{arose}

**user**
Letter 'a' is in the word and is in the CORRECT position
Letter 'r' is NOT in the word  
Letter 'o' is NOT in the word
Letter 's' is NOT in the word
Letter 'e' is in the word and is in the CORRECT position
Attempts remaining: 5. Previously guessed words: arose. Guess a different word.

**assistant**
Excellent! I know the word starts with 'a' and ends with 'e'. The pattern is a _ _ _ e. I need to think of a word that does not contain 'r', 'o', or 's'. Let me think about possible candidate words.
- 'a-n-t-ic'
    - This word contains the letter 'a' at the first position. However, it is missing an 'e'. Let's try another word
- 'a-m-p-l-e'
    - This word contains the letter 'a' at the first position. It also contains the letter 'e' at the 5th letter. I have not yet guessed the letters 'm', 'p', or 'l' yet. 

Since the word 'ample' contains previously verified letter positions and it uses three new letter options to maximally explore new options, let's choose it.

\boxed{ample}

**user**
Letter 'a' is in the word and is in the CORRECT position
Letter 'm' is NOT in the word
Letter 'p' is in the word and is in the CORRECT position
Letter 'l' is in the word but is in the WRONG position
Letter 'e' is in the word and is in the CORRECT position
Attempts remaining: 4. Previously guessed words: arose, ample. Guess a different word.

**assistant**
Perfect! I have a_ple with positions 1, 3, 4, and 5 confirmed. I need to find the letter for position 2. Letters ruled out: r, o, s, m.
Since 'p' is known to be in the word, it could also be in position 2, making "apple" - a common real word that fits perfectly.