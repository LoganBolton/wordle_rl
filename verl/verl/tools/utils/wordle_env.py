from wordle.wordle import Wordle


class WordleEnv:
    def __init__(self, word=None, max_attempts=6):
        self.max_attempts = max_attempts
        self.reset(word)

    def reset(self, word=None):
        """Reset the environment to start a new game"""
        self.game = Wordle(word=word, max_attempts=self.max_attempts, display=False)
        self.attempts = 0
        self.gameOver = False
        self.last_result = None
        self.guessed_words = set()
        return self.get_observation()

    def step(self, guess):
        """Take a step in the environment"""
        if self.gameOver:
            return self.get_observation(), 0, True, {"error": "Game is already over"}
        
        # Check if word was already guessed
        is_repeat_guess = guess.lower() in self.guessed_words
        
        # Make the guess and store result
        self.last_result = self.game.guess(guess)
        self.attempts += 1
        
        # Add guess to set of guessed words
        self.guessed_words.add(guess.lower())
        
        # Calculate reward
        reward = self.get_reward(is_repeat_guess)
        
        # Check if game is over
        done = self.game.solved or self.game.failed or self.attempts >= self.max_attempts
        self.gameOver = done
        
        # Get current state
        observation = self.get_observation()
        
        # Create info dict
        info = {
            "solved": self.game.solved,
            "failed": self.game.failed,
            "attempts": self.attempts,
            "target_word": self.game.word,
            "last_guess": guess,
            "last_result": self.last_result,
            "is_repeat_guess": is_repeat_guess
        }
        
        return observation, reward, done, info

    def get_reward(self, is_repeat_guess=False):
        """Calculate reward based on the last guess result"""
        if not self.last_result:  # Invalid guess
            return -2
            
        # Penalty for repeat guesses
        if is_repeat_guess:
            return -3
            
        # Check if game is solved
        if self.game.solved:
            return 10
            
        # Check if game failed (out of attempts)
        if self.game.failed:
            return -5
            
        # Calculate incremental reward based on letter positions
        reward = 0
        for letter, score in self.last_result:
            if score == 2:  # Correct position
                reward += 1
            elif score == 1:  # Wrong position
                reward += 0.5
            # score == 0 (not in word) adds 0
                
        return reward

    def get_observation(self):
        """Get current state observation"""
        return {
            "attempts_made": self.attempts,
            "attempts_remaining": self.max_attempts - self.attempts,
            "game_over": self.gameOver,
            "solved": self.game.solved if hasattr(self.game, 'solved') else False,
            "last_result": self.last_result,
            "alphabet_state": getattr(self.game, 'alphabet', {}),
            "attempt_history": getattr(self.game, 'attempts', []),
            "guessed_words": list(self.guessed_words)
        }

    def get_state_prompt(self, guess=None):
        """Get human-readable state description"""
        if guess is None and self.last_result is None:
            return "Game not started yet."
            
        result = self.last_result
        prompt = f""

        if not result:  # Invalid guess
            prompt += f"Guess '{guess}' is NOT a valid word or is NOT the correct length. Try a different word.\n"
            return prompt

        # Add the result to the prompt
        for letter, score in result:
            if score == 2:
                prompt += f"Letter '{letter}' is in the word and is in the CORRECT position\n"
            elif score == 1:
                prompt += f"Letter '{letter}' is in the word but is in the WRONG position\n"
            elif score == 0:
                prompt += f"Letter '{letter}' is NOT in the word\n"
        
        # Add game status
        if self.game.solved:
            prompt += f"\n🎉 Congratulations! You solved it in {self.attempts} attempts!\n"
        elif self.game.failed:
            prompt += f"\n💔 Game over! The word was '{self.game.word}'\n"
        else:
            guessed_list = ", ".join(sorted(self.guessed_words))
            prompt += f"\nAttempts remaining: {self.max_attempts - self.attempts}. Previously guessed words: {guessed_list}. Guess a different word.\n"
        
        return prompt