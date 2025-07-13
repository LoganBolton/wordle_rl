from wordle.wordle import Wordle


class WordleEnv:
    def __init__(self):
        self.game = Wordle(word = 'words')
        self.attempts = 0
        self.max_attempts = 6
        self.gameOver = False

    def reset(self):
        self.game = Wordle(word = 'words')

    def step(self, guess):
        self.attempts += 1
        return self.game.guess(guess)
    
    def get_reward(self, guess):
        reward = 0
        if self.game.guess(guess) == self.game.word:
            self.gameOver = True
            reward += 10
        elif self.attempts >= self.max_attempts:
            self.gameOver = True
            reward -= 10
        else:
            return self.calculate_reward(guess, self.game.word)
        
    def is_done(self):
        return self.attempts >= self.max_attempts or self.game.guess(guess) == self.game.word
    
        
    def calculate_reward(self, guess, word):
        reward = 0
        # reward for each letter in the correct position
        result = self.game.guess(guess)

        for _char, value in result.items():
            if value == 2:
                reward += 1
            elif value == 1:
                reward += 0.5
            elif value == 0:
                reward += 0

        # reward for guessing the word
        if guess == word:
            reward += 10
    
        return reward

    def get_state_prompt(self, guess, word):
        result = self.game.guess(guess)
        prompt = f"**Current state of the game after guess {guess}**\n"


        if len(result) == 0:
            prompt += f"Guess {guess} is NOT a valid word or is NOT the correct amount of letters. Try a different word.\n"
            return prompt

        # add the result to the prompt
        for char, value in result.items():
            if value == 2:
                prompt += f"letter {char} is in the word and is in the correct position\n"
            elif value == 1:
                prompt += f"letter {char} is in the word but is in the WRONG position\n"
            elif value == 0:
                prompt += f"letter {char} is NOT in the word\n"
            


        return prompt