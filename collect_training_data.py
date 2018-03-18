import random
import sample_run as sp

games = 10000
goals = 500
score_req = 50

def train_data_collect():
	t_data = []
	for _ in range(games):
		score = 0
		game_memory = []
		prev_obs = []
		for _ in range(goals):
			act = random.randrange(0,2)
			obs, reward, done, info = sp.env.step(act)
			if len(prev_obs) > 0:
				game_memory.append([prev_obs, act])
			prev_obs = obs
			score += reward
			if done:
				break
		if score >= score_req:
			for mem in game_memory:
				if mem[1] == 1:
					output = [0,1]
				elif mem[1] == 0:
					output = [1,0]

				t_data.append([mem[0], output])

		sp.env.reset()
	return t_data

training = train_data_collect()
