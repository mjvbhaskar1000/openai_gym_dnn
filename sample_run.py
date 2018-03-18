import gym

env = gym.make("CartPole-v1")
env.reset()

scores = []
def sample_run_env(printstats = False, games = 5, goals=500):
	for i in range(games):
		score = 0
		env.reset()
		for t in range(goals):
			env.render()
			act = env.action_space.sample()
			obs, reward, done, info = env.step(act)
			score += reward
			if done:
				break
		if printstats:
            		print ("Score in ", i+1, "before training is: ", score)
		scores.append(score)
	if printstats:
        	print ("Overall avg score for ", i+1, "runs is: ", sum(scores)/len(scores))
