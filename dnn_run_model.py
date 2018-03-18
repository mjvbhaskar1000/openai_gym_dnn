import sample_run as sp
import random as rd
import numpy as np


scores = []
def dnn_run(mdl, games=10, goals=500):
    for i in range(games):
        score = 0
        game_memory = []
        prev_obs = []
        sp.env.reset()
        for _ in range(goals):
            sp.env.render()
    
            if len(prev_obs)==0:
                act = rd.randrange(0,2)
            else:
                act = np.argmax(mdl.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
                    
            obs, reward, done, info = sp.env.step(act)
            prev_obs = obs
            game_memory.append([obs, act])
            score+=reward
            if done: 
                break
        scores.append(score)
        print ("Score in ", i + 1, "run after training is: ", score)
        
    print("Overall Avg score for ", i + 1, "runs is: ", sum(scores)/len(scores))
