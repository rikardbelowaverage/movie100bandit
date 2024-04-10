def get_regret(max_reward,reward):
    return max_reward-reward

def get_binary_reward(prediction,target):
    if prediction==target:
        return 1
    else:
        return 0