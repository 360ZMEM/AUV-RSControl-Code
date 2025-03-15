import numpy as np
def compute_reward(self): # oracle
    reward = np.zeros(self.N_AUV)
    # waypoint serving
    w_waypoint = 4000
    w_collision = 500
    w_border_xy = 6
    w_border_z = 12
    w_ec = 0.1

    for i in range(self.N_AUV):
        # distance to target waypoint
        dist_to_target = np.linalg.norm(self.xy[i] - self.target_Pcenter[i]) / np.linalg.norm(self.border)
        reward[i] += -w_waypoint * dist_to_target

        # distance between AUVs (for collision avoidance)
        for j in range(i+1,self.N_AUV):
            dist_between_auvs = np.linalg.norm(self.xy[j] - self.xy[i]) / np.linalg.norm(self.border)
            if dist_between_auvs < self.safe_dist:
                reward[i] -= w_collision * ((self.safe_dist - dist_between_auvs))
        
        # target waypoint serving reward (Sparse)
        if self.TL[i] > 0:
            reward[i] += w_waypoint

        reward[i] -= w_ec * self.ec[i] # Energy consumption (Sparse)

        reward[i] -= w_border_xy * self.FX[i] # Penalty of crossing the border (Sparse)

        border_dist = self.calc_AUV_border_dist(i)
        for b in border_dist:
            if b < 1.5 * self.safe_dist:
                reward[i] -= w_border_xy * ((1.5 * self.safe_dist - b)) # x / y constraint
                
        terrain_dist, dir_  = self.calc_AUV_terrain_dist(i)
        height_dist = self.safe_dist if dir_ else 2 * self.safe_dist
        if terrain_dist < height_dist:
            reward[i] -= w_border_z * ((height_dist - terrain_dist)) # z constraint
    return reward / 100 # normalize