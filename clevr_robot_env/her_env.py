import dataclasses
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

import clevr_robot_env


def as_dict(dc):
    return OrderedDict(dataclasses.asdict(dc))


@dataclass
class HERObs:
    observation: np.ndarray
    achieved_goal: str
    desired_goal: str


class HERClevrEnv(clevr_robot_env.ClevrEnv):

    def reset(self, new_scene_content=True):
        observation = super().reset(new_scene_content=new_scene_content)
        goal, goal_program = self.sample_goal()
        self.set_goal(goal, goal_program)
        return as_dict(HERObs(observation=observation, desired_goal=goal, achieved_goal="first step"))

    def step(self, a, **kwargs):
        s, r, t, i = super().step(a, record_achieved_goal=True, goal=self.current_goal, update_des=True)
        achieved = self.np_random.choice(self.achieved_last_step)
        s = as_dict(HERObs(observation=s, desired_goal=self.current_goal_text, achieved_goal=achieved))
        return s, r, t, i