'''
BlindBot MazeAgent meant to employ Propositional Logic,
Search, Planning, and Active Learning to navigate the
Maze Pitfall problem
'''

import time
import random
from pathfinder import *
from maze_problem import *
from queue import Queue
from maze_clause import *
from maze_knowledge_base import *
from constants import *

goBack = {"U": "D", "D":"U", "L": "R", "R": "L"}

class MazeAgent:
    
    ##################################################################
    # Constructor
    ##################################################################
    
    def __init__ (self, env):
        self.env  = env
        self.loc  = env.get_player_loc()
        self.goal = env.get_goal_loc()
        
        # The agent's maze can be manipulated as a tracking mechanic
        # for what it has learned; changes to this maze will be drawn
        # by the environment and is simply for visuals
        self.maze = env.get_agent_maze()
        
        # The agent's plan will be a queue storing the sequence of
        # actions that the environment will execute
        self.plan = Queue()
        
        # [!] TODO: Initialize any other knowledge-related attributes for
        # agent here, or any other record-keeping attributes you'd like
        self.brain = MazeKnowledgeBase()
        self.visited = []
    
    
    ##################################################################
    # Methods
    ##################################################################
    
    # [!] TODO! Agent currently just runs straight up
    def think(self, perception):
        """
        think is parameterized by the agent's perception of the tile type
        on which it is now standing, and is called during the environment's
        action loop. This method is the chief workhorse of your MazeAgent
        such that it must then generate a plan of action from its current
        knowledge about the environment.
        
        :perception: A dictionary providing the agent's current location
        and current tile type being stood upon, of the format:
          {"loc": (x, y), "tile": tile_type}
        """
        # First check the location you are at and add info to your brain
        fact = MazeClause([((perception["tile"], perception["loc"]),True)])
        self.brain.tell(fact)
        mp = MazeProblem(self.maze)
        transitions = mp.transitions(perception["loc"])
        if (perception["tile"] == Constants.SAFE_BLOCK or perception["tile"]) == Constants.WRN_BLOCK_2:
            # any next move you take won't be a pit
            for state in transitions:
                fact = MazeClause([(("P", state[2]), False)])
                self.brain.tell(fact)
        elif perception["tile"] == Constants.WRN_BLOCK_1:
            for state in transitions:
                if self.maze[state[2][0]][state[2][1]] == Constants.UNK_BLOCK:
                    query = MazeClause([(("P", state[2]), True)])
                    if self.brain.ask(query):
                        self.maze[state[2][0]][state[2][1]] == Constants.PIT_BLOCK
                        self.brain.tell(query)
        elif perception["tile"] == Constants.PIT_BLOCK:
            # check if you go back what the cost would be and compare it to not
            if heuristic(perception["loc"], self.goal) > 2:
                self.plan.put(goBack[self.visited[len(self.visited) - 1][1]])

        if (self.plan.empty()):
            # We updated our map and brain, so now lets get a path
            path = pathfind(MazeProblem(self.maze), perception["loc"], self.goal)
            # Agent simply moves randomly at the moment...
            # Do something that thinks about the perception!
            if len(path) != None and perception["loc"] != self.goal:
                self.plan.put(path[1][0])
                self.visited.append((perception, path[1][0]))
            else:
                direction = random.choice(Constants.MOVES)
                self.plan.put(direction)
                self.visited.append((perception, direction))

    
    def get_next_move(self):
        """
        Returns the next move in the plan, if there is one, otherwise None
        [!] You should NOT need to modify this method -- contact Dr. Forney
            if you're thinking about it
        """
        return None if self.plan.empty() else self.plan.get()


    