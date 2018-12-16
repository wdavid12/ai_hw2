import random, util
import math
from game import Agent

#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
  """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
  if gameState.isWin():
      return 10000
  if gameState.isLose():
      return -1000

  pacmanState = gameState.getPacmanState()

  grid = gameState.getFood()
  size = grid.width * grid.height

  max_dist = grid.width + grid.height

  food_items = gameState.getFood().asList()
  food_dists = [util.manhattanDistance(pacmanState.getPosition(), item) for item in food_items]

  ghosts = gameState.getGhostPositions()
  ghost_dists = [util.manhattanDistance(pacmanState.getPosition(), ghost) for ghost in ghosts]
  min_ghost_dist = min(ghost_dists)

  return gameState.getScore() + (size-gameState.getNumFood()) + (max_dist-min(food_dists)) + min_ghost_dist


#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """
  def minimax(self, gameState, depth, agentIndex):
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

      actions = gameState.getLegalActions(agentIndex)
      successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]

      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      nextDepth = depth
      if agentIndex == (gameState.getNumAgents()-1):
          nextDepth = depth-1

      scores = [self.minimax(state, nextDepth, nextAgent) for state in successors]

      if agentIndex == 0:
          return max(scores)
      else:
          return min(scores)

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    actions = gameState.getLegalActions(0)
    successors = [gameState.generateSuccessor(0, action) for action in actions]
    scores = [self.minimax(state,self.depth,0) for state in successors]

    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return actions[chosenIndex]



######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """
  def alpha_beta(self, gameState, depth, agentIndex, alpha, beta):
    if depth == 0 or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    nextAgent = (agentIndex + 1) % gameState.getNumAgents()
    nextDepth = depth
    if agentIndex == (gameState.getNumAgents()-1):
      nextDepth = depth-1

    if agentIndex == 0:
      cur_max = -math.inf
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        score = self.alpha_beta(successor, nextDepth, nextAgent, alpha, beta)
        cur_max = max(score, cur_max)
        alpha = max(alpha, cur_max)
        if cur_max >= beta:
          return math.inf
      return cur_max
    else:
      cur_min = math.inf
      for action in gameState.getLegalActions(agentIndex):
        successor = gameState.generateSuccessor(agentIndex, action)
        score = self.alpha_beta(successor, nextDepth, nextAgent, alpha, beta)
        cur_min = min(score, cur_min)
        beta = min(beta, cur_min)
        if alpha >= cur_min:
          return -math.inf
      return cur_min

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    actions = gameState.getLegalActions(0)
    successors = [gameState.generateSuccessor(0, action) for action in actions]
    scores = [self.alpha_beta(state,self.depth,0,-math.inf,math.inf) for state in successors]

    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return actions[chosenIndex]

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """

  def getAction(self, gameState):
    """
      Returns the action using self.depth and self.evaluationFunction

    """

    # BEGIN_YOUR_CODE
    raise Exception("Not implemented yet")
    # END_YOUR_CODE



