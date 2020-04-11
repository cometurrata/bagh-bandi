import time
from src.agents.agent import Agent
from random import shuffle
from copy import deepcopy


class MinimaxABAgent(Agent):
    """
        Minimax agent with Alpha Beta Pruning inspired from https://github.com/haryoa/evo-pawness/blob/master/ai_modules/classic_algorithm.py
    """
    def __init__(self, agent_type, engine, max_depth=4):
        """
        Initiation
        Parameters
        ----------
        max_depth : int
            The max depth of the tree
        player_color : int
            The player's index as MAX in minimax algorithm
        """
        super(MinimaxABAgent, self).__init__(agent_type, engine)
        self.max_depth = max_depth
        self.node_expanded = 0

    def enemy_turn_action(self, action_key, new_state):
        """
        Nothing to do here
        """
        pass

    def moves(self):
        tuple_move = self.choose_action(self.engine.board)
        if self.engine.board.terminal_test():
            pass
        elif not tuple_move:  # has no possible move
            if self.engine.board.turn == 'tigers':
                self.engine.skip_tiger_recapture()
            else:
                self.engine.skip_goat_turn()
        else:
            self.engine.board = self.engine.move(tuple_move[0], tuple_move[1], self.engine.board)

    def choose_action(self, state):
        """
        Predict the move using minimax alpha beta pruning algorithm
        Parameters
        ----------
        state : State
        Returns
        -------
        float, str:
            The evaluation or utility and the action key name
        """
        state = deepcopy(state)

        self.node_expanded = 0

        start_time = time.time()

        print("MINIMAX AB : Wait AI is choosing")
        # list_action = self.get_possible_action(state)
        eval_score, selected_action_tuple = self._minimax(0, state, True, float('-inf'), float('inf'))
        print("MINIMAX : Done, eval = %.2f, expanded %d" % (eval_score, self.node_expanded))
        print("--- %s seconds ---" % (time.time() - start_time))

        return selected_action_tuple

    def _minimax(self, current_depth, state, is_max_turn, alpha, beta):
        """
        Helper function of minimax
        :param current_depth: The current depth on the tree in recursive
        :param state: State of the current node in recursive
        :param is_max_turn: Check if the current node is the max turn in recursive
        :param alpha: parameter of AB Prunning, save the current maximizer best value
        :param beta: parameter of AB Prunning, save the current minimizer best value
        :return: int , str The value of the best action and the name of the action
        """
        if current_depth == self.max_depth or state.terminal_test():
            return self.evaluation_function(state), tuple()

        self.node_expanded += 1

        possible_action = self.get_possible_action(state)
        # key_of_actions = list(possible_action.keys())
        # list_of_actions = list(possible_action.items())
        list_of_actions = self.set_tuples_possible_action(possible_action)
        shuffle(list_of_actions)  # add randomness here
        best_value = float('-inf') if is_max_turn else float('inf')
        action_target = tuple()
        # for action_key in list_of_actions:
        for action_tuple in list_of_actions:
            # new_state = self.result_function(state,possible_action[action_key])
            new_state = self.result_function(action_tuple[0], action_tuple[1], state)

            eval_child, action_child = self._minimax(current_depth+1, new_state, not is_max_turn, alpha, beta)

            if is_max_turn and best_value < eval_child:
                best_value = eval_child
                # action_target = action_key
                action_target = action_tuple
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break

            elif (not is_max_turn) and best_value > eval_child:
                best_value = eval_child
                # action_target = action_key
                action_target = action_tuple
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        return best_value, action_target
