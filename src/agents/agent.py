from copy import deepcopy

class Agent(object):
    """docstring for Agent."""

    def __init__(self, agent_type, engine):
        """
        Initiation
        Parameters
        ----------
        agent_type : string
            goat or tiger
        engine : Engine
            engine gives the agent the rules of the environment
        """
        super(Agent, self).__init__()
        self.agent_type = agent_type
        self.engine = engine

    def get_possible_action(self, state):
        if state.turn == 'tigers':
            all_possible_actions = state.get_tigers_available_moves()
            all_possible_captures = state.get_tigers_available_captures()
            for capture in all_possible_captures.keys():
                if capture not in all_possible_actions.keys():
                    all_possible_actions[capture] = all_possible_captures[capture]
                else:
                    list_capture_moves_to_add = [move for move in all_possible_captures[capture] if move not in
                                                      all_possible_actions[capture]]
                    all_possible_actions[capture] += list_capture_moves_to_add
        else:
            all_possible_actions = state.get_goats_available_moves()
        all_possible_actions = self.remove_invalid_actions(all_possible_actions, state)
        all_possible_actions = self.remove_duplicate_actions(all_possible_actions)
        return all_possible_actions

    def remove_invalid_actions(self, list_of_actions, state):
        for dep in list_of_actions:
            list_moves_to_remove = []
            for dest in list_of_actions[dep]:
                is_valid = self.engine.is_valid_move(dep, dest, state)
                if not is_valid:
                    list_moves_to_remove.append(dest)
            for move in list_moves_to_remove:
                list_of_actions[dep].remove(move)
        return list_of_actions

    @staticmethod
    def remove_duplicate_actions(list_of_actions):
        for dep in list_of_actions:
            dedup_set_of_actions = set(list_of_actions[dep])
            list_of_actions[dep] = list(dedup_set_of_actions)
        return list_of_actions


    def set_tuples_possible_action(self, possible_actions):
        tuples_possible_action = []
        for dep in possible_actions:
            for dest in possible_actions[dep]:
                tuples_possible_action.append(tuple([dep, dest]))
        return tuples_possible_action

    def get_player(self, state):
        return state.turn

    def result_function(self, departure, destination, board):
        """
        return the new_state given a state and an action to perform on this state

        :param board:
            instance of Board
        :param departure:
            move decided by the player
        :param destination:
            move decided by the player
        :return:
            new_state

        :rtype:Board
        """
        # exploratory function -> you don't want to change the state of the board
        is_game_move = False
        next_board = deepcopy(board)
        next_board = self.engine.move(departure, destination, next_board, is_game_move)
        return next_board

    def evaluation_function(self, state):
        return state.utility_function(self.agent_type)