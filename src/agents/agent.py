class Agent(object):
    """docstring for Agent."""

    def __init__(self, agent_type, engine):
        super(Agent, self).__init__()
        self.agent_type = agent_type
        self.engine = engine

    def get_possible_action(self, state):
        if state.turn == 'tigers':
            all_possible_actions = state.get_tigers_available_moves()
            all_possible_actions += state.get_tigers_available_captures()
        else:
            all_possible_actions = state.get_goats_available_moves()
        return all_possible_actions

    def get_player(self, state):
        return state.turn


    def result_function(self, state, action):
        """
        return the new_state given a state and an action to perform on this state

        :param state:
            instance of Board
        :param action:
            move decided by the player
        :return:
            new_state

        :rtype:Board
        """


        pass

    def evaluation_function(self, state):
        return state.utility_function(self.agent_type)