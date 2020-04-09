from src.domain.board import Board


class Engine:

    def __init__(self):
        self.board = Board()
        self.re_capture_allowed = False

    def take_action(self, action, board, is_action_of_game):
        self.re_capture_allowed = False

        if board.turn == "goats":
            board.turn = "tigers"
            departure_index = board.goatsPositions.index(action.departure)
            board.goatsPositions[departure_index] = action.destination
        else:
            board.turn = "goats"
            departure_index = board.tigerPositions.index(action.departure)
            board.tigerPositions[departure_index] = action.destination

            if action.get_is_a_capture():
                position = self.get_capture_position(action.departure, action.destination)
                self.kill_goat_by_position(position, board)

                # Tigers can keep playing if did not jump over a goat stack
                if position not in board.goatsPositions:
                    board.turn = "tigers"
                    self.re_capture_allowed = True
        if is_action_of_game:
            board.action_history.append(action)
        return board

    @staticmethod
    def get_capture_position(departure, destination):
        return departure + ((destination - departure) >> 1)

    def kill_goat_by_position(self, position, board):
        board.goatsPositions.remove(position)
        board.dead_goats += 1

    def move(self, departure, destination, board, is_game_move=True):
        action = Action()
        action.set_destination(destination)
        action.set_departure(departure)

        if board.turn == "tigers":
            if self.is_move_a_capture(departure, destination, board):
                action.set_is_a_capture()

        board = self.take_action(action, board, is_game_move)
        return board

    def is_move_a_capture(self, departure, destination, board):
        if departure in board._capture_connections:
            if destination in board._capture_connections[departure]:
                return True
        return False

    def is_valid_goat_move(self, departure, destination, board):
        goat_moves = board.get_goats_available_moves()
        if departure in goat_moves.keys():
            if destination in goat_moves[departure]:
                return True
        return False

    def is_valid_tiger_move(self, departure, destination, board):
        tiger_moves = board.get_tigers_available_moves()
        if departure in tiger_moves.keys():
            if destination in tiger_moves[departure]:
                return True
        return False

    def is_valid_tiger_capture(self, departure, destination, board):
        tiger_captures = board.get_tigers_available_captures()
        if departure in tiger_captures.keys():
            if destination in tiger_captures[departure]:
                capture_position = self.get_capture_position(departure, destination)
                if capture_position in board.goatsPositions:
                    return True
        return False

    def is_valid_move(self, departure, destination, board):

        if board.turn == "goats" and self.is_valid_goat_move(departure, destination, board):
            return True

        elif board.turn == "tigers":
            if len(self.action_history) > 0 and self.action_history[-1].get_is_a_capture():
                return self.is_valid_multi_capture(departure, destination, board)
            elif self.is_valid_tiger_move(departure, destination, board):
                return True
            elif self.is_valid_tiger_capture(departure, destination, board):
                return True

        print("Move is not valid")
        return False

    def is_valid_multi_capture(self, departure, destination, board):
        if len(board.action_history) > 0 and board.action_history[-1].get_is_a_capture():
            if self.is_valid_tiger_capture(departure, destination, board):
                if board.action_history[-1].destination != departure:
                    print("Recapture must use same tiger")
                elif self.get_capture_position(departure, destination) == self.get_capture_position(
                        board.action_history[-1].departure, board.action_history[-1].destination):
                    print("Cannot capture back and forth")
                else:
                    return True
        return False

    def skip_tiger_recapture(self):
        if self.re_capture_allowed:
            self.re_capture_allowed = False
            self.board.turn = "goats"



class Action:

    def __init__(self):
        self.destination = None
        self.departure = None
        self.is_a_capture = False

    def set_destination(self, destination):
        self.destination = destination

    def set_departure(self, departure):
        self.departure = departure

    def set_is_a_capture(self):
        self.is_a_capture = True

    def get_is_a_capture(self):
        return self.is_a_capture
