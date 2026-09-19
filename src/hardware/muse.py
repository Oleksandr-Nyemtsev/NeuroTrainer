from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError
)


class MuseDevice:

    def __init__(self):
        self.params = BrainFlowInputParams()
        self.params.timeout = 10
        self.board_id = BoardIds.MUSE_2_BOARD
        self.board = BoardShim(
            self.board_id,
            self.params
        )


    def connect(self):

        try:
            self.board.prepare_session()

            print("Muse 2 connected successfully.")

            return True

        except BrainFlowError as error:

            print("Muse 2 connection failed.")
            print("BrainFlow error:", error)

            return False


    def start(self):

        if not self.board.is_prepared():
            print("Muse 2 is not connected.")
            return False

        self.board.start_stream()

        print("Muse EEG stream started.")

        return True


    def get_data(self, num_samples=512):

        if not self.board.is_prepared():
            print("Muse 2 is not connected.")
            return None

        return self.board.get_current_board_data(
            num_samples
        )


    def stop(self):

        if self.board.is_prepared():

            try:
                self.board.stop_stream()
            except BrainFlowError:
                pass

            self.board.release_session()

            print("Muse 2 disconnected.")