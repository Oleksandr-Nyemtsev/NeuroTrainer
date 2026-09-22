
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
)


class MuseDevice:
    def __init__(self):
        self.params = BrainFlowInputParams()
        self.params.timeout = 10

        self.board_id = BoardIds.MUSE_2_BOARD.value
        self.board = BoardShim(self.board_id, self.params)
        self.streaming = False

    def connect(self):
        try:
            if not self.board.is_prepared():
                self.board.prepare_session()

            print("Muse 2 connected.")
            return True

        except BrainFlowError as error:
            print("Muse connection failed:", error)
            return False

    def start(self):
        if not self.board.is_prepared():
            print("Muse is not connected.")
            return False

        if self.streaming:
            return True

        try:
            self.board.start_stream()
            self.streaming = True
            print("Muse EEG stream started.")
            return True

        except BrainFlowError as error:
            print("Stream error:", error)
            return False

    def get_data(self, num_samples=512):
        if not self.streaming:
            return None

        try:
            # Read and remove buffered samples.
            # Prevents repeatedly processing the same EEG.
            return self.board.get_board_data()

        except BrainFlowError as error:
            print("EEG read error:", error)
            return None

    def stop(self):
        if self.board.is_prepared():
            if self.streaming:
                try:
                    self.board.stop_stream()
                except BrainFlowError:
                    pass

            self.streaming = False

            try:
                self.board.release_session()
            except BrainFlowError:
                pass

            print("Muse disconnected.")