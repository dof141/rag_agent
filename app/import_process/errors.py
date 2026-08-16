class ImportTaskError(RuntimeError):
    def __init__(self, stage: str, public_message: str):
        super().__init__(public_message)
        self.stage = stage
        self.public_message = public_message
