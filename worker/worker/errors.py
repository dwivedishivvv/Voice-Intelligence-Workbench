class RejectError(Exception):
    """Terminal — bad input, never retried."""
    def __init__(self, code: str, detail: str = ""):
        self.code, self.detail = code, detail
        super().__init__(f"{code}: {detail}")
