class PointIterator:
    def __init__(self, df):
        self.df = df
        self.cur = 0
        self.len = df.shape[0]

    def get(self):
        ret = self.df.iloc[self.cur, :]
        self.cur += 1
        if self.cur >= self.len:
            self.cur = 0

        return ret




