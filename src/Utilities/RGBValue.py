class RGBValue:
    def __init__(self, r_mean, g_mean, b_mean, mean_score):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean
        self.mean_score = mean_score

    def rgbvals(self):
        return {'R':self.r_mean,'G':self.g_mean,'B':self.b_mean}
