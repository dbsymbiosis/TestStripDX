class color_space_values:
    def __init__(self, red=0, green=0, blue=0, rgb_score=0, l_star=0, a_star=0, b_star=0, cyan=0,
                 yellow=0, magenta=0, key_black=0):
        self.red = red
        self.green = green
        self.blue = blue
        self.rgb_score = rgb_score
        self.l_star = l_star
        self.a_star = a_star
        self.b_star = b_star
        self.cyan = cyan
        self.yellow = yellow
        self.magenta = magenta
        self.key_black = key_black

    def rgbvals(self):
        return {'R': self.red, 'G': self.green, 'B': self.blue}

    def update_rgb_mean(self):
        self.rgb_score = (self.red + self.green + self.blue) / 3
