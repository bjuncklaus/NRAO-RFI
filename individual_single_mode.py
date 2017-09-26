class Individual:

    def __init__(self, mode, attributes=None):
        if attributes is None:
            attributes = []
        self._attributes = attributes
        self.fitness = float("inf")
        self.flag_percentage = 0
        self.mode = mode

    def add_attribute(self, attribute):
        self._attributes.append(attribute)

    def update_attribute(self, index, new_attribute):
        self._attributes[index] = new_attribute

    def cmdlist(self):
        if (self.mode == 'tfcrop'):
            return [" mode='{0}' timecutoff={1} freqcutoff={2} maxnpieces={3} usewindowstats='{4}'"
                    .format(self.mode, self._attributes[0], self._attributes[1], self._attributes[2], self._attributes[3]),
                    " mode='extend' growtime={0} growfreq={1} flagneartime={2} flagnearfreq={3} growaround={4}"
                    .format(self._attributes[4], self._attributes[5], self._attributes[6], self._attributes[7], self._attributes[8])]
        else:
            return [" mode='{0}' timedevscale={1} freqdevscale={2} winsize={3} "
                    .format(self.mode, self._attributes[0], self._attributes[1], self._attributes[2]),
                    " mode='extend' growtime={0} growfreq={1} flagneartime={2} flagnearfreq={3} growaround={4}"
                    .format(self._attributes[3], self._attributes[4], self._attributes[5], self._attributes[6], self._attributes[7])]

    def get_attributes(self):
        return self._attributes

    # @property
    def get_fitness(self):
        return self.fitness

    # @get_fitness.setter
    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_flag_percentage(self, flag_percentage):
        self.flag_percentage = flag_percentage

    def __str__(self):
        if (self.mode == "tfcrop"):
            return "Attributes-> mode:{0}, timecutoff:{1}, freqcutoff:{2}, maxnpieces:{3}, usewindowstats:{4}, " \
                   "growtime:{5}, growfreq:{6}, flagneartime:{7}, flagnearfreq:{8}, growaround:{9} \n" \
                   "Fitness-> {10} | % Flagged -> {11}".format(self.mode,
                                          self._attributes[0], self._attributes[1], self._attributes[2], self._attributes[3],
                                          self._attributes[4], self._attributes[5], self._attributes[6], self._attributes[7],
                                          self._attributes[8], self.fitness, self.flag_percentage)
        else:
            return "Attributes-> mode:{0}, timedevscale:{1}, freqdevscale:{2}, winsize:{3}, " \
                   "growtime:{4}, growfreq:{5}, flagneartime:{6}, flagnearfreq:{7}, growaround:{8} \n" \
                   "Fitness-> {9} | % Flagged -> {10}".format(self.mode,
                                          self._attributes[0], self._attributes[1], self._attributes[2],
                                          self._attributes[3], self._attributes[4], self._attributes[5],
                                          self._attributes[6], self._attributes[7], self.fitness, self.flag_percentage)