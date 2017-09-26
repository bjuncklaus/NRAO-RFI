class Individual:

    def __init__(self, attributes=None):
        if attributes is None:
            attributes = []
        self._attributes = attributes
        self.fitness = float("inf")

    def add_attribute(self, attribute):
        self._attributes.append(attribute)

    def update_attribute(self, index, new_attribute):
        self._attributes[index] = new_attribute

    def get_attributes(self):
        return self._attributes

    def __str__(self):
        # mode, freqdevscale, flagneartime, flagnearfreq, growtime, growfreq
        return "Attributes-> mode:{0}, freqdevscale:{1}, flagneartime:{2}, flagnearfreq:{3}, growtime:{4}, growfreq:{5} \n" \
               "Fitness-> {6}".format(self._attributes[0], self._attributes[1], self._attributes[2],
                                     self._attributes[3], self._attributes[4], self._attributes[5], self.fitness)