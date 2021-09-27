#
# Implementation of the HyGene model
#

class Symptoms:
    """Represents a set of symptoms or cues, represented as an array of floats"""

    def __init__(self):
        self.symptoms = None

    def set_symptoms(self, symptoms):
        self.symptoms = symptoms

    def distance_to_symptoms(self, comparison_symptoms):
        """Passed in a set of symptoms, we return the distance to our symptoms"""



class Hygene:
    """HyGene model.


    """

    def __init__(self, t_max: float, act_min: float):
        self.symptoms = None
        self.T_max = t_max
        self.ACT_MIN = act_min
        self.num_semantic_failures = 0

    def add_episode(self, symptoms: Symptoms, result: float):
        """Add an episodic event to memory along with the result of that
        episodic event."""
        pass

    def add_semantic_memory(self, symptoms: Symptoms, result: float):
        """Add information to semantic memory"""
        pass

    def receive_symptoms(self, symptoms: Symptoms):
        self.symptoms = symptoms
        self.activate_episodic_traces()
        self.compare_with_semantic()
        return self.generate_probability_judgement()

    def activate_episodic_traces(self):
        pass

    def compare_with_semantic(self):
        while self.num_semantic_failures < self.T_max:
            score, semantic_index = self.get_highest_semantic()
            if score > self.ACT_MIN:
                self.add_semantic_to_working_memory(semantic_index)

    def generate_probability_judgement(self):
        pass

    def get_highest_semantic(self):
        return 0., 0

    def add_semantic_to_working_memory(self, semantic_index):
        pass
