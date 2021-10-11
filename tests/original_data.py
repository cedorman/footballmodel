# ----------------------------------------------
# This data is from the Original Paper
# ----------------------------------------------

TEST_PROBE = [0, 1, -1, 1, 0, 1, -1, 1, 0]

# Data component
TEST_DATA = [
    [0, 0, -1, 1, 0, 1, -1, 0, 0],
    [0, 1, -1, 1, 0, 1, -1, 1, 0],
    [0, 1, -1, 0, 0, 1, -1, 1, 0],
    [0, 1, -1, 1, 0, 1, -1, 1, 0],
    [0, 1, -1, 1, 0, 1, -1, 1, 0],
    [0, 0, -1, 1, 0, 1, -1, 1, 0],
    [-1, 0, -1, 0, -1, 1, 1, 0, 0],
    [-1, 0, -1, 1, -1, 1, 1, 0, 0],
    [-1, 0, 0, 1, -1, 1, 1, 1, 0],
    [1, 0, -1, -1, 1, 1, -1, 0, 0]
]

# Events that correspond to the above Dat components
TEST_EVENT = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3]

# Hypotheses that correspond to the above data components
TEST_HYPO = [
    [1, -1, 0, 0, 1, 1, -1, 1, 0],
    [1, -1, 0, 0, 1, 1, -1, 1, 0],
    [1, -1, 0, 0, 1, 1, -1, 1, 0],
    [1, 0, 0, 0, 1, 1, -1, 1, 0],
    [1, 0, 0, 0, 1, 1, -1, 1, 0],
    [0, 0, 0, 0, 1, 1, -1, 1, 0],
    [1, -1, 0, 0, 0, 1, 0, 1, 0],
    [0, -1, 0, 0, 0, 1, 0, 1, 0],
    [1, -1, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, -1, 1, 0]
]

# Activation level for the above data components compared to probe
TEST_ACTIVATION = [0.2963, 1., 0.5787, 1, 1, 0.5787, 0.002, 0.0156, 0.0156, 0.0156]

TEST_ACTIVATION_THRESHOLD = 0.216

TEST_CONTENT_VECTOR = [0, 3.58, -4.5, 3.87, 0, 4.45, -4.5, 4.16, 0]

TEST_CONTENT_HYPO_VECTOR = [3.87, -1.9, 0, 0, 4.45, 4.45, -4.5, 4.45, 0]

TEST_UNSPEC_PROBE_DATA = [0, 0.8, -1, 0.87, 0, 1, -1, 0.93, 0]
TEST_UNSPEC_PROBE_HYPO = [0.87, -0.4, 0, 0, 1, 1, -1, 1, 0]

TEST_SEMANTIC_MEMORY_DATA = [
    [0, 1, -1, 1, 0, 1, -1, 1, 0],
    [-1, 0, -1, 1, -1, 1, 1, 1, 0],
    [1, 0, -1, -1, 1, 1, -1, 0, 0]
]

TEST_SEMANTIC_MEMORY_HYPO = [
    [1, -1, 0, 0, 1, 1, -1, 1, 0],
    [1, -1, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, -1, 1, 0]
]

TEST_SEMANTIC_MEMORY_EVENT = [1, 2, 3]

TEST_SEMANTIC_ACTIVATION_NORMED = [0.8229, 0.0906, 0.0864]

TEST_H1_ECHO_INTENSITY = 0.742

TEST_H1_PROBABILITY = 1
