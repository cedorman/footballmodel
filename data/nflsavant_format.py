#
# Mapping between the Football data col headers and the number of the column
#
from enum import Enum


class FootballHeader(Enum):
    GameId =(0, 'int')
    GameDate = (1, 'str')
    Quarter = (2, 'int')
    Minute = (3, 'int')
    Second = (4, 'int')
    OffenseTeam = (5, 'str')
    DefenseTeam = (6, 'str')
    Down = (7, 'int')
    ToGo = (8, 'int')
    YardLine = (9, 'int')
    empty_10 = (10, 'str')
    SeriesFirstDown = (11, 'int')
    empty_11 = (12, 'str')
    NextScore = (13, 'int')
    Description = (14, 'str')
    TeamWin = (15, 'int')
    empty_16 = (16, 'str')
    empty_17 = (17, 'str')
    SeasonYear = (18, 'int')
    Yards = (19, 'int')
    Formation = (20, 'str')
    PlayType = (21, 'str')
    IsRush = (22, 'int')
    IsPass = (23, 'int')
    IsIncomplete = (24, 'int')
    IsTouchdown = (25, 'int')
    PassType = (26, 'int')
    IsSack = (27, 'int')
    IsChallenge = (28, 'int')
    IsChallengeReversed = (29, 'int')
    Challenger = (30, 'str')
    IsMeasurement = (31, 'int')
    IsInterception = (32, 'int')
    IsFumble = (33, 'int')
    IsPenalty = (34, 'int')
    IsTwoPointConversion = (35, 'int')
    IsTwoPointConversionSuccessful = (36, 'int')
    RushDirection = (37, 'str')
    YardLineFixed = (38, 'int')
    YardLineDirection = (39, 'str')
    IsPenaltyAccepted = (40, 'int')
    PenaltyTeam = (41, 'str')
    IsNoPlay = (42, 'int')
    PenaltyType = (43, 'str')
    PenaltyYards = (44, 'int')

    def __init__(self, val, type):
        self._value = val
        self._type = type

    @staticmethod
    def get_dtypes():
        # Create types from the nflsavant enum
        dtypes = {}
        for field in FootballHeader:
            dtypes[field.name] = field._type
        return dtypes