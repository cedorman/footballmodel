#
# Mapping between the Football data col headers and the number of the column
#
from enum import IntEnum


class FootballHeader(IntEnum):
    GameId = 0,
    GameDate = 1,
    Quarter = 2,
    Minute = 3,
    Second = 4,
    OffenseTeam = 5,
    DefenseTeam = 6,
    Down = 7,
    ToGo = 8,
    YardLine = 9,
    # =10,
    SeriesFirstDown = 11,
    # =12,
    NextScore = 13,
    Description = 14,
    TeamWin = 15,
    # =16,
    # =17,
    SeasonYear = 18,
    Yards = 19,
    Formation = 20,
    PlayType = 21,
    IsRush = 22,
    IsPass = 23,
    IsIncomplete = 24,
    IsTouchdown = 25,
    PassType = 26,
    IsSack = 27,
    IsChallenge = 28,
    IsChallengeReversed = 29,
    Challenger = 30,
    IsMeasurement = 31,
    IsInterception = 32,
    IsFumble = 33,
    IsPenalty = 34,
    IsTwoPointConversion = 35,
    IsTwoPointConversionSuccessful = 36,
    RushDirection = 37,
    YardLineFixed = 38,
    YardLineDirection = 39,
    IsPenaltyAccepted = 40,
    PenaltyTeam = 41,
    IsNoPlay = 42,
    PenaltyType = 43,
    PenaltyYards = 44
