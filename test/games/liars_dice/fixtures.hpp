
#ifndef NOR_LIARS_DICE_FIXTURES_HPP
#define NOR_LIARS_DICE_FIXTURES_HPP

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <vector>

#include "liars_dice/liars_dice.hpp"

/// the die-face count used throughout most tests (smallest paper benchmark)
constexpr uint8_t test_n_faces = 3;

struct LiarsDiceState: public ::testing::Test {
   liars_dice::DiceConfig config = liars_dice::DiceConfig(test_n_faces);
   liars_dice::State state = liars_dice::State(config);
};

/// applies the two chance rolls in seat order
inline void apply_rolls(liars_dice::State& state, uint8_t face_one, uint8_t face_two)
{
   state.apply_action(liars_dice::Roll{liars_dice::Player::one, face_one});
   state.apply_action(liars_dice::Roll{liars_dice::Player::two, face_two});
}

/// bid legality scenario: (optional standing bid, candidate bid, whether it is legal)
using BidLegalityCase = std::tuple< std::optional< liars_dice::Bid >, liars_dice::Bid, bool >;

class BidLegalityParamsF: public ::testing::TestWithParam< BidLegalityCase > {
  protected:
   liars_dice::DiceConfig config = liars_dice::DiceConfig(test_n_faces);
   liars_dice::State state = liars_dice::State(config);
};

/// challenge resolution truth table: (dice, standing bid, expected winning player)
using ChallengeResolutionCase = std::
   tuple< std::array< uint8_t, 2 >, liars_dice::Bid, liars_dice::Player >;

class ChallengeResolutionParamsF: public ::testing::TestWithParam< ChallengeResolutionCase > {
  protected:
   liars_dice::DiceConfig config = liars_dice::DiceConfig(test_n_faces);
   liars_dice::State state = liars_dice::State(config);
};

/// scripted terminality: (dice, action sequence, expected terminality)
using TerminalityCase = std::
   tuple< std::array< uint8_t, 2 >, std::vector< liars_dice::Action >, bool >;

class TerminalityParamsF: public ::testing::TestWithParam< TerminalityCase > {
  protected:
   liars_dice::DiceConfig config = liars_dice::DiceConfig(test_n_faces);
   liars_dice::State state = liars_dice::State(config);
};

#endif  // NOR_LIARS_DICE_FIXTURES_HPP
