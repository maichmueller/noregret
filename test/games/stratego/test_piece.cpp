

#include <gtest/gtest.h>

#include <range/v3/all.hpp>

#include "fixtures.hpp"

using namespace stratego;

TEST(Piece, constructor)
{
   std::vector< Piece > pieces;
   pieces.emplace_back(Team::BLUE, Position{0, 1}, Token::scout);
   pieces.emplace_back(Team::BLUE, Position{101, 49}, Token::miner);
   pieces.emplace_back(Team::RED, Position{-43, 9 / 3}, Token::bomb);

   EXPECT_EQ(pieces[0].position(), (Position{0, 1}));
   EXPECT_EQ(pieces[1].position(), (Position{101, 49}));
   EXPECT_EQ(pieces[2].position(), (Position{-43, 3}));
   EXPECT_EQ(pieces[0].token(), Token::scout);
   EXPECT_EQ(pieces[1].token(), Token::miner);
   EXPECT_EQ(pieces[2].token(), Token::bomb);
   EXPECT_EQ(pieces[0].flag_hidden(), true);
   EXPECT_EQ(pieces[1].flag_hidden(), true);
   EXPECT_EQ(pieces[2].flag_hidden(), true);

   pieces[0].flag_hidden(false);
   EXPECT_EQ(pieces[0].flag_hidden(), false);

   EXPECT_EQ(pieces[0], Piece(Team::BLUE, Position{0, 1}, Token::scout, false));
   EXPECT_EQ(pieces[1], Piece(Team::BLUE, Position{101, 49}, Token::miner));
   EXPECT_EQ(pieces[2], Piece(Team::RED, Position{-43, 3}, Token::bomb));
}
